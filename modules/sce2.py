import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io


_REFERENCE_DATE = "1900-01-01 "

# 预先生成 24 小时区间的起止时间，避免每次调用 get_bins
_TIME_BIN_START = pd.to_datetime([f"{_REFERENCE_DATE}{h:02d}:00" for h in range(24)])
_TIME_BIN_END   = pd.to_datetime([f"{_REFERENCE_DATE}{(h+1)%24:02d}:00" for h in range(24)])

_WEEKDAY_ORDER = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def _parse_time_series(series: pd.Series) -> pd.Series:
    """
    将"HH:MM"格式的字符串一次性转为 Timestamp (1900-01-01 HH:MM)。
    """
    return pd.to_datetime(_REFERENCE_DATE + series.astype(str))


@st.cache_data(show_spinner=False)
def _compute_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    向量化地将原始 df（需含 'In Room', 'Out Room', 'Count' 三列）映射成包含 24 列（0–23 小时）的 DataFrame。
    返回值：原始 df（去掉 NaN）+ 24 列"在房人数"。
    """
    temp = (
        df
        .dropna(subset=['In Room', 'Out Room'])
        .reset_index(drop=True)
        .copy()
    )
    temp['In Room']  = temp['In Room'].astype(str)
    temp['Out Room'] = temp['Out Room'].astype(str)

    # 向量化解析 "HH:MM" 到 Timestamp
    in_times  = _parse_time_series(temp['In Room'])
    out_times = _parse_time_series(temp['Out Room'])

    # 如果 out < in 则加一天
    wrap_mask = out_times < in_times
    out_times = out_times.where(~wrap_mask, out_times + pd.Timedelta(days=1))

    # 利用广播生成 (N,24) 布尔矩阵，表示每行是否在对应小时区间"存在"
    overlap = (
        (in_times.values.reshape(-1, 1) < _TIME_BIN_END.values.reshape(1, -1)) &
        (out_times.values.reshape(-1, 1) > _TIME_BIN_START.values.reshape(1, -1))
    )

    # "Count" 作为权重
    counts = temp['Count'].astype(int).values.reshape(-1, 1)
    presence_matrix = overlap * counts  # 形状 (N,24)，元素为 0 或 Count

    # 转成 DataFrame，列名 0–23
    presence_df = pd.DataFrame(
        presence_matrix,
        index=temp.index,
        columns=list(range(24))
    )

    # 最终 DataFrame = 原始行信息 + 24 列在房数据
    out_df = pd.concat([temp.reset_index(drop=True), presence_df.reset_index(drop=True)], axis=1)
    return out_df


@st.cache_data(show_spinner=False)
def _compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 'Month'（Period）列汇总 'Count' 并生成 'MonthLabel'，不改变原始 df。
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_summary = df.groupby('Month')['Count'].sum().reset_index()
    
    # 确保月份按时间顺序排序
    monthly_summary = monthly_summary.sort_values('Month')
    
    monthly_summary['MonthLabel'] = (
        monthly_summary['Month']
          .dt.to_timestamp()
          .dt.strftime('%b %y')     # 格式化成 'Jan 25'
    )

    return monthly_summary


@st.cache_data(show_spinner=False)
def _weekday_total_summary(df_with_time: pd.DataFrame) -> pd.DataFrame:
    """
    对已经包含 24 列（0–23）和 'weekday' 字段的 DataFrame 做 groupby，返回 7 行×1 列 'Total'。
    """
    summed = df_with_time.groupby('weekday')[list(range(24))].sum()
    summed['Total'] = summed.sum(axis=1)
    return summed.reindex(_WEEKDAY_ORDER)[['Total']]


@st.cache_data(show_spinner=False)
def _compute_normalized_heatmap(df_with_time: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    1. 按 'weekday' × 24 列求和，得到 raw_counts (7×24)。
    2. 计算 start_date 到 end_date 之间每天出现的次数 (7,)。
    3. 用 raw_counts 除以对应天数并四舍五入返回 int 格式热力图矩阵。
    """
    raw = df_with_time.groupby('weekday')[list(range(24))].sum().reindex(_WEEKDAY_ORDER)
    drange = pd.date_range(start=start_date, end=end_date)
    day_counts = drange.day_name().value_counts().reindex(_WEEKDAY_ORDER).fillna(0).astype(int)
    normalized = (raw.div(day_counts, axis=0)).round().astype(int)
    return normalized.fillna(0)


def month_analysis():
    st.markdown(
        """
        <style>
        /* 限制 expander 最大宽度 */
        div[data-testid="stExpander"] {
            max-width: 100px;
        }
        /* 缩小标题栏内边距 */
        div[role="button"][aria-expanded] {
            padding: 0.25rem 0.5rem;
        }
        /* 缩小展开内容的内边距 */
        div[data-testid="stExpander"] > div {
            padding: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # —— 1. 检查是否已上传并处理数据 —— #
    if 'crna_data' not in st.session_state:
        st.info("Please upload a file to begin.")
        return

    # 复制 df，避免真实 session_state 被修改
    df = st.session_state['crna_data'].copy()
    # 统一做一次日期与 weekday
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    df['weekday'] = df['Date'].dt.day_name()

    # —— 2. 每月汇总 (带缓存 + Spinner) —— #
    with st.spinner("正在计算每月汇总…"):
        monthly_summary = _compute_monthly_summary(df)

    # 处理标题（只保留一次）
    title1 = st.text_input("Pie Chart Title", "Monthly Demand", key="title1")


    # 创建饼图
    fig1 = px.pie(
        monthly_summary,
        values='Count',
        names='MonthLabel',
        hole=0.3,
        category_orders={"MonthLabel": monthly_summary['MonthLabel'].tolist()},
        labels={'MonthLabel': 'Month', 'Count': 'Demand'},
        custom_data=['Count']
    )
    
    # 一次性更新所有布局和标签
    fig1.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="Month: %{label}<br>Demand: %{value}<extra></extra>"
    )
    
    fig1.update_layout(
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.1, yanchor="top"
        ),
        template="plotly_white",
        title={"text": title1, "x": 0.5, "xanchor": "center"},
        margin=dict(t=50, l=20, r=20, b=50),  # 添加适当的边距
        height=450  # 适当增加高度
    )

    # —— 3. 计算 Presence 矩阵 (带缓存 + Spinner) —— #
    with st.spinner("正在计算每行在房时段数据，可能需要几秒…"):
        output = _compute_presence_matrix(df)

    # —— 4. 按 weekday 汇总"Total" (带缓存 + Spinner) —— #
    with st.spinner("正在汇总每周每日总需求…"):
        df2 = _weekday_total_summary(output)
    df2 = df2.reset_index()  # 将 weekday 转为列

    title2 = st.text_input("Bar Chart Title", "Total Month Demand by Weekday", key="title2")

    fig2 = px.bar(
        df2,
        x='weekday',
        y='Total',
        labels={'weekday': 'Day of Week', 'Total': 'Total Demand'},
        text='Total',
        template='plotly_white'
    )
    single_color = px.colors.qualitative.Plotly[0]
    fig2.update_traces(marker_color=single_color)
    fig2.update_layout(title={'text': title2, 'x': 0.5, 'xanchor': 'center'})

    # —— 5. 归一化热力图 (带缓存 + Spinner) —— #
    # 此处使用实际数据范围作为 start_date/end_date
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date   = df['Date'].max().strftime('%Y-%m-%d')
    with st.spinner("正在计算归一化热力图数据…"):
        agg_df = _compute_normalized_heatmap(output, start_date, end_date)

    title3 = st.text_input("Heatmap Title", "Normalized Demand Heatmap", key="title3")

    fig3, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(agg_df, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
    ax.set_title(title3,fontdict={'fontsize': 18, 'fontweight': 'bold'},loc='center', pad=20)
    ax.set_ylabel("DOW", fontsize=14)
    plt.tight_layout()

    # —— 6. 展示饼图 + 下载按钮 —— #
    st.subheader("Monthly Demand")
    st.plotly_chart(fig1, use_container_width=True)
    col_l, col_c, col_r = st.columns([3, 1, 3])
    with col_c:
        with st.expander("💾 Save ", expanded=False):
            buf1 = io.BytesIO()
            fig1.write_image(buf1, format="png", scale=2)
            st.download_button(
                label="🏞️ PNG",
                data=buf1.getvalue(),
                file_name="monthly_distribution.png",
                mime="image/png"
            )
            csv1 = monthly_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv1,
                file_name="monthly_distribution.csv",
                mime="text/csv"
            )

    # —— 7. 展示柱状图 + 下载按钮 —— #
    st.subheader("Total Month Demand by Weekday")
    st.plotly_chart(fig2, use_container_width=True)
    col_l, col_c, col_r = st.columns([3, 1, 3])
    with col_c:
        with st.expander("💾 Save ", expanded=False):
            buf2 = io.BytesIO()
            fig2.write_image(buf2, format="png", scale=2)
            st.download_button(
                label="🏞️ PNG",
                data=buf2.getvalue(),
                file_name="weekday_summary.png",
                mime="image/png"
            )
            csv2 = df2.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv2,
                file_name="weekday_summary.csv",
                mime="text/csv"
            )

    # —— 8. 展示热力图 + 下载按钮 —— #
    st.subheader("Normalized Demand Heatmap")
    st.pyplot(fig3)
    col_l, col_c, col_r = st.columns([3, 1, 3])
    with col_c:
        with st.expander("💾 Save ", expanded=False):
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format="png", dpi=150, bbox_inches="tight")
            st.download_button(
                label="🏞️ PNG",
                data=buf3.getvalue(),
                file_name=f"{title3}.png",
                mime="image/png"
            )
            csv3 = agg_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv3,
                file_name="normalized_heatmap.csv",
                mime="text/csv"
            )

    # —— 9. "Back" 与 "Go to Week Analysis" 按钮 —— #
    back_col, _, week_col = st.columns([1, 8, 1])
    with back_col:
        if st.button("⬅️ Back"):
            # 清除所有相关的session state
            keys_to_remove = [
                "crna_data",
                "analysis_type",
                "analysis_view",
                "col_error",
                "duplicate_error",
                "format_error",
                "format_error_msg",
                "uploaded_file",
                "crna_uploader"
            ]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.page = "Upload"
            st.rerun()

    with week_col:
        if st.button("🔍 Go to Week Analysis"):
            if st.session_state.analysis_type == "presence":
                st.session_state.page = "sce3"
            else:
                st.session_state.page = "sce5"
            st.rerun()
