import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io, zipfile

# —— 全局常量与预计算 —— #
_REFERENCE_DATE = "1900-01-01 "
_TIME_BIN_START = pd.to_datetime([f"{_REFERENCE_DATE}{h:02d}:00" for h in range(24)])
_TIME_BIN_END   = pd.to_datetime([f"{_REFERENCE_DATE}{(h+1)%24:02d}:00" for h in range(24)])
_WEEKDAY_ORDER  = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
_WEEKS_LIST     = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']


def _parse_time_series(series: pd.Series) -> pd.Series:
    """将"HH:MM"格式字符串一次性转为 Timestamp (1900-01-01 HH:MM)。"""
    return pd.to_datetime(_REFERENCE_DATE + series.astype(str))


@st.cache_data(show_spinner=False)
def _compute_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    向量化地将原始 df（需含 'In Room','Out Room','Count'）映射成包含 24 列（0–23 小时）的 DataFrame。
    返回：原始行信息 + 24 列"在房人数"。
    """
    temp = (
        df
        .dropna(subset=['In Room', 'Out Room'])
        .reset_index(drop=True)
        .copy()
    )
    temp['In Room']  = temp['In Room'].astype(str)
    temp['Out Room'] = temp['Out Room'].astype(str)

    in_times  = _parse_time_series(temp['In Room'])
    out_times = _parse_time_series(temp['Out Room'])

    # 如果 out < in，则加一天
    wrap_mask = out_times < in_times
    out_times = out_times.where(~wrap_mask, out_times + pd.Timedelta(days=1))

    # 利用广播一次性生成 (N,24) 布尔矩阵，标记每行在对应小时区间是否有人
    overlap = (
        (in_times.values.reshape(-1, 1) < _TIME_BIN_END.values.reshape(1, -1)) &
        (out_times.values.reshape(-1, 1) > _TIME_BIN_START.values.reshape(1, -1))
    )
    counts = temp['Count'].astype(int).values.reshape(-1, 1)
    presence_matrix = overlap * counts  # 形状 (N,24)

    presence_df = pd.DataFrame(
        presence_matrix,
        index=temp.index,
        columns=list(range(24))
    )
    out_df = pd.concat([temp.reset_index(drop=True), presence_df.reset_index(drop=True)], axis=1)
    return out_df


@st.cache_data(show_spinner=False)
def _assign_month_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    为已有 'Date' 列的 DataFrame 添加 'week_of_month' 列（Week 1 到 Week 5）。
    要求：df['Date'] 已是 datetime 类型。
    """
    temp = df.copy()
    temp['Date'] = pd.to_datetime(temp['Date'])
    temp['day'] = temp['Date'].dt.day
    temp['week_of_month'] = pd.cut(
        temp['day'],
        bins=[0, 7, 14, 21, 28, 31],
        labels=_WEEKS_LIST,
        right=True
    )
    return temp


@st.cache_data(show_spinner=False)
def _compute_week_hm_data(df_with_time: pd.DataFrame, week_label: str) -> pd.DataFrame:
    """
    根据 week_label（'Week 1'…'Week 5'）过滤 df_with_time，然后生成 7×24 的热力图数据：
    1. groupby 'weekday' × 0–23，得到 raw_counts；
    2. 如果是 Week 5，则先 fillna(0)；
    3. 除以 3，round并转换为 int。
    返回排序好的 DataFrame，索引为 _WEEKDAY_ORDER，列为 0–23。
    """
    # 只保留该周的数据
    df_wk = df_with_time[df_with_time['week_of_month'] == week_label]
    df_wk['weekday'] = df_wk['Date'].dt.day_name()

    raw = df_wk.groupby('weekday')[list(range(24))].sum().reindex(_WEEKDAY_ORDER)
    if week_label == 'Week 5':
        raw = raw.fillna(0)

    hm_data = raw.div(3).round().astype(int)
    return hm_data


def week_analysis():
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

    if 'crna_data' not in st.session_state:
        st.info("Please upload a file to begin.")
        return

    df = st.session_state['crna_data'].copy()
    st.markdown("# Week Data Charts")

    # —— 1. 转换 Date 为 datetime —— #
    df['Date'] = pd.to_datetime(df['Date'])

    # —— 2. 计算 Presence 矩阵 (带缓存 + Spinner) —— #
    with st.spinner("正在计算在房时段（可能需要几秒）…"):
        output = _compute_presence_matrix(df)

    # —— 3. 为 output 添加 'week_of_month' 列 (带缓存) —— #
    weekfile_detail = _assign_month_week(output)

    # —— 4. 下拉框让用户选择要查看的 Week —— #
    selected_wk = st.selectbox("📊 Select Week to Display", _WEEKS_LIST)

    # —— 5. 根据选择的 Week 计算热力图数据 (带缓存 + Spinner) —— #
    with st.spinner(f"正在计算 {selected_wk} 的热力图数据…"):
        hm_data = _compute_week_hm_data(weekfile_detail, selected_wk)

    # —— 6. 用户自定义标题 —— #
    default_title = f"Demand for {selected_wk}"
    title_input = st.text_input(
        label=f"{selected_wk} Chart Title",
        value=default_title,
        key=f"title_{selected_wk}"
    )
    if len(title_input) > 40:
        title_input = title_input[:37] + "..."

    # —— 7. 绘制这周的热力图 —— #
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(hm_data, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
    ax.set_title(title_input, fontdict={'fontsize': 18, 'fontweight': 'bold'}, loc='center', pad=20)
    ax.set_ylabel("DOW", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # —— 8. 左侧：下载当前 Week 的 PNG/CSV —— #
    col_l, _, col_r = st.columns([1, 8, 1])
    with col_l:
        with st.expander(f"💾 Save {selected_wk}", expanded=False):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            st.download_button(
                label="🏞️ PNG",
                data=buf.getvalue(),
                file_name=f"{selected_wk}_heatmap.png",
                mime="image/png"
            )
            csv_bytes = hm_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv_bytes,
                file_name=f"{selected_wk}_heatmap_data.csv",
                mime="text/csv"
            )

    # —— 9. 右侧：下载所有 Weeks 的 PNG/CSV —— #
    with col_r:
        with st.expander("💾 Save All Weeks", expanded=False):
            # ZIP 所有 PNG
            png_zip = io.BytesIO()
            with zipfile.ZipFile(png_zip, mode="w") as zf:
                for wk in _WEEKS_LIST:
                    df_hm = _compute_week_hm_data(weekfile_detail, wk)
                    fig_w, ax_w = plt.subplots(figsize=(10, 3))
                    sns.heatmap(df_hm, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax_w)
                    ax_w.set_title(f"Average Demand for {wk}", loc="center")
                    plt.tight_layout()
                    buf_w = io.BytesIO()
                    fig_w.savefig(buf_w, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig_w)
                    zf.writestr(f"{wk}_heatmap.png", buf_w.getvalue())
            png_zip.seek(0)
            st.download_button(
                label="🏞️ PNGs",
                data=png_zip.getvalue(),
                file_name="all_weeks_heatmaps.zip",
                mime="application/zip"
            )

            # ZIP 所有 CSV
            csv_zip = io.BytesIO()
            with zipfile.ZipFile(csv_zip, mode="w") as zf2:
                for wk in _WEEKS_LIST:
                    df_hm = _compute_week_hm_data(weekfile_detail, wk)
                    csv_bytes = df_hm.to_csv(index=False).encode("utf-8")
                    zf2.writestr(f"{wk}_heatmap_data.csv", csv_bytes)
            csv_zip.seek(0)
            st.download_button(
                label="📥 CSVs",
                data=csv_zip.getvalue(),
                file_name="all_weeks_heatmap_data.zip",
                mime="application/zip"
            )

    # —— 10. Back 与 Go to Month Analysis 按钮 —— #
    back_col, _, month_col = st.columns([1, 8, 1])
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

    with month_col:
        if st.button("🔍 Go to month Analysis"):
            if st.session_state.analysis_type == "presence":
                st.session_state.page = "sce2"
            else:
                st.session_state.page = "sce4"
            st.rerun()
