import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io
import matplotlib
import pandas as pd
from modules.function import _parse_time_series
from modules.function import _compute_presence_matrix
from modules.function import _compute_monthly_summary
from modules.function import _weekday_total_summary
from modules.function import _compute_normalized_heatmap

# Constants

_WEEKDAY_ORDER = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def month_analysis():
    """Main function for monthly analysis visualization."""
    # Apply custom CSS
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] {
            max-width: 100px;
        }
        div[role="button"][aria-expanded] {
            padding: 0.25rem 0.5rem;
        }
        div[data-testid="stExpander"] > div {
            padding: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Check if data exists
    if 'crna_data' not in st.session_state:
        st.info("Please upload a file to begin.")
        return

    # Initialize DataFrame
    df = st.session_state['crna_data']
    
    # Convert pandas DataFrame to polars if needed
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    df = df.clone()
    
    # Standardize Date column
    df = df.with_columns([
        pl.col('Date')
            .cast(pl.String)
            .str.strptime(pl.Date, format='%Y/%m/%d')
            .alias('Date')
    ])
    
    # Add weekday column
    df = df.with_columns([
        pl.col('Date').dt.strftime('%A').alias('weekday')
    ])

    # —— 2. Monthly summary (with cache + spinner) —— #
    with st.spinner("Calculating monthly summary…"):
        monthly_summary = _compute_monthly_summary(df)

    # Handle title for pie chart (only initialize once)
    title1 = st.text_input("Pie Chart Title", "Number of Record for Month", key="title1")

    # Create pie chart
    fig1 = px.pie(
        monthly_summary.to_pandas(),
        values='Count',
        names='MonthLabel',
        hole=0.3,
        category_orders={"MonthLabel": monthly_summary.get_column('MonthLabel').to_list()},
        labels={'MonthLabel': 'Month', 'Count': 'Demand'},
        custom_data=['Count']
    )
    
    # Update all traces and layout at once
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
        margin=dict(t=50, l=20, r=20, b=50),
        height=450
    )

    # —— 3. Compute Presence matrix (with cache + spinner) —— #
    with st.spinner("Computing presence matrix, may take a few seconds…"):
        output = _compute_presence_matrix(df)
    start_date = output.get_column('Date').min().strftime('%Y-%m-%d')
    end_date = output.get_column('Date').max().strftime('%Y-%m-%d')
    # —— 4. Aggregate 'Total' by weekday (with cache + spinner) —— #
    with st.spinner("Aggregating total demand by weekday…"):
        df2 = _weekday_total_summary(output, start_date, end_date)

    df2 = df2.to_pandas().set_index('weekday').reindex(_WEEKDAY_ORDER).reset_index()

    title2 = st.text_input("Bar Chart Title", "Average Hour Demand by Weekday ", key="title2")

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
    fig2.update_layout(
        title={'text': title2, 'x': 0.5, 'xanchor': 'center'}
    )
    fig2.update_traces(texttemplate='%{text:.0f}')

    # —— 5. Normalized heatmap (with cache + spinner) —— #
    # Use the actual data range for start_date/end_date

    
    with st.spinner("Calculating normalized heatmap data…"):
        agg_df = _compute_normalized_heatmap(output, start_date, end_date)

    title3 = st.text_input("Heatmap Title", "Normalized Demand Heatmap", key="title3")

    #theme_mode = st.sidebar.radio("Theme", ["Light", "Dark"])
    #is_dark = (theme_mode == "Dark")
    
    bg_color = st.get_option("theme.backgroundColor")
    is_dark = (bg_color is not None and bg_color.lower() == "#0f1117")

    fig3, ax = plt.subplots(figsize=(20, 5))
    df_plot = agg_df.to_pandas().set_index('weekday').reindex(_WEEKDAY_ORDER)
    if is_dark:
        fig3.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        cbar_kws = {"format": "%d", "shrink": 0.98}
        hm = sns.heatmap(df_plot, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax, cbar_kws=cbar_kws)
        # 设置所有文本为白色
        ax.set_title(
            title3,
            fontdict={'fontsize': 18, 'fontweight': 'bold', 'color': 'white'},
            loc='center',
            pad=20
        )
        ax.set_ylabel("DOW", fontsize=14, color="white")
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        # 设置annot文本为白色
        for t in hm.texts:
            t.set_color("white")
        # 设置colorbar刻度和label为白色
        cbar = hm.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        cbar.set_label(cbar.ax.get_ylabel(), color='white')
        cbar.ax.yaxis.label.set_color('white')
    else:
        hm = sns.heatmap(df_plot, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
        ax.set_title(
            title3,
            fontdict={'fontsize': 18, 'fontweight': 'bold'},
            loc='center',
            pad=20
        )
        ax.set_ylabel("DOW", fontsize=14)
    plt.tight_layout()

    # —— 6. Display pie chart + bar chart in one row —— #
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        with st.expander("💾 Save ", expanded=False):
            buf1 = io.BytesIO()
            fig1.write_image(buf1, format="png", scale=2)
            st.download_button(
                label="🏞️ PNG",
                data=buf1.getvalue(),
                file_name=f"{title1}.png",
                mime="image/png"
            )
            csv1 = monthly_summary.write_csv().encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv1,
                file_name=f"{title1}.csv",
                mime="text/csv"
            )
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("💾 Save ", expanded=False):
            buf2 = io.BytesIO()
            fig2.write_image(buf2, format="png", scale=2)
            st.download_button(
                label="🏞️ PNG",
                data=buf2.getvalue(),
                file_name=f"{title2}.png",
                mime="image/png"
            )
            csv2 = df2.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv2,
                file_name=f"{title2}.csv",
                mime="text/csv"
            )

    # —— 7. Display heatmap in a new row —— #
    st.pyplot(fig3)
    with st.expander("💾 Save ", expanded=False):
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            label="🏞️ PNG",
            data=buf3.getvalue(),
            file_name=f"{title3}.png",
            mime="image/png"
        )
        flattened_data = df_plot.values.flatten(order='C')
        df_transformed = pd.DataFrame(flattened_data)
        csv3 = df_transformed.to_csv(index=False, header=['Demand']).encode("utf-8")
        st.download_button(
            label="📥 CSV",
            data=csv3,
            file_name=f"{title3}.csv",
            mime="text/csv"
        )

    # —— 8. 'Back' and 'Go to Week Analysis' buttons —— #
    back_col, _, week_col = st.columns([1, 8, 1])
    with back_col:
        if st.button("⬅️ Back"):
            # Clear all related session_state keys
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

