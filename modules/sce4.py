import polars as pl
import pandas as _pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io
import numpy as np
import pandas as pd
from modules.function import _parse_time_series
from modules.function import _compute_monthly_summary
from modules.function import _weekday_total_summary
from modules.function import _compute_normalized_heatmap
from modules.function import _compute_duration_matrix

# Constants

_WEEKDAY_ORDER = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def duration_month_analysis():
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
    if isinstance(df, _pd.DataFrame):
        df = pl.from_pandas(df)
    df = df.clone()
    
    # Standardize Date column
    df = df.with_columns([
        # First standardize the format by replacing all separators with '/'
        pl.col('Date').str.replace_all(r'[-.]', '/').alias('Date')
    ]).with_columns([
        # Then try both date formats with strict=False
        pl.when(pl.col('Date').str.strptime(pl.Date, format='%Y/%m/%d', strict=False).is_not_null())
            .then(pl.col('Date').str.strptime(pl.Date, format='%Y/%m/%d', strict=False))
            .otherwise(
                pl.col('Date').str.strptime(pl.Date, format='%m/%d/%Y', strict=False)
            )
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
        labels={'MonthLabel': 'Month', 'Count': 'Duration (hours)'},
        custom_data=['Count']
    )
    
    # Update all traces and layout at once
    fig1.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="Month: %{label}<br>Duration: %{value} hours<extra></extra>"
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

    # —— 3. Compute Duration matrix (with cache + spinner) —— #
    with st.spinner("Computing duration matrix, may take a few seconds…"):
        output = _compute_duration_matrix(df)

    # —— 4. Aggregate 'Total' by weekday (with cache + spinner) —— #
    # with st.spinner("Aggregating total duration by weekday…"):
    #     start_date = output.get_column('Date').min().strftime('%Y-%m-%d')
    #     end_date = output.get_column('Date').max().strftime('%Y-%m-%d')
    #     df2 = _weekday_total_summary(output, start_date, end_date)

    # df2_plot = df2.to_pandas().set_index('weekday').reindex(_WEEKDAY_ORDER).reset_index()
    # title2 = st.text_input("Bar Chart Title", "Average Hour Demand by Weekday", key="title2")

    # fig2 = px.bar(
    #     df2_plot,
    #     x='weekday',
    #     y='Total',
    #     labels={'weekday': 'Day of Week', 'Total': 'Total Duration (hours)'},
    #     text='Total',
    #     template='plotly_white'
    # )
    # single_color = px.colors.qualitative.Plotly[0]
    # fig2.update_traces(marker_color=single_color)
    # fig2.update_layout(title={'text': title2, 'x': 0.5, 'xanchor': 'center'})
    # fig2.update_traces(texttemplate='%{text:.0f}')

    # —— 5. Normalized heatmap (with cache + spinner) —— #
    # Use the actual data range for start_date/end_date
    start_date = output.get_column('Date').min().strftime('%Y-%m-%d')
    end_date = output.get_column('Date').max().strftime('%Y-%m-%d')
    
    with st.spinner("Calculating normalized heatmap data…"):
        agg_df = _compute_normalized_heatmap(output, start_date, end_date)

    df_plot = agg_df.to_pandas().set_index('weekday').reindex(_WEEKDAY_ORDER)
    title3 = st.text_input("Heatmap Title", "Normalized Duration Heatmap", key="title3")

    fig3, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(df_plot, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
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
            csv2 = df2.write_csv().encode("utf-8")
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

    # —— 9. 'Back' and 'Go to Week Analysis' buttons —— #
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
