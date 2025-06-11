import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
from pathlib import Path
from modules.layout import set_narrow,set_fullwidth

def uploadstep1_page():
    #upload file
    uploaded = st.session_state.get("uploaded_file", None)
    if uploaded is None:
        st.error("Please upload a file first.")
        return

    try:
        name = uploaded.name.lower()
        if name.endswith('.csv'):
            df = pl.read_csv(uploaded)
        elif name.endswith(('.xls', '.xlsx')):
            df = pl.read_excel(uploaded)
        else:
            st.error("❌ Unsupported file format. Please upload a CSV or Excel file.")
            return
    except pl.NoDataError:
        st.error("❌ The file appears to be empty.")
        return
    except Exception as e:
        st.error(f"❌ Failed to read file: {str(e)}")
        return

    #select columns
    st.title("Select Your Columns")
    with st.form("col_selector_form"):
        raw_cols = df.columns
        placeholder = [""]  
        cols = placeholder + raw_cols

        cd = st.selectbox("Select **Date** column", cols, index=0)
        ci = st.selectbox("Select **In Time** column", cols, index=0)
        co = st.selectbox("Select **Out Time** column", cols, index=0)
        cc = st.selectbox("Select **Count** column [Optional]", cols, index=0)

        submit_cols = st.form_submit_button("✅ Process & Save ")
    
    #check the data
    if submit_cols:
        # empty value
        if "" in (cd, ci, co):
            st.warning("⚠ Please select first three columns to proceed.")
            return

        # duplicate columns
        selected_cols = [cd, ci, co] + ([cc] if cc and cc !="" else [])
        if len(selected_cols) != len(set(selected_cols)):
            st.error("❌ Duplicate columns selected! Please select different columns for each field.")
            return
          
        # Create new dataframe with selected columns
        df2 = df.select([cd, ci, co])
        # Handle Count column
        df2 = df2.drop_nulls(subset=["Date", "In Room", "Out Room"])

        if cc and cc != "":

            count_series = df.filter(
                ~pl.col(cd).is_null() & ~pl.col(ci).is_null() & ~pl.col(co).is_null()
            ).select(cc).to_series()
            df2 = df2.with_columns([
                count_series.alias('Count')
            ])
        else:
            df2 = df2.with_columns([
                pl.lit(1).alias('Count')
            ])
            count_error_count = 0

        # normalize data
        original_dates = df2.get_column('Date').to_list()
        df2 = df2.with_columns([
            pl.col('Date').cast(pl.String).str.strptime(pl.Datetime, None).alias('Date')
        ])
        invalid_dates = df2.with_row_count("row_nr").filter(pl.col('Date').is_null()).get_column("row_nr").to_list()
        date_error_count = len(invalid_dates)
        
        if invalid_dates:
            error_details = [f"Row {idx+1}: '{original_dates[idx]}'" for idx in invalid_dates[:5]]
            error_msg = "\n".join(error_details)
            if len(invalid_dates) > 5:
                error_msg += f"\n... and {len(invalid_dates) - 5} more errors"
            st.warning(f"⚠️ Found {date_error_count} invalid date(s):\n{error_msg}")
            # Restore original dates for invalid entries
            df2 = df2.with_columns([
                pl.when(pl.col('Date').is_null())
                .then(pl.Series(name='Date', values=original_dates))
                .otherwise(pl.col('Date'))
                .alias('Date')
            ])
        
        # Format time strings to HH:MM
        def format_time(time_str):
            if not time_str or time_str == '':
                return ''
            try:
                # Remove any date part if present
                if ' ' in str(time_str):
                    time_str = str(time_str).split(' ')[-1]
                
                # Handle different time formats
                time_str = str(time_str).strip()
                if ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) >= 2:
                        hours = int(parts[0].strip())
                        minutes = int(parts[1].strip())
                        if 0 <= hours < 24 and 0 <= minutes < 60:
                            return f"{hours:02d}:{minutes:02d}"
                return time_str
            except:
                return str(time_str)

        # Check time formats
        original_in_times = df2.get_column('In Room').cast(pl.String).to_list()
        original_out_times = df2.get_column('Out Room').cast(pl.String).to_list()
        
        in_time_errors = []
        out_time_errors = []
        
        for idx, (in_time, out_time) in enumerate(zip(original_in_times, original_out_times)):
            try:
                # Remove any date part if present
                in_time = str(in_time).split(' ')[-1] if ' ' in str(in_time) else str(in_time)
                # Check time format
                parts = in_time.strip().split(':')
                if not (len(parts) >= 2 and 0 <= int(parts[0].strip()) < 24 and 0 <= int(parts[1].strip()) < 60):
                    in_time_errors.append(idx)
            except:
                in_time_errors.append(idx)
            
            try:
                # Remove any date part if present
                out_time = str(out_time).split(' ')[-1] if ' ' in str(out_time) else str(out_time)
                # Check time format
                parts = out_time.strip().split(':')
                if not (len(parts) >= 2 and 0 <= int(parts[0].strip()) < 24 and 0 <= int(parts[1].strip()) < 60):
                    out_time_errors.append(idx)
            except:
                out_time_errors.append(idx)

        in_time_error_count = len(in_time_errors)
        out_time_error_count = len(out_time_errors)

        if in_time_errors:
            error_details = [f"Row {idx+1}: '{original_in_times[idx]}'" for idx in in_time_errors[:5]]
            error_msg = "\n".join(error_details)
            if len(in_time_errors) > 5:
                error_msg += f"\n... and {len(in_time_errors) - 5} more errors"
            st.warning(f"⚠️ Found {in_time_error_count} invalid In Room time(s):\n{error_msg}")

        if out_time_errors:
            error_details = [f"Row {idx+1}: '{original_out_times[idx]}'" for idx in out_time_errors[:5]]
            error_msg = "\n".join(error_details)
            if len(out_time_errors) > 5:
                error_msg += f"\n... and {len(out_time_errors) - 5} more errors"
            st.warning(f"⚠️ Found {out_time_error_count} invalid Out Room time(s):\n{error_msg}")

        # Format times
        df2 = df2.with_columns([
            pl.col('In Room').cast(pl.String).map_elements(format_time).alias('In Room'),
            pl.col('Out Room').cast(pl.String).map_elements(format_time).alias('Out Room')
        ])

        # Handle Count column
        if cc and cc!="":
            original_counts = df2.get_column('Count').to_list()
            df2 = df2.with_columns(pl.col('Count').cast(pl.Int64, strict=False).alias('Count'))
            invalid_counts = df2.with_row_count("row_nr").filter(pl.col('Count').is_null()).get_column("row_nr").to_list()
            count_error_count = len(invalid_counts)
            
            if invalid_counts:
                error_details = [f"Row {idx+1}: '{original_counts[idx]}'" for idx in invalid_counts[:5]]
                error_msg = "\n".join(error_details)
                if len(invalid_counts) > 5:
                    error_msg += f"\n... and {len(invalid_counts) - 5} more errors"
                st.warning(f"⚠️ Found {count_error_count} invalid count value(s). These will be set to 1:\n{error_msg}")

        # Fill null counts with 1
        df2 = df2.with_columns(pl.col('Count').fill_null(1).cast(pl.Int64).alias('Count'))

        # Format dates
        if not df2.get_column('Date').is_null().all():
            df2 = df2.with_columns([
                pl.col('Date').cast(pl.Datetime).dt.strftime('%Y/%m/%d').alias('Date')
            ])
            df2 = df2.with_columns([
                pl.col('Date').fill_null('').alias('Date')
            ])

        # Sum the errors
        total_errors = date_error_count + in_time_error_count + out_time_error_count + count_error_count
        
        if total_errors > 0: 
            st.error(f"⚠️ Found {total_errors} error(s) in your data!")
            st.warning("Please fix the errors in your data before proceeding.")
        
            col1, col2 = st.columns(2)
            with col1:
                if st.button("❌ Cancel Processing"):
                
                    for key in st.session_state.keys():
            
                        if key != "page":
                            del st.session_state[key]
                    st.session_state.page = "Upload"
                    st.rerun()
            return 

        # Store polars DataFrame directly without converting to pandas
        st.session_state["crna_data"] = df2

        if total_errors > 0:
            st.markdown("### 📊 Error Statistics Summary")
            st.markdown(f"""
            **Found {total_errors} total errors:**
            - Date format errors: {date_error_count}
            - In Room time errors: {in_time_error_count}
            - Out Room time errors: {out_time_error_count}
            - Count value errors: {count_error_count}
            """)
            st.markdown("---")
        
        st.success("✅ Columns selected and standardized.")
        st.write("🔍 Data Preview (cleaned):", df2.head())

    if st.session_state.get("col_error"):
        st.warning("⚠ Please select all four columns to proceed.")
        if st.button("OK", key="col_error_ok"):
            st.session_state.pop("col_error")
        return

    if st.session_state.get("duplicate_error"):
        st.error("❌ Duplicate columns selected! Please select different columns for each field.")
        if st.button("OK", key="duplicate_error_ok"):
            st.session_state.pop("duplicate_error")
        return

    if st.session_state.get("format_error"):
        st.error("❌ Invalid data format!")
        st.warning("Please ensure the Count column contains only numbers (empty values are allowed)")
        if st.button("OK", key="format_error_ok"):
            st.session_state.pop("format_error")
            st.session_state.pop("format_error_msg", None)
        return

    if "crna_data" not in st.session_state:
        return

    st.markdown("---")
    st.success("✅ Ready for analysis.")

    # --------- ：Presence or Duration ---------  
    st.markdown("### Choose your Tracking Type")
    choice = st.radio(
        "", 
        ("📈 Presence", "⏱ Duration"),
        index=0 if st.session_state.get("analysis_type","presence")=="presence" else 1,
        key="analysis_type_radio"
    )
    if choice.startswith("📈"):
        st.session_state.analysis_type = "presence"
    else:
        st.session_state.analysis_type = "duration"

    # --------- ：Month or Week ---------  
    st.markdown("### Choose Your Analysis Scope")
    view_choice = st.radio(
        "",
        ("📅 Month Analysis", "🗓️ Week Analysis"),
        index=0 if st.session_state.get("analysis_view","month")=="month" else 1,
        key="analysis_view_radio"
    )
    if view_choice.startswith("📅"):
        st.session_state.analysis_view = "month"
    else:
        st.session_state.analysis_view = "week"
 
    btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
    with btn_col1:
        if st.button('Back'):

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
                st.session_state.pop(key, None)
            st.session_state.page = "Upload"
            st.rerun()
    with btn_col3:
        if st.button("Next ➡️"):
            mode = st.session_state.analysis_type    # "presence" or "duration"
            view = st.session_state.analysis_view      # "month" or "week"

            if mode == "presence" and view == "month":
                st.session_state.page = "sce2"
            elif mode == "presence" and view == "week":
                st.session_state.page = "sce3"
            elif mode == "duration" and view == "month":
                st.session_state.page = "sce4"
            else:  # duration + week
                st.session_state.page = "sce5"
            st.rerun()