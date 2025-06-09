import pandas as pd
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
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        try:
            df = pd.read_excel(uploaded)
        except Exception as excel_e:
            st.error("Please make sure it is either a valid CSV or Excel file.")
            return

    #select columns
    st.title("Select Your Columns")
    with st.form("col_selector_form"):
        raw_cols = df.columns.tolist()
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
        
        df2 = df[[cd, ci, co]].reset_index(drop=True)  
        df2.columns = ['Date','In Room','Out Room']
        if cc and cc!="":
            df2['Count']=df[cc]
            non_empty_counts=df2['count'].dropna()
            numeric_mask = pd.to_numeric(non_empty_counts, errors='coerce').isna()
            error_indices = non_empty_counts[numeric_mask].index.tolist()
            count_error_count = len(error_indices)
        else:
            df2['Count']=1
            count_error_count=0

    
        # normalize data
        original_dates = df2['Date'].copy()
        df2['Date'] = pd.to_datetime(df2['Date'].astype(str), errors='coerce')
        invalid_dates = df2[df2['Date'].isna()].index.tolist()
        date_error_count = len(invalid_dates)
        
        if invalid_dates:
            error_details = [f"Row {idx+1}: '{original_dates[idx]}'" for idx in invalid_dates[:5]]
            error_msg = "\n".join(error_details)
            if len(invalid_dates) > 5:
                error_msg += f"\n... and {len(invalid_dates) - 5} more errors"
            st.warning(f"⚠️ Found {date_error_count} invalid date(s):\n{error_msg}")
            df2.loc[invalid_dates, 'Date'] = original_dates[invalid_dates]
        
        #  HH:MM
        def format_time(time_str):
            if pd.isna(time_str) or time_str == '':
                return ''
            try:
                dt = pd.to_datetime(str(time_str), format=None)
                return dt.strftime('%H:%M')
            except:
                try:
                    # add a virtual data
                    dt = pd.to_datetime(f"2000-01-01 {str(time_str)}")
                    return dt.strftime('%H:%M')
                except:
                    return str(time_str)  
        # check the format for "In Room" and "Out Room"
        original_in_times = df2['In Room'].copy()
        original_out_times = df2['Out Room'].copy()
        
        in_time_errors = []
        out_time_errors = []
        
        for idx, (in_time, out_time) in enumerate(zip(df2['In Room'], df2['Out Room'])):
            try:
                pd.to_datetime(str(in_time))
            except:
                in_time_errors.append(idx)
            try:
                pd.to_datetime(str(out_time))
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

        # process Count column
        if cc and cc!="":
            original_counts = df2['Count'].copy()
            df2['Count'] = pd.to_numeric(df2['Count'], errors='coerce')
            invalid_counts = df2[df2['Count'].isna()].index.tolist()
            count_error_count = len(invalid_counts)
            
            if invalid_counts:
                error_details = [f"Row {idx+1}: '{original_counts[idx]}'" for idx in invalid_counts[:5]]
                error_msg = "\n".join(error_details)
                if len(invalid_counts) > 5:
                    error_msg += f"\n... and {len(invalid_counts) - 5} more errors"
                st.warning(f"⚠️ Found {count_error_count} invalid count value(s). These will be set to 1:\n{error_msg}")

        # sum the errors
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
        
  
        if not df2['Date'].isna().all():  

            df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')

            df2['Date'] = df2['Date'].fillna(pd.NaT).dt.strftime('%Y/%m/%d')
            df2['Date'] = df2['Date'].fillna('') 
        
        df2['In Room'] = df2['In Room'].apply(format_time)
        df2['Out Room'] = df2['Out Room'].apply(format_time)
        df2['Count'] = df2['Count'].fillna(1).astype(int)


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