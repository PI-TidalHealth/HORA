import streamlit as st

def render_upload_page():
    set_narrow(800)
    ROOT = Path(__file__).resolve().parents[1]
    LOGO = ROOT / "assets" / "Logo4.PNG"
    col_left, col_middle, col_right = st.columns([1, 12, 1])
    with col_middle:
        st.image(str(LOGO), use_container_width=True)
        uploaded = st.file_uploader(
            "Upload a file", 
            type=["csv", "xlsx"], 
            key="crna_uploader"
        )

    if not uploaded:
        return

    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([2,2,1,2,2])
    with btn_col2:
        demand_clicked = st.button("Demand ➡️", use_container_width=True, key="demand_btn")
    with btn_col4:
        capacity_clicked = st.button("Capacity ➡️", use_container_width=True, key="capacity_btn")

    if demand_clicked:
        st.session_state.uploaded_file = uploaded
        st.session_state.page = "step1"
        st.rerun()

    if capacity_clicked:
        # Save uploaded file to a temp location or BytesIO for processing
        import tempfile
        import os

        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded.getbuffer())
            tmp_path = tmp_file.name

        # Call your preprocessing function
        result_df = process_schedule_excel(tmp_path)
        st.session_state.preprocessed_result = result_df  # Optional: store result for later use
        st.session_state.uploaded_file = uploaded
        st.session_state.page = "step1"
        st.rerun()
