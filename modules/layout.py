
import streamlit as st
def set_narrow(max_width: int = 700):
    st.markdown(f"""
    <style>
      .block-container {{
        max-width: {max_width}px;
        margin: auto;
      }}
    </style>
    """, unsafe_allow_html=True)

def set_fullwidth():
    st.markdown("""
    <style>
      .block-container {
        max-width: none;
        padding-left: 2rem;
        padding-right: 2rem;
      }
    </style>
    """, unsafe_allow_html=True)
