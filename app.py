"""STEER entry point: open the profile explorer by default."""

import streamlit as st

st.set_page_config(page_title="STEER | School Profile Explorer", layout="wide")

explorer = st.Page("pages/explorer.py", title="Explorer", default=True)
story = st.Page("pages/1_Story.py", title="Story")
page = st.navigation([explorer, story], position="hidden")

page.run()
