import streamlit as st
from multi_hyde_inferencing import end_to_end
import os
import pandas as pd

# Sidebar: API Key input
st.sidebar.title("GROQ API ONLY")
api_key = st.sidebar.text_input("Enter your API Key", type="password")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

# Main interface
st.title("ICD Code Detector")

# Input field
user_input = st.text_area("Enter Doctor's Note")

# Run button
if st.button("Run"):
    if not api_key:
        st.warning("Please enter an API Key in the sidebar.")
    elif not user_input.strip():
        st.warning("Please enter some input.")
    else:
        # Simulate processing
        # output = f"### Output\n\nYou entered:\n\n```\n{user_input}\n```"
        output = end_to_end(user_input)

        df = pd.DataFrame(output)
        df.columns = ["ICD Code", "ICD Version", "ICD Description"]

        st.dataframe(df)

        # Additional content inside expander
        with st.expander("See more details"):
            st.markdown("**Additional Info:** This section can include metadata, logs, or debug information.")
