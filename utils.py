from dotenv import load_dotenv
import os
import streamlit as st

def get_api_key():
    """Get API key from either Streamlit secrets (deployed) or local environment (development)"""
    # First try Streamlit secrets (deployment)
    try:
        api_key = st.secrets["FMP_API_KEY"]
        if api_key:
            return api_key
    except Exception:
        pass
    
    # Then try local .env file (development)
    load_dotenv()  # Ensure .env is loaded
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        raise ValueError(
            "FMP API key not found. Please set it either:\n"
            "1. In Streamlit secrets for deployment, or\n"
            "2. In a local .env file as FMP_API_KEY=your_key_here"
        )
    
    return api_key 