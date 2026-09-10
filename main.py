import os
import sys
from streamlit.web import cli as stcli

def main():
    print("Hello from Intelligent-data-cleaner!")
    # Path to the streamlit application script
    script_path = os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")
    
    # Configure sys.argv to simulate running `streamlit run app/streamlit_app.py`
    sys.argv = ["streamlit", "run", script_path]
    
    # Launch Streamlit
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
