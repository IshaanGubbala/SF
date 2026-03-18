"""CLI entry points for AlzDetect."""
import subprocess
import sys


def run_app():
    """Launch the Streamlit web app."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
