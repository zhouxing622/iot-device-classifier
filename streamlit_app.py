"""
Streamlit Cloud Entry Point for IoT Device Classifier
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.app import main

if __name__ == "__main__":
    main()
