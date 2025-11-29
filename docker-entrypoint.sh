#!/bin/bash
set -e

echo "Starting Streamlit app..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
