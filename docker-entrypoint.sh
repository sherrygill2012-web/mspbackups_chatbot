#!/bin/bash
set -e

case "$RUN_MODE" in
    "api")
        echo "Starting REST API server..."
        exec uvicorn api:app --host 0.0.0.0 --port 8000
        ;;
    "slack")
        echo "Starting Slack bot..."
        exec python slack_bot.py
        ;;
    "streamlit"|*)
        echo "Starting Streamlit app..."
        exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
        ;;
esac

