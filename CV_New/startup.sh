#!/bin/bash
# Azure App Service (Linux) runs this as the startup command.
# Set this exact line in Azure Portal -> Configuration -> General settings -> Startup Command
gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT app:app --timeout 120
