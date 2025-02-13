#!/bin/env sh

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# Start Panel UI from app_runner
panel serve app_runner.py --port 5006 --allow-websocket-origin="*"
