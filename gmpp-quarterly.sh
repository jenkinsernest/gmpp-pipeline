#!/bin/bash
# launchd wrapper. Uses this folder's virtual environment and appends a log line.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:${PATH}"
export GMPP_PIPELINE_ROOT="$DIR"
mkdir -p "$DIR/logs" "$DIR/state"
{
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') launchd wrapper start"
  "$DIR/venv/bin/python3" "$DIR/gmpp_pipeline.py" --run
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') launchd wrapper end exit $?"
} >> "$DIR/logs/launchd.wrapper.log" 2>&1
