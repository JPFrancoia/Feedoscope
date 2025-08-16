#!/bin/bash

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

cd ~/projects/Feedoscope || exit 1

# Source .envrc if it exists
if [ -f .envrc ]; then
  source .envrc
fi

# Start port-forward in the background
kubectl -n ttrss port-forward service/db 5432:5432 > /dev/null 2>&1 &
PF_PID=$!
echo $PF_PID > /tmp/db_port_forward.pid
echo "Port-forward started (PID $PF_PID)."

# Cleanup function to kill port-forward
cleanup() {
  kill $PF_PID 2>/dev/null
  rm -f /tmp/db_port_forward.pid
}
trap cleanup EXIT

case "$1" in
  infer)
    make llm_infer
    ;;
  train)
    make llm_train
    ;;
  *)
    echo "Usage: $0 {infer|train}"
    exit 1
    ;;
esac
