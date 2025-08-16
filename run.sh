#!/bin/bash

# Start port-forward in the background
kubectl -n ttrss port-forward service/db 5432:5432 > /dev/null 2>&1 &
PF_PID=$!
echo $PF_PID > /tmp/db_port_forward.pid
echo "Port-forward started (PID $PF_PID)."

# Check argument and run the appropriate make command
case "$1" in
  infer)
    make llm_infer
    ;;
  train)
    make llm_train
    ;;
  *)
    echo "Usage: $0 {infer|train}"
    kill $PF_PID
    rm /tmp/db_port_forward.pid
    exit 1
    ;;
esac

echo "To stop port-forwarding, run:"
echo "kill \$(cat /tmp/db_port_forward.pid) && rm /tmp/db_port_forward.pid"
