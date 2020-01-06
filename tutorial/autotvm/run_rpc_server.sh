#!/bin/bash
while (true)
do
    echo "started server at " $(date) >> status.log
    python3 -m tvm.exec.rpc_server --key 1080ti --tracker 0.0.0.0:9190
    sleep 30
done
