#!/bin/bash
python3 -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190 --no-fork &
while (true)
do
    res=$(python3 -m tvm.exec.query_rpc_tracker --host 0.0.0.0 --port 9190 | grep 'Cannot connect to tracker')
    if [ "$res" == "" ]; then
	echo "OK @ " $(date) "..." >> status.log
    else
	echo "RESTARTING @ " $(date) "..." >> status.log
	python3 -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190 --no-fork &
    fi
    sleep 60
done

	   
