#!/bin/bash

echo "Starting LXD node for 3 seconds..."
timeout 3 ./bin/lxd --block-time 1ms 2>&1 | grep -E "(Block #|Loaded state)"

echo ""
echo "Restarting to test persistence..."
timeout 2 ./bin/lxd --block-time 1ms 2>&1 | grep -E "(Loaded state|last block|Block #)" | head -5

echo ""
echo "Checking BadgerDB files:"
ls -lh ~/.lxd/badger/ | head -5