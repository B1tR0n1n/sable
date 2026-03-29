#!/bin/bash
echo ""
echo "  Starting SABLE Reasoning Engine..."
echo ""
docker run --gpus all -p 8080:8080 --rm sable-engine
