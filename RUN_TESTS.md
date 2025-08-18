# LX DEX - Complete Test and Run Guide

## Summary

All components of the LX DEX are now **fully functional and tested**. The system integrates with Lux netrunner for proper network management.

## Quick Start

### 1. Run with Lux Netrunner (Recommended)
```bash
# Uses netrunner to start a proper Lux network
./scripts/run-lux-network.sh
```

### 2. Run Standalone DEX (Development)
```bash
# Single node for quick testing
./bin/xchain-dex --standalone --node-id node1
```

### 3. Run Multi-Node Network (Testing)
```bash
# 3-node network with FPC consensus
./scripts/run-fpc-network.sh
```
