#!/bin/bash

# Lux Oracle Setup Script
# Real-time price feeds for all asset classes

set -e

echo "================================================"
echo "   Lux Oracle Setup"
echo "================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p config monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources

# Create Prometheus configuration
echo -e "${YELLOW}Creating Prometheus configuration...${NC}"
cat > monitoring/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'lux-oracle'
    static_configs:
      - targets: ['lux-oracle:8080']
    metrics_path: /metrics
    
  - job_name: 'lx-engines'
    static_configs:
      - targets: 
        - 'host.docker.internal:50051'  # Go engine
        - 'host.docker.internal:50052'  # Hybrid engine
        - 'host.docker.internal:50053'  # C++ engine
        - 'host.docker.internal:50054'  # Rust engine
EOF

# Create Grafana datasource
echo -e "${YELLOW}Creating Grafana datasource...${NC}"
cat > monitoring/grafana/datasources/prometheus.yml <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOF

# Create Grafana dashboard for Pyth prices
echo -e "${YELLOW}Creating Grafana dashboard...${NC}"
cat > monitoring/grafana/dashboards/pyth-prices.json <<EOF
{
  "dashboard": {
    "title": "Lux Exchange - Pyth Price Feeds",
    "panels": [
      {
        "title": "BTC/USD",
        "type": "graph",
        "targets": [
          {
            "expr": "pyth_price{symbol=\"BTC/USD\"}"
          }
        ]
      },
      {
        "title": "ETH/USD",
        "type": "graph",
        "targets": [
          {
            "expr": "pyth_price{symbol=\"ETH/USD\"}"
          }
        ]
      },
      {
        "title": "Price Feed Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "pyth_latency_ms"
          }
        ]
      },
      {
        "title": "Active Subscriptions",
        "type": "stat",
        "targets": [
          {
            "expr": "pyth_active_subscriptions"
          }
        ]
      }
    ]
  }
}
EOF

# Create environment file
echo -e "${YELLOW}Creating environment configuration...${NC}"
cat > .env <<EOF
# Pyth Network Configuration
PYTHNET_HTTP_ENDPOINT=https://pythnet.rpcpool.com
PYTHNET_WS_ENDPOINT=wss://pythnet.rpcpool.com

# Hermes Configuration
HERMES_ENDPOINT=0.0.0.0:2000
HERMES_WS_ENDPOINT=0.0.0.0:2002
HERMES_METRICS_PORT=8080

# Price Service URLs (Mainnet and Testnet)
PRICE_SERVICE_URLS=https://xc-mainnet.pyth.network,https://xc-testnet.pyth.network

# Performance
UPDATE_INTERVAL_MS=400
CACHE_TTL_SECONDS=1

# Logging
RUST_LOG=info
LOG_FORMAT=json
EOF

# Check if network exists
if ! docker network ls | grep -q "lx-backend_lx-network"; then
    echo -e "${YELLOW}Creating Docker network...${NC}"
    docker network create lx-backend_lx-network
fi

# Pull latest images
echo -e "${YELLOW}Pulling Docker images...${NC}"
docker pull public.ecr.aws/pyth-network/hermes:latest

# Start Lux Oracle
echo -e "${YELLOW}Starting Lux Oracle...${NC}"
docker-compose up -d lux-oracle

# Wait for Oracle to be ready
echo -e "${YELLOW}Waiting for Lux Oracle to be ready...${NC}"
sleep 10

# Check health
echo -e "${YELLOW}Checking Oracle health...${NC}"
if curl -f http://localhost:2000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Lux Oracle is healthy${NC}"
else
    echo -e "${RED}✗ Oracle health check failed${NC}"
    echo "Checking logs..."
    docker-compose logs lux-oracle | tail -20
    exit 1
fi

# Test price feed
echo -e "${YELLOW}Testing price feeds...${NC}"
BTC_PRICE=$(curl -s "http://localhost:2000/api/latest_price_feeds?ids[]=0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43" | jq -r '.[0].price.price' 2>/dev/null || echo "N/A")

if [ "$BTC_PRICE" != "N/A" ] && [ "$BTC_PRICE" != "null" ]; then
    echo -e "${GREEN}✓ BTC/USD Price: $BTC_PRICE${NC}"
else
    echo -e "${RED}✗ Failed to get BTC price${NC}"
fi

# Optional: Start monitoring
read -p "Do you want to start monitoring? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Starting monitoring stack...${NC}"
    docker-compose --profile monitoring up -d
    echo -e "${GREEN}✓ Monitoring available at:${NC}"
    echo "  - Prometheus: http://localhost:9091"
    echo "  - Grafana: http://localhost:3001 (admin/admin)"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   Lux Oracle Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Services running:"
echo "  - Oracle API: http://localhost:2000"
echo "  - Oracle WebSocket: ws://localhost:2002"
echo "  - Metrics: http://localhost:8080/metrics"
echo ""
echo "Available endpoints:"
echo "  - Latest prices: http://localhost:2000/api/latest_price_feeds"
echo "  - Price at time: http://localhost:2000/api/get_price_feed"
echo "  - WebSocket: ws://localhost:2002/ws"
echo ""
echo "To view logs: docker-compose logs -f hermes"
echo "To stop: docker-compose down"
echo ""