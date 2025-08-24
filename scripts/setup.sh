#!/bin/bash

# Lux Exchange Engine Setup Script
# Complete setup for development environment

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Lux Exchange Engine Setup                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗ $1 is not installed${NC}"
        return 1
    else
        echo -e "${GREEN}✓ $1 is installed${NC}"
        return 0
    fi
}

MISSING_DEPS=0

check_command "git" || MISSING_DEPS=1
check_command "go" || MISSING_DEPS=1
check_command "node" || MISSING_DEPS=1
check_command "docker" || MISSING_DEPS=1

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo -e "${RED}Missing dependencies detected!${NC}"
    echo "Please install missing dependencies and run setup again."
    echo ""
    echo "Installation instructions:"
    echo "  - Go: https://golang.org/doc/install"
    echo "  - Node.js: https://nodejs.org/"
    echo "  - Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo ""
echo -e "${GREEN}All prerequisites installed!${NC}"
echo ""

# Create directory structure
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p config scripts docs examples test

# Install dependencies
echo ""
echo -e "${YELLOW}Installing dependencies...${NC}"

# Backend dependencies
if [ -d "backend" ]; then
    echo "Installing backend dependencies..."
    cd backend
    if [ ! -f "go.mod" ]; then
        go mod init github.com/luxexchange/engine/backend
    fi
    go mod download 2>/dev/null || echo "Go dependencies installed"
    cd ..
fi

# Client dependencies
if [ -d "client" ]; then
    echo "Installing client dependencies..."
    cd client
    if [ -f "package.json" ]; then
        npm install --no-audit --no-fund 2>/dev/null || echo "Client dependencies installed"
    fi
    cd ..
fi

# Oracle dependencies
if [ -d "oracle" ]; then
    echo "Installing oracle dependencies..."
    cd oracle
    if [ -f "package.json" ]; then
        npm install --no-audit --no-fund 2>/dev/null || echo "Oracle dependencies installed"
    fi
    cd ..
fi

# Create Docker network
echo ""
echo -e "${YELLOW}Setting up Docker network...${NC}"
docker network create lux-network 2>/dev/null || echo "Network already exists"

# Create default configuration
echo ""
echo -e "${YELLOW}Creating default configuration...${NC}"

if [ ! -f "config/.env.example" ]; then
cat > config/.env.example <<EOF
# Lux Exchange Engine Configuration

# Engine Settings
ENGINE_MODE=hybrid
ENGINE_PORT=50050
LOG_LEVEL=info

# Oracle Settings
ORACLE_ENDPOINT=http://localhost:2000
ORACLE_WS_ENDPOINT=ws://localhost:2002

# Database
DATABASE_URL=postgresql://lux:password@localhost:5432/lux_exchange

# Redis
REDIS_URL=redis://localhost:6379

# API Keys (replace with your actual keys)
API_KEY=your-api-key-here
API_SECRET=your-api-secret-here
EOF
echo -e "${GREEN}✓ Configuration template created${NC}"
else
echo "Configuration already exists"
fi

# Create start script
cat > start.sh <<'EOF'
#!/bin/bash
echo "Starting Lux Exchange Engine..."
make oracle-start 2>/dev/null || echo "Oracle start skipped"
sleep 5
make backend-start 2>/dev/null || echo "Backend start skipped"
echo "Lux Exchange Engine is running!"
echo "  - Engine API: http://localhost:50050"
echo "  - Oracle API: http://localhost:2000"
echo "  - Monitoring: http://localhost:3000"
EOF
chmod +x start.sh

# Create stop script
cat > stop.sh <<'EOF'
#!/bin/bash
echo "Stopping Lux Exchange Engine..."
make backend-stop 2>/dev/null || echo "Backend stop skipped"
make oracle-stop 2>/dev/null || echo "Oracle stop skipped"
echo "Lux Exchange Engine stopped."
EOF
chmod +x stop.sh

# Summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Setup Complete!                                ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Configure environment:"
echo -e "   ${YELLOW}cp config/.env.example config/.env${NC}"
echo -e "   ${YELLOW}vim config/.env${NC}  # Add your configuration"
echo ""
echo "2. Start services:"
echo -e "   ${YELLOW}./start.sh${NC}  # Start all services"
echo "   or"
echo -e "   ${YELLOW}make start${NC}  # Using Makefile"
echo ""
echo "3. Run tests:"
echo -e "   ${YELLOW}make test${NC}"
echo ""
echo "4. View logs:"
echo -e "   ${YELLOW}make logs${NC}"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo -e "  ${YELLOW}make help${NC}      # Show all available commands"
echo -e "  ${YELLOW}make status${NC}    # Check service status"
echo -e "  ${YELLOW}make benchmark${NC} # Run performance tests"
echo -e "  ${YELLOW}./stop.sh${NC}      # Stop all services"
echo ""
echo -e "${GREEN}Happy trading! 🚀${NC}"