#!/bin/bash

# LX DEX Development Environment Setup Script
# This script sets up a complete development environment for LX DEX

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
BIN_DIR="$PROJECT_ROOT/bin"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Go
    if ! command -v go &> /dev/null; then
        print_error "Go is not installed. Please install Go 1.23+"
        exit 1
    fi
    
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    print_success "Go $GO_VERSION found"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker"
        exit 1
    fi
    print_success "Docker found"
    
    # Check Make
    if ! command -v make &> /dev/null; then
        print_error "Make is not installed. Please install make"
        exit 1
    fi
    print_success "Make found"
    
    # Check g++ for C++ compilation
    if ! command -v g++ &> /dev/null; then
        print_info "g++ not found. C++ engine will not be available"
        CGO_AVAILABLE=0
    else
        print_success "g++ found - C++ engine available"
        CGO_AVAILABLE=1
    fi
}

# Install Go dependencies
install_go_deps() {
    print_info "Installing Go dependencies..."
    cd "$BACKEND_DIR"
    go mod download
    go mod tidy
    print_success "Go dependencies installed"
}

# Install development tools
install_dev_tools() {
    print_info "Installing development tools..."
    
    # Install golangci-lint
    if ! command -v golangci-lint &> /dev/null; then
        print_info "Installing golangci-lint..."
        go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
    fi
    
    # Install security tools
    print_info "Installing security tools..."
    go install github.com/securego/gosec/v2/cmd/gosec@latest
    go install golang.org/x/vuln/cmd/govulncheck@latest
    
    # Install performance tools
    print_info "Installing performance tools..."
    go install github.com/google/pprof@latest
    
    # Install protobuf compiler
    if ! command -v protoc &> /dev/null; then
        print_info "protoc not found. Please install protobuf compiler manually"
    else
        print_success "protoc found"
    fi
    
    print_success "Development tools installed"
}

# Build all binaries
build_binaries() {
    print_info "Building binaries..."
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    rm -rf "$BIN_DIR"
    mkdir -p "$BIN_DIR"
    
    # Build with appropriate CGO setting
    if [ "$CGO_AVAILABLE" == "1" ]; then
        print_info "Building with CGO enabled (C++ optimizations)..."
        CGO_ENABLED=1 make build
    else
        print_info "Building with pure Go..."
        CGO_ENABLED=0 make build
    fi
    
    print_success "Binaries built successfully"
}

# Setup Docker environment
setup_docker() {
    print_info "Setting up Docker environment..."
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Build Docker images
    print_info "Building Docker images..."
    cd "$PROJECT_ROOT"
    
    if [ -f "docker-compose.yml" ]; then
        docker-compose build
        print_success "Docker images built"
    else
        print_info "docker-compose.yml not found, skipping Docker setup"
    fi
}

# Initialize configuration
init_config() {
    print_info "Initializing configuration..."
    
    # Create config directory
    mkdir -p "$PROJECT_ROOT/config"
    
    # Create default config if not exists
    if [ ! -f "$PROJECT_ROOT/config/dev.yaml" ]; then
        cat > "$PROJECT_ROOT/config/dev.yaml" << EOF
# LX DEX Development Configuration
engine:
  type: hybrid  # go, cpp, hybrid
  port: 50051
  
orderbook:
  max_depth: 1000
  decimal_places: 7
  
performance:
  batch_size: 100
  max_orders_per_second: 100000
  
monitoring:
  prometheus:
    enabled: true
    port: 9090
  
logging:
  level: debug
  format: json
EOF
        print_success "Development configuration created"
    else
        print_info "Configuration already exists"
    fi
}

# Run tests to verify setup
verify_setup() {
    print_info "Verifying setup with tests..."
    
    cd "$BACKEND_DIR"
    
    # Run quick tests
    if go test -v -short ./pkg/... > /dev/null 2>&1; then
        print_success "Tests passed"
    else
        print_error "Some tests failed. Run 'make test' for details"
    fi
    
    # Run quick benchmark
    print_info "Running quick benchmark..."
    if [ -f "$BIN_DIR/lx-bench" ]; then
        "$BIN_DIR/lx-bench" -iter 1000 > /dev/null 2>&1
        print_success "Benchmark completed"
    fi
}

# Create helpful aliases
create_aliases() {
    print_info "Creating development aliases..."
    
    cat > "$SCRIPTS_DIR/dev-aliases.sh" << 'EOF'
#!/bin/bash

# LX DEX Development Aliases

# Quick commands
alias lx-server='cd backend && go run ./cmd/dex-server'
alias lx-trader='cd backend && go run ./cmd/dex-trader'
alias lx-bench='make bench'
alias lx-test='make test'

# Docker commands
alias lx-up='docker-compose up -d'
alias lx-down='docker-compose down'
alias lx-logs='docker-compose logs -f'

# Development workflow
alias lx-fmt='cd backend && go fmt ./...'
alias lx-lint='cd backend && golangci-lint run'
alias lx-sec='cd backend && gosec ./...'

# Performance testing
alias lx-perf-quick='cd backend && make bench-quick'
alias lx-perf-full='cd backend && make bench-full'
alias lx-perf-max='cd backend && make bench-max'

echo "LX DEX development aliases loaded!"
EOF
    
    chmod +x "$SCRIPTS_DIR/dev-aliases.sh"
    print_success "Development aliases created. Source with: source scripts/dev-aliases.sh"
}

# Main setup flow
main() {
    echo "======================================"
    echo "   LX DEX Development Setup"
    echo "======================================"
    echo
    
    check_prerequisites
    echo
    
    install_go_deps
    echo
    
    install_dev_tools
    echo
    
    build_binaries
    echo
    
    setup_docker
    echo
    
    init_config
    echo
    
    verify_setup
    echo
    
    create_aliases
    echo
    
    print_success "Development environment setup complete!"
    echo
    echo "Next steps:"
    echo "  1. Source aliases: source scripts/dev-aliases.sh"
    echo "  2. Start server: make server"
    echo "  3. Run trader: make trader"
    echo "  4. Run benchmarks: make bench"
    echo
    echo "Happy coding! 🚀"
}

# Run main function
main