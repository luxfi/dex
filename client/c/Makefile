# LX DEX C CLI Trading Client
# Build for HFT and embedded systems

CC ?= cc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -O3 -march=native
LDFLAGS = -lwebsockets -lpthread

# Debug build
DEBUG_CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -g -O0 -DDEBUG

# Source and output
SRC = src/main.c
BIN = lx-cli
BUILD_DIR = build

# Platform detection
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    # macOS - use Homebrew paths
    BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
    CFLAGS += -I$(BREW_PREFIX)/include
    LDFLAGS += -L$(BREW_PREFIX)/lib
endif

ifeq ($(UNAME), Linux)
    LDFLAGS += -lrt
endif

.PHONY: all clean debug install uninstall test help

all: $(BUILD_DIR)/$(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(BIN): $(SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)
	@echo "Built $(BIN) successfully"
	@ls -lh $@

debug: CFLAGS = $(DEBUG_CFLAGS)
debug: $(BUILD_DIR)/$(BIN)

# Static build (for embedded/portable)
static: LDFLAGS += -static
static: $(BUILD_DIR)/$(BIN)

# Install to /usr/local/bin
install: $(BUILD_DIR)/$(BIN)
	install -m 755 $(BUILD_DIR)/$(BIN) /usr/local/bin/$(BIN)
	@echo "Installed to /usr/local/bin/$(BIN)"

uninstall:
	rm -f /usr/local/bin/$(BIN)

clean:
	rm -rf $(BUILD_DIR)

# Test - requires running LX DEX server
test: $(BUILD_DIR)/$(BIN)
	@echo "Testing connection..."
	./$(BUILD_DIR)/$(BIN) --timeout 2000 ping || echo "Server not running (expected)"
	@echo "Testing help..."
	./$(BUILD_DIR)/$(BIN) --help
	@echo "Testing version..."
	./$(BUILD_DIR)/$(BIN) --version

# Benchmark latency
bench: $(BUILD_DIR)/$(BIN)
	@echo "Running latency benchmark (requires server)..."
	@for i in 1 2 3 4 5; do ./$(BUILD_DIR)/$(BIN) ping 2>/dev/null; done

help:
	@echo "LX DEX C CLI Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       Build optimized binary (default)"
	@echo "  debug     Build with debug symbols"
	@echo "  static    Build static binary (portable)"
	@echo "  install   Install to /usr/local/bin"
	@echo "  uninstall Remove from /usr/local/bin"
	@echo "  clean     Remove build artifacts"
	@echo "  test      Run basic tests"
	@echo "  bench     Run latency benchmark"
	@echo ""
	@echo "Dependencies:"
	@echo "  - libwebsockets"
	@echo "  - pthreads"
	@echo ""
	@echo "Install dependencies:"
	@echo "  macOS:   brew install libwebsockets"
	@echo "  Ubuntu:  apt install libwebsockets-dev"
	@echo "  Fedora:  dnf install libwebsockets-devel"
