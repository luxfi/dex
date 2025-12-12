#!/bin/bash
# LX Trading SDK - Unified Test Runner
# Runs all tests across all SDK implementations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Results tracking
declare -A RESULTS
TOTAL_TESTS=0
TOTAL_PASSED=0
TOTAL_FAILED=0

log_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Test Go SDK
test_go() {
    log_header "Testing Go SDK (lx-trading-go)"
    cd "$SDK_DIR/lx-trading-go"

    if go test -v ./... 2>&1 | tee /tmp/go_test.log; then
        PASSED=$(grep -c "^--- PASS" /tmp/go_test.log || echo "0")
        RESULTS["Go"]="PASS ($PASSED tests)"
        TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
        TOTAL_TESTS=$((TOTAL_TESTS + PASSED))
        log_success "Go SDK: $PASSED tests passed"
        return 0
    else
        RESULTS["Go"]="FAIL"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        log_error "Go SDK tests failed"
        return 1
    fi
}

# Test TypeScript SDK
test_typescript() {
    log_header "Testing TypeScript SDK (lx-trading-ts)"
    cd "$SDK_DIR/lx-trading-ts"

    # Build first
    npm run build 2>/dev/null || true

    if npm test 2>&1 | tee /tmp/ts_test.log; then
        PASSED=$(grep -E "^ℹ pass [0-9]+" /tmp/ts_test.log | grep -oE "[0-9]+" || echo "0")
        RESULTS["TypeScript"]="PASS ($PASSED tests)"
        TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
        TOTAL_TESTS=$((TOTAL_TESTS + PASSED))
        log_success "TypeScript SDK: $PASSED tests passed"
        return 0
    else
        RESULTS["TypeScript"]="FAIL"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        log_error "TypeScript SDK tests failed"
        return 1
    fi
}

# Test Python SDK
test_python() {
    log_header "Testing Python SDK (lx-trading-py)"
    cd "$SDK_DIR/lx-trading-py"

    export PYTHONPATH="$SDK_DIR/lx-trading-py/python:$PYTHONPATH"

    if python -m pytest tests/ -v 2>&1 | tee /tmp/py_test.log; then
        PASSED=$(grep -oE "[0-9]+ passed" /tmp/py_test.log | grep -oE "[0-9]+" || echo "0")
        RESULTS["Python"]="PASS ($PASSED tests)"
        TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
        TOTAL_TESTS=$((TOTAL_TESTS + PASSED))
        log_success "Python SDK: $PASSED tests passed"
        return 0
    else
        RESULTS["Python"]="FAIL"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        log_error "Python SDK tests failed"
        return 1
    fi
}

# Test Rust SDK
test_rust() {
    log_header "Testing Rust SDK (lx-trading-core)"
    cd "$SDK_DIR/lx-trading-core"

    if cargo test --all-features 2>&1 | tee /tmp/rust_test.log; then
        PASSED=$(grep -oE "[0-9]+ passed" /tmp/rust_test.log | tail -1 | grep -oE "[0-9]+" || echo "0")
        RESULTS["Rust"]="PASS ($PASSED tests)"
        TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
        TOTAL_TESTS=$((TOTAL_TESTS + PASSED))
        log_success "Rust SDK: $PASSED tests passed"
        return 0
    else
        RESULTS["Rust"]="FAIL"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        log_error "Rust SDK tests failed"
        return 1
    fi
}

# Test C++ SDK
test_cpp() {
    log_header "Testing C++ SDK (lx-trading-cpp)"
    cd "$SDK_DIR/lx-trading-cpp"

    # Build
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>/dev/null
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>/dev/null

    if ./lx_trading_tests 2>&1 | tee /tmp/cpp_test.log; then
        PASSED=$(grep -oE "[0-9]+ assertions" /tmp/cpp_test.log | grep -oE "[0-9]+" || echo "0")
        RESULTS["C++"]="PASS ($PASSED assertions)"
        TOTAL_PASSED=$((TOTAL_PASSED + 30))  # Approximate test count
        TOTAL_TESTS=$((TOTAL_TESTS + 30))
        log_success "C++ SDK: tests passed"
        return 0
    else
        RESULTS["C++"]="FAIL"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        log_error "C++ SDK tests failed"
        return 1
    fi
}

# Print summary
print_summary() {
    log_header "Test Summary"

    echo ""
    printf "%-15s %s\n" "SDK" "Result"
    printf "%-15s %s\n" "---" "------"

    for sdk in "Go" "TypeScript" "Python" "Rust" "C++"; do
        result="${RESULTS[$sdk]:-NOT RUN}"
        if [[ "$result" == PASS* ]]; then
            printf "%-15s ${GREEN}%s${NC}\n" "$sdk" "$result"
        elif [[ "$result" == "FAIL" ]]; then
            printf "%-15s ${RED}%s${NC}\n" "$sdk" "$result"
        else
            printf "%-15s ${YELLOW}%s${NC}\n" "$sdk" "$result"
        fi
    done

    echo ""
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $TOTAL_PASSED"
    echo "Failed SDKs: $TOTAL_FAILED"
    echo ""

    if [ $TOTAL_FAILED -eq 0 ]; then
        log_success "All SDK tests passed!"
        return 0
    else
        log_error "Some SDK tests failed!"
        return 1
    fi
}

# Main
main() {
    log_header "LX Trading SDK - Unified Test Suite"
    echo "SDK Directory: $SDK_DIR"
    echo ""

    # Parse arguments
    SDKS_TO_TEST=("go" "typescript" "python" "rust" "cpp")

    if [ $# -gt 0 ]; then
        SDKS_TO_TEST=("$@")
    fi

    # Run tests
    for sdk in "${SDKS_TO_TEST[@]}"; do
        case "$sdk" in
            go|Go)
                test_go || true
                ;;
            typescript|ts|TypeScript)
                test_typescript || true
                ;;
            python|py|Python)
                test_python || true
                ;;
            rust|Rust)
                test_rust || true
                ;;
            cpp|c++|C++)
                test_cpp || true
                ;;
            *)
                log_error "Unknown SDK: $sdk"
                ;;
        esac
    done

    print_summary
}

main "$@"
