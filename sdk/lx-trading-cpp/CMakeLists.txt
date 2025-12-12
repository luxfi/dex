cmake_minimum_required(VERSION 3.16)
project(lx_trading VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Allow older CMake policies for dependencies
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Options
option(LX_TRADING_BUILD_TESTS "Build tests" ON)
option(LX_TRADING_BUILD_EXAMPLES "Build examples" ON)
option(LX_TRADING_USE_SIMD "Enable SIMD optimizations" ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(LX_TRADING_USE_SIMD)
        add_compile_options(-march=native)
    endif()
endif()

# Find dependencies
find_package(Threads REQUIRED)

# FetchContent for dependencies
include(FetchContent)

# nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

# cpr (HTTP client)
FetchContent_Declare(
    cpr
    GIT_REPOSITORY https://github.com/libcpr/cpr.git
    GIT_TAG 1.10.5
)

FetchContent_MakeAvailable(json cpr)

# Library
add_library(lx_trading
    src/types.cpp
    src/config.cpp
    src/adapter.cpp
    src/native_adapter.cpp
    src/ccxt_adapter.cpp
    src/hummingbot_adapter.cpp
    src/orderbook.cpp
    src/client.cpp
    src/risk.cpp
    src/execution.cpp
    src/math.cpp
    # Arbitrage module
    src/arbitrage/types.cpp
    src/arbitrage/scanner.cpp
    src/arbitrage/lx_first.cpp
    src/arbitrage/unified.cpp
    src/arbitrage/cross_chain.cpp
)

target_include_directories(lx_trading
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(lx_trading
    PUBLIC
        nlohmann_json::nlohmann_json
        cpr::cpr
        Threads::Threads
)

# Tests
if(LX_TRADING_BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.0
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(lx_trading_tests
        tests/test_types.cpp
        tests/test_orderbook.cpp
        tests/test_math.cpp
        tests/test_risk.cpp
    )

    target_link_libraries(lx_trading_tests
        PRIVATE
            lx_trading
            Catch2::Catch2WithMain
    )

    include(CTest)
    include(Catch)
    catch_discover_tests(lx_trading_tests)
endif()

# Examples
if(LX_TRADING_BUILD_EXAMPLES)
    add_executable(example_basic examples/basic.cpp)
    target_link_libraries(example_basic PRIVATE lx_trading)

    add_executable(example_orderbook examples/orderbook_demo.cpp)
    target_link_libraries(example_orderbook PRIVATE lx_trading)
endif()

# Install (simplified - dependencies handled separately)
include(GNUInstallDirs)
install(TARGETS lx_trading
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
