cmake_minimum_required(VERSION 3.16)
project(lx_trading_c VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(LX_TRADING_C_BUILD_TESTS "Build tests" ON)
option(LX_TRADING_C_USE_SIMD "Enable SIMD optimizations" OFF)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(LX_TRADING_C_USE_SIMD)
        add_compile_options(-march=native)
    endif()
endif()

# Find dependencies
find_package(Threads REQUIRED)

# Math library (required on Linux, optional on macOS/Windows)
find_library(MATH_LIBRARY m)

# Library sources
set(LX_TRADING_C_SOURCES
    src/types.c
    src/orderbook.c
    src/risk.c
    src/math.c
)

# Static library
add_library(lx_trading_c STATIC ${LX_TRADING_C_SOURCES})

target_include_directories(lx_trading_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(lx_trading_c
    PUBLIC
        Threads::Threads
)

if(MATH_LIBRARY)
    target_link_libraries(lx_trading_c PUBLIC ${MATH_LIBRARY})
endif()

# Tests
if(LX_TRADING_C_BUILD_TESTS)
    enable_testing()

    # Test executable
    add_executable(lx_trading_c_tests
        tests/test_main.c
        tests/test_types.c
        tests/test_orderbook.c
        tests/test_risk.c
        tests/test_math.c
    )

    target_link_libraries(lx_trading_c_tests
        PRIVATE
            lx_trading_c
    )

    # Register tests with CTest
    add_test(NAME types_test COMMAND lx_trading_c_tests types)
    add_test(NAME orderbook_test COMMAND lx_trading_c_tests orderbook)
    add_test(NAME risk_test COMMAND lx_trading_c_tests risk)
    add_test(NAME math_test COMMAND lx_trading_c_tests math)
    add_test(NAME all_tests COMMAND lx_trading_c_tests all)
endif()

# Install
include(GNUInstallDirs)
install(TARGETS lx_trading_c
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
