# LX C++ CLI Trading Client
# Copyright (c) 2025 Lux Partners Limited
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(lx-cli VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release with debug info by default for profiling
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(Threads REQUIRED)

include(FetchContent)

# nlohmann/json - JSON parsing
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# websocketpp - WebSocket client
FetchContent_Declare(
    websocketpp
    GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
    GIT_TAG 0.8.2
)
FetchContent_Populate(websocketpp)

# asio standalone (no Boost)
FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG asio-1-28-2
)
FetchContent_Populate(asio)

# CLI executable
add_executable(lx-cli
    src/main.cpp
)

target_include_directories(lx-cli
    PRIVATE
        ${websocketpp_SOURCE_DIR}
        ${asio_SOURCE_DIR}/asio/include
)

target_link_libraries(lx-cli
    PRIVATE
        nlohmann_json::nlohmann_json
        Threads::Threads
)

target_compile_definitions(lx-cli
    PRIVATE
        ASIO_STANDALONE
        _WEBSOCKETPP_CPP11_STL_
)

# Platform-specific
if(APPLE)
    target_compile_definitions(lx-cli PRIVATE _DARWIN_C_SOURCE)
endif()

if(WIN32)
    target_link_libraries(lx-cli PRIVATE ws2_32)
endif()

# Optimization flags for HFT
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lx-cli PRIVATE
        -O3
        -march=native
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# Install
install(TARGETS lx-cli
    RUNTIME DESTINATION bin
)
