cmake_minimum_required(VERSION 3.20)
project(lx_engine_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto")

# Find required packages
find_package(Threads REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# Proto files
set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../proto")
set(PROTO_FILES "${PROTO_PATH}/lx_engine.proto")

# Generate proto sources
set(PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/lx_engine.pb.cc")
set(PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/lx_engine.pb.h")
set(GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/lx_engine.grpc.pb.cc")
set(GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/lx_engine.grpc.pb.h")

add_custom_command(
    OUTPUT "${PROTO_SRCS}" "${PROTO_HDRS}" "${GRPC_SRCS}" "${GRPC_HDRS}"
    COMMAND $<TARGET_FILE:protobuf::protoc>
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
         -I "${PROTO_PATH}"
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         "${PROTO_FILES}"
    DEPENDS "${PROTO_FILES}"
)

# Include generated files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# Main executable
add_executable(lx-engine-cpp
    main.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

target_link_libraries(lx-engine-cpp
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
)

# Optimized standalone orderbook benchmark
add_executable(orderbook-bench
    orderbook_bench.cpp
)

target_link_libraries(orderbook-bench
    Threads::Threads
)