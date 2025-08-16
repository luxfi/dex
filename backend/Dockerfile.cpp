# Pure C++ Engine Dockerfile
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    make \
    libboost-all-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY cpp-engine/ ./cpp-engine/
COPY proto/ ./proto/

# Build C++ engine
RUN cd cpp-engine && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc)

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libstdc++6 \
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    libboost-filesystem1.74.0 \
    libprotobuf23 \
    libgrpc++1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install grpcurl for health checks
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://github.com/fullstorydev/grpcurl/releases/download/v1.8.7/grpcurl_1.8.7_linux_x86_64.tar.gz | tar -xz -C /usr/local/bin

WORKDIR /root/
COPY --from=builder /app/cpp-engine/build/lx-cpp-engine .
COPY configs/ ./configs/

EXPOSE 50053
CMD ["./lx-cpp-engine"]