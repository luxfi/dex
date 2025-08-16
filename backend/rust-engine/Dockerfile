# Rust Engine Dockerfile
FROM rust:1.75 AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY build.rs ./
COPY ../proto ../proto

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install grpcurl for health checks
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://github.com/fullstorydev/grpcurl/releases/download/v1.8.7/grpcurl_1.8.7_linux_x86_64.tar.gz | tar -xz -C /usr/local/bin

WORKDIR /root/
COPY --from=builder /app/target/release/lx-engine .

EXPOSE 50054
CMD ["./lx-engine"]