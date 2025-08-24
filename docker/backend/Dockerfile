# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev linux-headers git make

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build with CGO for hybrid engine
ENV CGO_ENABLED=1
ENV GOOS=linux
ENV GOARCH=amd64

# Build the binary
RUN make build

# Runtime stage
FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache ca-certificates curl

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/bin/lx-dex /app/lx-dex

# Copy configuration files
COPY --from=builder /app/config /app/config

# Create data directory
RUN mkdir -p /data

# Expose ports
EXPOSE 8080 8081 5000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/lx-dex"]

# Default arguments
CMD ["-http", "8080", "-ws", "8081", "-node-port", "5000", "-test", "-engine", "hybrid"]