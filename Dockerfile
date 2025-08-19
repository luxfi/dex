# Multi-stage build for LX DEX
FROM golang:1.22-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o lx-dex ./cmd/demo
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o dex-server ./cmd/dex-server

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -g 1000 -S dex && \
    adduser -u 1000 -S dex -G dex

# Set working directory
WORKDIR /app

# Copy binaries from builder
COPY --from=builder /app/lx-dex /app/lx-dex
COPY --from=builder /app/dex-server /app/dex-server

# Change ownership
RUN chown -R dex:dex /app

# Switch to non-root user
USER dex

# Expose ports
EXPOSE 8080 8081

# Default command
CMD ["/app/dex-server"]