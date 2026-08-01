# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# dexd-testnet — the multi-node testnet image (CPU, pure-Go, native arch).
#
# Builds BOTH binaries the testnet needs from this repo:
#   /usr/local/bin/dexd      — the D-Chain venue (cmd/dexd, `dexd run`)
#   /usr/local/bin/dexbench  — the signed-ZAP load-gen + cross-arch determinism
#                              checker (cmd/dexbench)
#
# CGO_ENABLED=0 → pkg/lx's pure-Go CPU matcher (the !cgo build tag). No CUDA, no
# MLX — so this ONE image builds and runs natively on BOTH boxes:
#   evo   = linux/amd64 (WSL)         spark = linux/arm64
# The build does NOT cross-compile: GOARCH defaults to the build host's arch, so
# `docker build` on each box yields that box's native image. The ZAP wire is a
# frozen big-endian codec (pkg/zapwire), byte-identical on amd64⇄arm64, which is
# what makes the cross-arch venue fleet sound (prove it with `dexbench -verify`).
#
# GPU NOTE: the GPU matcher (CGO_ENABLED=1) is arm64+CUDA only and lives in
# Dockerfile.dvenue (spark only, needs the private luxcpp dep tarball). The CPU
# image here is the canonical, portable testnet path; see README.md "GPU".
# ---------------------------------------------------------------------------
# Must be >= the `go` directive in go.mod (1.26.4); pinned to the latest stable
# patch so the image ships current compiler/stdlib security fixes. GOTOOLCHAIN
# below stays `auto` so a future floor bump downloads rather than hard-fails.
ARG GO_VERSION=1.26.5
FROM golang:${GO_VERSION}-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates && rm -rf /var/lib/apt/lists/*

# No GOPRIVATE — public modules resolve via the immutable public proxy + sumdb.
# dex's deps (luxfi/*, hanzoai/*, hanzos3/go-sdk) are all PUBLIC now; the immutable
# proxy+sumdb hash a force-moved tag can't break. The gh_token BuildKit secret
# below stays only for a cold `direct` fetch fallback.
ENV GOFLAGS=-mod=mod
ENV GOTOOLCHAIN=auto
ENV GOEXPERIMENT=jsonv2

WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=secret,id=gh_token,required=false \
    if [ -s /run/secrets/gh_token ]; then \
        git config --global url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf "https://github.com/"; \
    fi && \
    go mod download
COPY . .

# Build both binaries pure-Go, native arch (no GOOS/GOARCH override).
RUN --mount=type=secret,id=gh_token,required=false \
    if [ -s /run/secrets/gh_token ]; then \
        git config --global url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf "https://github.com/"; \
    fi && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/dexd      ./cmd/dexd && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/dexbench  ./cmd/dexbench

FROM debian:12-slim AS execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates netcat-openbsd && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 dexd && useradd -u 1000 -g dexd -m -s /usr/sbin/nologin dexd \
    && mkdir -p /data && chown -R dexd:dexd /data
COPY --from=builder /out/dexd     /usr/local/bin/dexd
COPY --from=builder /out/dexbench /usr/local/bin/dexbench
USER dexd
EXPOSE 9099
# Default: standalone venue on all interfaces. The compose overrides -db per node.
ENTRYPOINT ["/usr/local/bin/dexd"]
CMD ["run", "-addr", "0.0.0.0:9099", "-db", "/data/db"]
