# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# dexd — the ONE D-Chain DEX image (CPU, linux/amd64) — ghcr.io/luxfi/dex
#
# Builds the single `dexd` binary and runs it as the standalone D-Chain venue:
# it opens the persisted zapdb, serves the frozen clob_* surface over ZAP, and
# runs the consensus sealer loop itself (WaitForEvent->BuildBlock->Verify->Accept)
# so every write returns consensus-computed fills. CGO_ENABLED=0 selects pkg/lx's
# pure-Go CPU matcher (the !cgo build tag) — no CUDA, no MLX, runs on any amd64.
#
# Same binary, other modes (override the default CMD):
#   docker run … plugin     run as a luxd rpcchainvm plugin (luxd execs it)
#   docker run … version    print the plugin version line
# ---------------------------------------------------------------------------
ARG GO_VERSION=1.26.4
FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates && rm -rf /var/lib/apt/lists/*

# luxfi module resolution: private modules resolve via the gh_token BuildKit
# secret over HTTPS (the shared hanzoai/.github docker-build.yml passes it);
# skip sumdb for luxfi (tags may be rewritten — go.sum in the build context is
# already realigned to current tag bits).
ENV GOPRIVATE=github.com/luxfi/*,github.com/hanzoai/*
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

# CPU build: CGO_ENABLED=0 → !cgo build tag → pure-Go CPU matcher in pkg/lx.
RUN --mount=type=secret,id=gh_token,required=false \
    if [ -s /run/secrets/gh_token ]; then \
        git config --global url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf "https://github.com/"; \
    fi && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-s -w" -o /out/dexd ./cmd/dexd

FROM debian:12-slim AS execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 dexd && useradd -u 1000 -g dexd -m -s /usr/sbin/nologin dexd \
    && mkdir -p /data && chown -R dexd:dexd /data
COPY --from=builder /out/dexd /usr/local/bin/dexd
USER dexd
EXPOSE 9099
# Default: standalone venue. Override CMD with `plugin` / `version` for the
# other run modes.
ENTRYPOINT ["/usr/local/bin/dexd"]
CMD ["run", "-addr", "0.0.0.0:9099", "-db", "/data/db"]
