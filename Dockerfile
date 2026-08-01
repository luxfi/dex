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
# Must be >= the `go` directive in go.mod (1.26.4); pinned to the latest stable
# patch so the image ships current compiler/stdlib security fixes. GOTOOLCHAIN
# below stays `auto` so a future floor bump downloads rather than hard-fails.
ARG GO_VERSION=1.26.5
FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-bookworm AS builder

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
