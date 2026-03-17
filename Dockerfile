# syntax=docker/dockerfile:1
FROM --platform=$BUILDPLATFORM golang:1.26-alpine AS builder

RUN apk add --no-cache git ca-certificates

ARG GITHUB_TOKEN
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
ENV GOPRIVATE=github.com/luxfi/*,github.com/hanzoai/*

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o dex-server ./cmd/dex-server/main.go

FROM alpine:3.21

RUN apk add --no-cache ca-certificates && \
    addgroup -g 1000 -S dex && \
    adduser -u 1000 -S dex -G dex

WORKDIR /home/dex
COPY --from=builder /app/dex-server .
RUN chown dex:dex dex-server

USER dex
EXPOSE 8080 9090 50051

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

CMD ["./dex-server"]
