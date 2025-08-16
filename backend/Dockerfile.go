# Pure Go Engine Dockerfile
FROM golang:1.21-alpine AS builder

RUN apk add --no-cache git

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o lx-engine ./cmd/engine

FROM alpine:latest

RUN apk --no-cache add ca-certificates
RUN apk add --no-cache grpcurl

WORKDIR /root/
COPY --from=builder /app/lx-engine .
COPY --from=builder /app/configs ./configs

EXPOSE 50051
CMD ["./lx-engine"]