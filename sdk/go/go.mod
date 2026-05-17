module github.com/luxfi/dex/sdk/go

go 1.26.3

// google.golang.org/grpc is only compiled when the `grpc` build tag is
// set. Default builds (JSON-RPC + WebSocket) pull zero gRPC code.
require (
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674
	github.com/luxfi/dex v0.0.0
	google.golang.org/grpc v1.79.1
)

require (
	go.opentelemetry.io/otel v1.43.0 // indirect
	golang.org/x/net v0.52.0 // indirect
	golang.org/x/sys v0.42.0 // indirect
	golang.org/x/text v0.35.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260217215200-42d3e9bedb6d // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

replace github.com/luxfi/dex => ../../
