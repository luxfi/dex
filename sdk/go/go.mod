module github.com/luxfi/dex/sdk/go

go 1.24.5

require (
	github.com/gorilla/websocket v1.5.3
	github.com/luxfi/dex v0.0.0
	google.golang.org/grpc v1.74.2
)

require (
	go.opentelemetry.io/otel v1.39.0 // indirect
	golang.org/x/net v0.43.0 // indirect
	golang.org/x/sys v0.35.0 // indirect
	golang.org/x/text v0.28.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250721164621-a45f3dfb1074 // indirect
	google.golang.org/protobuf v1.36.7 // indirect
)

replace github.com/luxfi/dex => ../../
