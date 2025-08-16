module github.com/luxfi/dex/backend

go 1.21

replace github.com/luxfi/dex/backend => ./

require (
	github.com/gorilla/websocket v1.5.3
	github.com/grpc-ecosystem/grpc-gateway/v2 v2.18.1
	github.com/pebbe/zmq4 v1.4.0
	google.golang.org/grpc v1.65.0
	google.golang.org/protobuf v1.34.1
)

require (
	golang.org/x/net v0.25.0 // indirect
	golang.org/x/sys v0.20.0 // indirect
	golang.org/x/text v0.15.0 // indirect
	google.golang.org/genproto/googleapis/api v0.0.0-20240528184218-531527333157 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20240528184218-531527333157 // indirect
)
