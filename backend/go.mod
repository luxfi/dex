module github.com/luxfi/dex/backend

go 1.23.0

toolchain go1.24.6

replace github.com/luxfi/dex/backend => ./

require (
	github.com/nats-io/nats.go v1.44.0
	github.com/pebbe/zmq4 v1.4.0
)

require (
	github.com/klauspost/compress v1.18.0 // indirect
	github.com/nats-io/nkeys v0.4.11 // indirect
	github.com/nats-io/nuid v1.0.1 // indirect
	golang.org/x/crypto v0.37.0 // indirect
	golang.org/x/sys v0.32.0 // indirect
)
