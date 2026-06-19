// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command fix-gateway runs the FIX 4.4 acceptor as a long-lived TCP server in
// front of a running D-Chain venue (cmd/dvenue / the node runtime). It is the
// SERVER side of the FIX trading path: a FIX client (the SDK's EnableFIX(), or
// any FIX 4.4 engine) connects, logs on, and sends NewOrderSingle / OrderCancel
// messages; the gateway translates each into the SAME clob_* venue operation that
// ZAP, WS, and the 0x9010 precompile use, and renders the consensus-computed
// result back as a FIX ExecutionReport.
//
// It owns NO matcher and NO book — it dials the venue's clob_* ZAP endpoint
// (-venue) and is a pure protocol front-end (pkg/fix + pkg/venue). This is the
// "one matcher, many protocols" topology: the matcher lives in the venue
// (pkg/dchain), and fix-gateway is the FIX dialect onto it, exactly parallel to
// the WS and precompile front-ends.
package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/luxfi/dex/pkg/fix"
	"github.com/luxfi/dex/pkg/venue"
	log "github.com/luxfi/log"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:5001", "TCP listen address for the FIX acceptor")
	venueAddr := flag.String("venue", "127.0.0.1:9099", "ZAP endpoint of the D-Chain venue (clob_*)")
	target := flag.String("target", "LXDEX", "this acceptor's CompID (the client's TargetCompID)")
	heartBt := flag.Int("heartbeat", 30, "heartbeat interval advertised on Logon (seconds)")
	dialTimeout := flag.Duration("dial-timeout", 10*time.Second, "venue dial timeout")
	flag.Parse()

	logger := log.Root()

	// Dial the venue's clob_* surface — the SAME socket the precompile and proxy
	// dial. The gateway holds this one connection and routes every FIX order
	// through it.
	dialCtx, dialCancel := context.WithTimeout(context.Background(), *dialTimeout)
	v, err := venue.DialVenue(dialCtx, *venueAddr)
	dialCancel()
	if err != nil {
		logger.Error("dial venue", "venue", *venueAddr, "err", err)
		os.Exit(1)
	}

	acc, err := fix.NewAcceptor(fix.Config{
		Addr:         *addr,
		Venue:        v,
		TargetCompID: *target,
		HeartBtInt:   *heartBt,
		Logger:       logger,
	})
	if err != nil {
		logger.Error("new acceptor", "addr", *addr, "err", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Graceful shutdown on SIGINT/SIGTERM.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		logger.Info("fix-gateway shutting down")
		cancel()
	}()

	logger.Info("fix-gateway up", "addr", acc.Addr(), "venue", *venueAddr, "target", *target, "heartbeat", *heartBt)
	if err := acc.Serve(ctx); err != nil {
		logger.Error("acceptor serve", "err", err)
		os.Exit(1)
	}
}
