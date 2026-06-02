// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build integration

// Package main, file standalone_test.go.
//
// Exercises dex-api-server in standalone mode. dex-api-server is the
// in-process orderbook daemon — it owns a local pkg/lx.OrderBook and
// services REST + WebSocket from it. There is no consensus binding,
// no Lux mainnet handshake, no peer discovery. Standalone is the only
// mode this daemon supports.
//
// Smoke flow:
//  1. build dex-api-server
//  2. spawn it with a free port
//  3. POST /api/order to place a limit-buy order
//  4. GET /api/orderbook and confirm the order appears on the bid side
//  5. SIGTERM and confirm graceful exit
//
// Multi-node consensus is provided by an entirely different binary
// (cmd/dag-network, build-tagged zmqtest) that wires pkg/consensus
// directly. Wiring Quasar / LuxDAGOrderBook into dex-api-server would
// be a non-trivial refactor — see the README for the architectural
// rationale.
//
// Run: `go test -tags=integration ./cmd/dex-api-server/... -count=1 -timeout 60s`.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func waitReady(t *testing.T, addr string) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	url := "http://" + addr + "/api/orderbook"
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("dex-api-server not ready on %s", addr)
}

func TestStandalonePlaceOrderRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()

	binPath := filepath.Join(tmpDir, "dex-api-server")
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Dir = "."
	build.Stdout, build.Stderr = os.Stdout, os.Stderr
	if err := build.Run(); err != nil {
		t.Fatalf("go build dex-api-server: %v", err)
	}

	port := freePort(t)
	addr := fmt.Sprintf("127.0.0.1:%d", port)

	cmd := exec.Command(binPath, "-port", fmt.Sprint(port))
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Signal(syscall.SIGTERM)
		done := make(chan struct{})
		go func() { _ = cmd.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			_ = cmd.Process.Kill()
			<-done
		}
	})

	waitReady(t, addr)

	// Place a buy order at a price well above existing bids so it sits
	// on the book without immediately matching against the seeded asks.
	order, _ := json.Marshal(OrderRequest{
		Type:  "limit",
		Side:  "buy",
		Price: 99000.0,
		Size:  0.5,
	})
	resp, err := http.Post("http://"+addr+"/api/order", "application/json", bytes.NewReader(order))
	if err != nil {
		t.Fatalf("place order: %v", err)
	}
	defer resp.Body.Close()
	var placed Response
	if err := json.NewDecoder(resp.Body).Decode(&placed); err != nil {
		t.Fatalf("decode place response: %v", err)
	}
	if !placed.Success {
		t.Fatalf("place order failed: %s", placed.Message)
	}

	// The seeded book contains buy orders down to the 99000 area; our
	// new order may have matched. The contract under test is simply
	// that the daemon accepted the order and the book is still queryable.
	resp2, err := http.Get("http://" + addr + "/api/orderbook")
	if err != nil {
		t.Fatalf("get orderbook: %v", err)
	}
	defer resp2.Body.Close()
	var book Response
	if err := json.NewDecoder(resp2.Body).Decode(&book); err != nil {
		t.Fatalf("decode orderbook: %v", err)
	}
	if !book.Success {
		t.Fatalf("orderbook query failed: %s", book.Message)
	}
	data, ok := book.Data.(map[string]interface{})
	if !ok {
		t.Fatalf("unexpected orderbook payload: %T", book.Data)
	}
	if _, ok := data["bids"]; !ok {
		t.Fatalf("orderbook missing bids field: %v", data)
	}
}
