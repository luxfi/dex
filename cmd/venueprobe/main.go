// venueprobe queries the live D-Chain venue book for an arbitrary hex poolId
// (e.g. the EVM 0x9010 PoolKey.ID()) and optionally cancels a resting order —
// proving funds locked via the EVM place are visible + recoverable on the venue
// via the authoritative ZAP clob_* surface.
package main

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/luxfi/rpc"
)

func main() {
	addr := flag.String("addr", "dchain-venue.lux-devnet.svc.cluster.local:9099", "venue ZAP addr")
	poolHex := flag.String("pool", "", "32-byte poolId hex (EVM PoolKey.ID())")
	cancelID := flag.Uint64("cancel", 0, "orderID to cancel (0 = depth-only)")
	flag.Parse()

	pid, err := hex.DecodeString(*poolHex)
	if err != nil || len(pid) != 32 {
		fmt.Println("FATAL: -pool must be 32-byte hex")
		os.Exit(1)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	c, err := rpc.ZAPDial(ctx, *addr)
	if err != nil {
		fmt.Printf("FATAL dial %s: %v\n", *addr, err)
		os.Exit(1)
	}
	defer c.Close()
	fmt.Printf("probing venue %s poolId=%s\n", *addr, *poolHex)

	resp, err := c.Call(ctx, "clob_depth", pid)
	if err != nil {
		fmt.Printf("FATAL clob_depth: %v\n", err)
		os.Exit(1)
	}
	if len(resp) < 29 {
		fmt.Printf("short depth resp %d bytes\n", len(resp))
		os.Exit(1)
	}
	orders := binary.BigEndian.Uint32(resp[0:4])
	remaining := math.Float64frombits(binary.BigEndian.Uint64(resp[4:12]))
	bestBid := math.Float64frombits(binary.BigEndian.Uint64(resp[12:20]))
	bestAsk := math.Float64frombits(binary.BigEndian.Uint64(resp[20:28]))
	fmt.Printf("BOOK (EVM market on venue): orders=%d remaining=%.3f bestBid=%.4f bestAsk=%.4f\n",
		orders, remaining, bestBid, bestAsk)

	if *cancelID != 0 {
		payload := make([]byte, 40)
		copy(payload[0:32], pid)
		binary.BigEndian.PutUint64(payload[32:40], *cancelID)
		cr, cerr := c.Call(ctx, "clob_cancel", payload)
		if cerr != nil {
			fmt.Printf("clob_cancel err: %v\n", cerr)
			os.Exit(1)
		}
		fmt.Printf("clob_cancel orderID=%d -> ack %d bytes (RECOVERY PATH WORKS)\n", *cancelID, len(cr))
		// Re-query depth post-cancel.
		resp2, _ := c.Call(ctx, "clob_depth", pid)
		if len(resp2) >= 12 {
			o2 := binary.BigEndian.Uint32(resp2[0:4])
			r2 := math.Float64frombits(binary.BigEndian.Uint64(resp2[4:12]))
			fmt.Printf("BOOK post-cancel: orders=%d remaining=%.3f\n", o2, r2)
		}
	}
}
