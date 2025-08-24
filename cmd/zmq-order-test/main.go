package main

import (
	"encoding/binary"
	"flag"
	"log"
	"math/rand"
	"time"

	zmq "github.com/pebbe/zmq4"
)

func main() {
	endpoint := flag.String("endpoint", "tcp://localhost:7001", "ZMQ endpoint for orders")
	numOrders := flag.Int("orders", 10, "Number of orders to send")
	flag.Parse()

	// Connect to X-Chain order processor
	socket, err := zmq.NewSocket(zmq.PUSH)
	if err != nil {
		log.Fatalf("Failed to create socket: %v", err)
	}
	defer socket.Close()

	err = socket.Connect(*endpoint)
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}

	log.Printf("Connected to %s", *endpoint)
	log.Printf("Sending %d orders...", *numOrders)

	// Send test orders in binary FIX format (60 bytes each)
	for i := 0; i < *numOrders; i++ {
		order := make([]byte, 60)

		// Message type (1 byte) - 'D' for new order
		order[0] = 'D'

		// Side (1 byte) - '1' for buy, '2' for sell
		if rand.Float32() < 0.5 {
			order[1] = '1' // Buy
		} else {
			order[1] = '2' // Sell
		}

		// Order type (1 byte) - '2' for limit
		order[2] = '2'

		// Padding (1 byte)
		order[3] = 0

		// Symbol (8 bytes) - "BTC-USD\0"
		symbol := "BTC-USD"
		copy(order[4:12], []byte(symbol))

		// Order ID (8 bytes)
		orderID := uint64(1000000 + i)
		binary.BigEndian.PutUint64(order[12:20], orderID)

		// Timestamp (8 bytes)
		timestamp := uint64(time.Now().Unix())
		binary.BigEndian.PutUint64(order[20:28], timestamp)

		// Price (8 bytes) - as fixed point with 8 decimals
		price := uint64((50000 + rand.Float64()*1000) * 1e8)
		binary.BigEndian.PutUint64(order[28:36], price)

		// Size (8 bytes) - as fixed point with 8 decimals
		size := uint64((0.1 + rand.Float64()*2) * 1e8)
		binary.BigEndian.PutUint64(order[36:44], size)

		// Remaining bytes are padding

		// Send the order
		_, err := socket.SendBytes(order, 0)
		if err != nil {
			log.Printf("Failed to send order %d: %v", i, err)
			continue
		}

		side := "BUY"
		if order[1] == '2' {
			side = "SELL"
		}

		log.Printf("Sent order %d: %s %.2f BTC @ $%.2f",
			orderID, side,
			float64(size)/1e8,
			float64(price)/1e8)

		// Small delay between orders
		time.Sleep(100 * time.Millisecond)
	}

	log.Println("✅ All orders sent!")
	log.Println("Check the X-Chain nodes to see if blocks are produced with trades")
}
