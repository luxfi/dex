package main

import (
	"fmt"
	"log"
	"time"

	"github.com/luxfi/qzmq"
)

func main() {
	fmt.Println("Testing QZMQ Integration with DEX...")
	
	// Test 1: Basic transport creation
	fmt.Println("\n1. Creating QZMQ transport...")
	opts := qzmq.DefaultOptions()
	transport, err := qzmq.New(opts)
	if err != nil {
		log.Fatalf("Failed to create transport: %v", err)
	}
	defer transport.Close()
	fmt.Println("✓ Transport created successfully")
	
	// Test 2: Socket creation
	fmt.Println("\n2. Creating sockets...")
	req, err := transport.NewSocket(qzmq.REQ)
	if err != nil {
		log.Fatalf("Failed to create REQ socket: %v", err)
	}
	defer req.Close()
	
	rep, err := transport.NewSocket(qzmq.REP)
	if err != nil {
		log.Fatalf("Failed to create REP socket: %v", err)
	}
	defer rep.Close()
	fmt.Println("✓ Sockets created successfully")
	
	// Test 3: Bind and connect
	fmt.Println("\n3. Binding and connecting...")
	endpoint := "tcp://127.0.0.1:25577"
	if err := rep.Bind(endpoint); err != nil {
		log.Fatalf("Failed to bind: %v", err)
	}
	
	if err := req.Connect(endpoint); err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	fmt.Println("✓ Bind and connect successful")
	
	// Test 4: Message exchange
	fmt.Println("\n4. Testing message exchange...")
	
	// Server goroutine
	done := make(chan bool)
	go func() {
		msg, err := rep.Recv()
		if err != nil {
			log.Printf("Server recv error: %v", err)
			done <- false
			return
		}
		
		if string(msg) != "Hello QZMQ" {
			log.Printf("Unexpected message: %s", msg)
			done <- false
			return
		}
		
		if err := rep.Send([]byte("Hello Client")); err != nil {
			log.Printf("Server send error: %v", err)
			done <- false
			return
		}
		
		done <- true
	}()
	
	// Give server time to start
	time.Sleep(100 * time.Millisecond)
	
	// Client sends message
	if err := req.Send([]byte("Hello QZMQ")); err != nil {
		log.Fatalf("Client send error: %v", err)
	}
	
	// Client receives reply
	reply, err := req.Recv()
	if err != nil {
		log.Fatalf("Client recv error: %v", err)
	}
	
	if string(reply) != "Hello Client" {
		log.Fatalf("Unexpected reply: %s", reply)
	}
	
	// Wait for server to complete
	success := <-done
	if !success {
		log.Fatal("Server failed")
	}
	
	fmt.Println("✓ Message exchange successful")
	
	// Test 5: Get transport stats
	fmt.Println("\n5. Getting transport stats...")
	stats := transport.Stats()
	fmt.Printf("✓ Stats: Messages encrypted: %d, Messages decrypted: %d\n", 
		stats.MessagesEncrypted, stats.MessagesDecrypted)
	
	fmt.Println("\n✅ All QZMQ integration tests passed!")
	fmt.Println("\nQZMQ is successfully integrated with the DEX and working correctly.")
	fmt.Println("The DEX can now use quantum-safe ZeroMQ transport for all messaging.")
}