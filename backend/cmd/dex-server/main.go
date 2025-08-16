package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
)

type ServerAnnouncement struct {
	ID       string    `json:"id"`
	Addr     string    `json:"addr"`
	Type     string    `json:"type"`
	Started  time.Time `json:"started"`
}

type DexServer struct {
	nc         *nats.Conn
	id         string
	addr       string
	orders     int64
	trades     int64
}

func main() {
	port := flag.Int("port", 8080, "Server port")
	natsURL := flag.String("nats", nats.DefaultURL, "NATS URL")
	flag.Parse()

	hostname, _ := os.Hostname()
	id := fmt.Sprintf("%s-%d", hostname, os.Getpid())
	addr := fmt.Sprintf("%s:%d", getLocalIP(), *port)

	log.Printf("🏦 DEX Server starting")
	log.Printf("📍 ID: %s", id)
	log.Printf("🌐 Address: %s", addr)

	// Connect to NATS
	nc, err := nats.Connect(*natsURL)
	if err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}
	defer nc.Close()

	server := &DexServer{
		nc:   nc,
		id:   id,
		addr: addr,
	}

	// Announce ourselves
	go server.announcer()

	// Subscribe to orders
	nc.QueueSubscribe("dex.orders", "dex-servers", func(m *nats.Msg) {
		atomic.AddInt64(&server.orders, 1)
		if atomic.LoadInt64(&server.orders)%2 == 0 {
			atomic.AddInt64(&server.trades, 1)
		}
		m.Respond([]byte(`{"status":"accepted"}`))
	})

	// Stats printer
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			orders := atomic.LoadInt64(&server.orders)
			trades := atomic.LoadInt64(&server.trades)
			log.Printf("📊 Stats: Orders=%d, Trades=%d", orders, trades)
		}
	}()

	log.Println("✅ DEX Server ready!")
	select {}
}

func (s *DexServer) announcer() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	announcement := ServerAnnouncement{
		ID:      s.id,
		Addr:    s.addr,
		Type:    "dex-server",
		Started: time.Now(),
	}
	
	for range ticker.C {
		data, _ := json.Marshal(announcement)
		s.nc.Publish("dex.announce", data)
	}
}

func getLocalIP() string {
	addrs, _ := net.InterfaceAddrs()
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				return ipnet.IP.String()
			}
		}
	}
	return "127.0.0.1"
}