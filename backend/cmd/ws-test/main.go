package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	pb "github.com/luxfi/dex/backend/pkg/proto/engine"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	grpcEndpoint = flag.String("grpc", "localhost:50051", "gRPC server endpoint")
	wsPort       = flag.Int("port", 8081, "WebSocket server port")
	upgrader     = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}
)

type OrderBookUpdate struct {
	Type   string      `json:"type"`
	Symbol string      `json:"symbol"`
	Bids   []PriceLevel `json:"bids"`
	Asks   []PriceLevel `json:"asks"`
	Timestamp int64    `json:"timestamp"`
}

type PriceLevel struct {
	Price    float64 `json:"price"`
	Quantity float64 `json:"quantity"`
}

func handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	// Connect to gRPC server
	grpcConn, err := grpc.Dial(*grpcEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("Failed to connect to gRPC: %v", err)
		return
	}
	defer grpcConn.Close()

	client := pb.NewEngineServiceClient(grpcConn)
	ctx := context.Background()

	// Start streaming order book
	stream, err := client.StreamOrderBook(ctx, &pb.StreamOrderBookRequest{
		Symbol: "BTC-USD",
		Depth:  10,
	})
	if err != nil {
		log.Printf("Failed to start stream: %v", err)
		return
	}

	log.Printf("WebSocket client connected from %s", r.RemoteAddr)

	// Send updates to WebSocket client
	for {
		update, err := stream.Recv()
		if err != nil {
			log.Printf("Stream error: %v", err)
			break
		}

		// Convert to WebSocket format
		wsUpdate := OrderBookUpdate{
			Type:      "orderbook",
			Symbol:    update.Symbol,
			Timestamp: time.Now().UnixMilli(),
		}

		// Convert bid updates
		for _, bid := range update.BidUpdates {
			wsUpdate.Bids = append(wsUpdate.Bids, PriceLevel{
				Price:    bid.Price,
				Quantity: bid.Quantity,
			})
		}

		// Convert ask updates
		for _, ask := range update.AskUpdates {
			wsUpdate.Asks = append(wsUpdate.Asks, PriceLevel{
				Price:    ask.Price,
				Quantity: ask.Quantity,
			})
		}

		// Send to WebSocket client
		if err := conn.WriteJSON(wsUpdate); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}
}

func main() {
	flag.Parse()

	http.HandleFunc("/ws", handleWebSocket)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(`
<!DOCTYPE html>
<html>
<head>
    <title>LX WebSocket Test</title>
</head>
<body>
    <h1>LX WebSocket Test</h1>
    <div id="status">Disconnected</div>
    <div id="messages"></div>
    <script>
        const ws = new WebSocket('ws://localhost:8081/ws');
        const status = document.getElementById('status');
        const messages = document.getElementById('messages');
        
        ws.onopen = () => {
            status.textContent = 'Connected';
            status.style.color = 'green';
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const msg = document.createElement('pre');
            msg.textContent = JSON.stringify(data, null, 2);
            messages.insertBefore(msg, messages.firstChild);
            if (messages.children.length > 10) {
                messages.removeChild(messages.lastChild);
            }
        };
        
        ws.onerror = (error) => {
            status.textContent = 'Error: ' + error;
            status.style.color = 'red';
        };
        
        ws.onclose = () => {
            status.textContent = 'Disconnected';
            status.style.color = 'red';
        };
    </script>
</body>
</html>
		`))
	})

	log.Printf("WebSocket server starting on http://localhost:%d", *wsPort)
	log.Printf("Connecting to gRPC server at %s", *grpcEndpoint)
	
	if err := http.ListenAndServe(":8081", nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}