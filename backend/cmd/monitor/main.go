package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/nats-io/nats.go"
)

type Monitor struct {
	nc          *nats.Conn
	stats       Stats
	clients     map[*websocket.Conn]bool
	mu          sync.RWMutex
	upgrader    websocket.Upgrader
}

type Stats struct {
	Orders       int64     `json:"orders"`
	Trades       int64     `json:"trades"`
	OrdersPerSec float64   `json:"orders_per_sec"`
	TradesPerSec float64   `json:"trades_per_sec"`
	ActiveNodes  int       `json:"active_nodes"`
	Memory       MemStats  `json:"memory"`
	Timestamp    time.Time `json:"timestamp"`
}

type MemStats struct {
	Alloc      uint64 `json:"alloc_mb"`
	Sys        uint64 `json:"sys_mb"`
	NumGC      uint32 `json:"num_gc"`
	NumCPU     int    `json:"num_cpu"`
	Goroutines int    `json:"goroutines"`
}

var dashboardHTML = `
<!DOCTYPE html>
<html>
<head>
    <title>LX DEX Monitor</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .metric {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .label {
            font-size: 0.9em;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .chart {
            height: 200px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            position: relative;
            overflow: hidden;
        }
        .bar {
            position: absolute;
            bottom: 0;
            width: 3px;
            background: linear-gradient(to top, #00f260 0%, #0575e6 100%);
        }
        .status {
            padding: 5px 10px;
            border-radius: 20px;
            display: inline-block;
            font-size: 0.9em;
        }
        .status.online {
            background: #4caf50;
        }
        .status.offline {
            background: #f44336;
        }
        #log {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 0.9em;
            max-height: 300px;
            overflow-y: auto;
        }
        .log-entry {
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 LX DEX Real-Time Monitor</h1>
        
        <div class="grid">
            <div class="card">
                <div class="label">Total Orders</div>
                <div class="metric" id="orders">0</div>
                <div class="label">Orders/sec: <span id="orders-rate">0</span></div>
            </div>
            
            <div class="card">
                <div class="label">Total Trades</div>
                <div class="metric" id="trades">0</div>
                <div class="label">Trades/sec: <span id="trades-rate">0</span></div>
            </div>
            
            <div class="card">
                <div class="label">Memory Usage</div>
                <div class="metric" id="memory">0 MB</div>
                <div class="label">GC Runs: <span id="gc">0</span></div>
            </div>
            
            <div class="card">
                <div class="label">System</div>
                <div class="metric" id="goroutines">0</div>
                <div class="label">Goroutines | CPUs: <span id="cpus">0</span></div>
            </div>
        </div>
        
        <div class="card">
            <div class="label">Orders/sec History (Last 60 seconds)</div>
            <div class="chart" id="chart">
                <canvas id="canvas" width="1200" height="200"></canvas>
            </div>
        </div>
        
        <div class="card">
            <div class="label">Live Log</div>
            <div id="log"></div>
        </div>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:8080/ws');
        const history = [];
        const maxHistory = 60;
        
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update metrics
            document.getElementById('orders').textContent = data.orders.toLocaleString();
            document.getElementById('trades').textContent = data.trades.toLocaleString();
            document.getElementById('orders-rate').textContent = Math.round(data.orders_per_sec);
            document.getElementById('trades-rate').textContent = Math.round(data.trades_per_sec);
            document.getElementById('memory').textContent = data.memory.alloc_mb + ' MB';
            document.getElementById('gc').textContent = data.memory.num_gc;
            document.getElementById('goroutines').textContent = data.memory.goroutines;
            document.getElementById('cpus').textContent = data.memory.num_cpu;
            
            // Update history
            history.push(data.orders_per_sec);
            if (history.length > maxHistory) {
                history.shift();
            }
            
            // Draw chart
            drawChart();
            
            // Add log entry
            addLogEntry(data);
        };
        
        function drawChart() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (history.length < 2) return;
            
            const maxValue = Math.max(...history, 1);
            const barWidth = canvas.width / maxHistory;
            
            // Draw grid lines
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {
                const y = (canvas.height / 5) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
            
            // Draw bars
            const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
            gradient.addColorStop(0, '#00f260');
            gradient.addColorStop(1, '#0575e6');
            ctx.fillStyle = gradient;
            
            history.forEach((value, index) => {
                const height = (value / maxValue) * canvas.height;
                const x = index * barWidth;
                ctx.fillRect(x, canvas.height - height, barWidth - 1, height);
            });
        }
        
        function addLogEntry(data) {
            const log = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            const time = new Date(data.timestamp).toLocaleTimeString();
            entry.textContent = time + ' | Orders: ' + data.orders + ' | Rate: ' + 
                               Math.round(data.orders_per_sec) + '/s | Memory: ' + 
                               data.memory.alloc_mb + 'MB';
            log.insertBefore(entry, log.firstChild);
            
            // Keep only last 20 entries
            while (log.children.length > 20) {
                log.removeChild(log.lastChild);
            }
        }
        
        ws.onopen = () => {
            addLogEntry({ timestamp: new Date(), orders: 0, orders_per_sec: 0, memory: { alloc_mb: 0 } });
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    </script>
</body>
</html>
`

func main() {
	monitor := &Monitor{
		clients:  make(map[*websocket.Conn]bool),
		upgrader: websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }},
	}

	// Connect to NATS
	nc, err := nats.Connect(nats.DefaultURL)
	if err != nil {
		log.Printf("Running without NATS: %v", err)
	} else {
		monitor.nc = nc
		defer nc.Close()
	}

	// Start background workers
	go monitor.collectStats()
	go monitor.broadcastStats()

	// HTTP handlers
	http.HandleFunc("/", monitor.handleDashboard)
	http.HandleFunc("/ws", monitor.handleWebSocket)
	http.HandleFunc("/api/stats", monitor.handleAPI)

	fmt.Println("📊 LX DEX Monitor")
	fmt.Println("=================")
	fmt.Println("Dashboard: http://localhost:8080")
	fmt.Println("WebSocket: ws://localhost:8080/ws")
	fmt.Println("API: http://localhost:8080/api/stats")
	
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func (m *Monitor) handleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(dashboardHTML))
}

func (m *Monitor) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := m.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	m.mu.Lock()
	m.clients[conn] = true
	m.mu.Unlock()

	defer func() {
		m.mu.Lock()
		delete(m.clients, conn)
		m.mu.Unlock()
	}()

	// Keep connection alive
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			break
		}
	}
}

func (m *Monitor) handleAPI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	m.mu.RLock()
	json.NewEncoder(w).Encode(m.stats)
	m.mu.RUnlock()
}

func (m *Monitor) collectStats() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var lastOrders, lastTrades int64
	lastTime := time.Now()

	for range ticker.C {
		now := time.Now()
		deltaTime := now.Sub(lastTime).Seconds()

		// Get memory stats
		var memStats runtime.MemStats
		runtime.ReadMemStats(&memStats)

		// Calculate rates
		orders := atomic.LoadInt64(&m.stats.Orders)
		trades := atomic.LoadInt64(&m.stats.Trades)
		
		ordersPerSec := float64(orders-lastOrders) / deltaTime
		tradesPerSec := float64(trades-lastTrades) / deltaTime

		// Update stats
		m.mu.Lock()
		m.stats.OrdersPerSec = ordersPerSec
		m.stats.TradesPerSec = tradesPerSec
		m.stats.Memory = MemStats{
			Alloc:      memStats.Alloc / 1024 / 1024,
			Sys:        memStats.Sys / 1024 / 1024,
			NumGC:      memStats.NumGC,
			NumCPU:     runtime.NumCPU(),
			Goroutines: runtime.NumGoroutine(),
		}
		m.stats.Timestamp = now
		m.mu.Unlock()

		lastOrders = orders
		lastTrades = trades
		lastTime = now
	}
}

func (m *Monitor) broadcastStats() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		m.mu.RLock()
		data, _ := json.Marshal(m.stats)
		clients := make([]*websocket.Conn, 0, len(m.clients))
		for client := range m.clients {
			clients = append(clients, client)
		}
		m.mu.RUnlock()

		// Broadcast to all connected clients
		for _, client := range clients {
			if err := client.WriteMessage(websocket.TextMessage, data); err != nil {
				client.Close()
				m.mu.Lock()
				delete(m.clients, client)
				m.mu.Unlock()
			}
		}
	}
}

// Simulate some data for demo
func init() {
	go func() {
		time.Sleep(2 * time.Second)
		// Demo data generation commented out - will use real data
		// for {
		//     atomic.AddInt64(&Stats{}.Orders, int64(100+rand.Intn(500)))
		//     atomic.AddInt64(&Stats{}.Trades, int64(50+rand.Intn(250)))
		//     time.Sleep(100 * time.Millisecond)
		// }
	}()
}