import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics
const orderLatency = new Trend('order_latency');
const orderErrors = new Counter('order_errors');
const orderRate = new Rate('order_success_rate');
const wsMessages = new Counter('ws_messages');

// Test configuration
export const options = {
  scenarios: {
    // Scenario 1: Gradual ramp-up
    gradual_load: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '30s', target: 10 },   // Ramp up to 10 users
        { duration: '1m', target: 50 },    // Ramp up to 50 users
        { duration: '2m', target: 100 },   // Ramp up to 100 users
        { duration: '3m', target: 100 },   // Stay at 100 users
        { duration: '1m', target: 0 },     // Ramp down
      ],
      gracefulRampDown: '30s',
    },
    
    // Scenario 2: Spike test
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '10s', target: 1 },    // Baseline
        { duration: '5s', target: 200 },   // Spike to 200 users
        { duration: '30s', target: 200 },  // Stay at peak
        { duration: '5s', target: 1 },     // Back to baseline
        { duration: '10s', target: 1 },    // Recovery
      ],
      gracefulRampDown: '10s',
      startTime: '10m',  // Start after gradual_load
    },
    
    // Scenario 3: Constant load
    constant_load: {
      executor: 'constant-vus',
      vus: 50,
      duration: '5m',
      startTime: '15m',  // Start after spike_test
    },
    
    // Scenario 4: WebSocket connections
    websocket_test: {
      executor: 'constant-vus',
      vus: 20,
      duration: '5m',
      exec: 'websocketScenario',
      startTime: '0s',
    },
  },
  
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],  // 95% of requests under 500ms
    order_latency: ['p(95)<300', 'p(99)<500'],       // Order latency thresholds
    order_success_rate: ['rate>0.95'],               // 95% success rate
    http_req_failed: ['rate<0.05'],                  // Less than 5% failure rate
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const WS_URL = __ENV.WS_URL || 'ws://localhost:8081';

// Helper function to generate order
function generateOrder() {
  const side = Math.random() > 0.5 ? 0 : 1;  // 0 = buy, 1 = sell
  const basePrice = 50000;
  const priceVariation = (Math.random() - 0.5) * 1000;  // +/- $500
  
  return {
    jsonrpc: '2.0',
    method: 'lx_placeOrder',
    params: {
      symbol: 'BTC-USD',
      type: 0,  // Limit order
      side: side,
      price: basePrice + priceVariation,
      size: Math.random() * 0.1 + 0.01,  // 0.01 to 0.11 BTC
      userID: `user-${__VU}-${__ITER}`,
      clientID: `order-${Date.now()}-${Math.random()}`,
    },
    id: Math.floor(Math.random() * 1000000),
  };
}

// Main test scenario
export default function () {
  // Place an order
  const order = generateOrder();
  const startTime = Date.now();
  
  const response = http.post(
    `${BASE_URL}/rpc`,
    JSON.stringify(order),
    {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: '10s',
    }
  );
  
  const latency = Date.now() - startTime;
  orderLatency.add(latency);
  
  // Check response
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'has result': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.result && body.result.orderId;
      } catch (e) {
        return false;
      }
    },
    'no error': (r) => {
      try {
        const body = JSON.parse(r.body);
        return !body.error;
      } catch (e) {
        return false;
      }
    },
  });
  
  orderRate.add(success);
  if (!success) {
    orderErrors.add(1);
  }
  
  // Get order book occasionally
  if (Math.random() < 0.1) {  // 10% of iterations
    const orderBookReq = {
      jsonrpc: '2.0',
      method: 'lx_getOrderBook',
      params: {
        symbol: 'BTC-USD',
        depth: 20,
      },
      id: Math.floor(Math.random() * 1000000),
    };
    
    const obResponse = http.post(
      `${BASE_URL}/rpc`,
      JSON.stringify(orderBookReq),
      {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: '5s',
      }
    );
    
    check(obResponse, {
      'orderbook status 200': (r) => r.status === 200,
      'orderbook has bids/asks': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.result && body.result.Bids && body.result.Asks;
        } catch (e) {
          return false;
        }
      },
    });
  }
  
  // Get recent trades occasionally
  if (Math.random() < 0.05) {  // 5% of iterations
    const tradesReq = {
      jsonrpc: '2.0',
      method: 'lx_getTrades',
      params: {
        symbol: 'BTC-USD',
        limit: 10,
      },
      id: Math.floor(Math.random() * 1000000),
    };
    
    http.post(
      `${BASE_URL}/rpc`,
      JSON.stringify(tradesReq),
      {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: '5s',
      }
    );
  }
  
  // Small delay between requests
  sleep(Math.random() * 0.5);  // 0-500ms random delay
}

// WebSocket scenario
export function websocketScenario() {
  const url = `${WS_URL}/ws`;
  
  const response = ws.connect(url, {}, function (socket) {
    socket.on('open', () => {
      console.log(`VU ${__VU}: WebSocket connected`);
      
      // Subscribe to order book
      socket.send(JSON.stringify({
        type: 'subscribe',
        channels: ['orderbook:BTC-USD', 'trades:BTC-USD'],
      }));
    });
    
    socket.on('message', (data) => {
      wsMessages.add(1);
      
      try {
        const msg = JSON.parse(data);
        
        // Validate message structure
        check(msg, {
          'ws message has type': (m) => m.type !== undefined,
          'ws message has timestamp': (m) => m.timestamp !== undefined,
        });
        
        // Occasionally place order via WebSocket
        if (Math.random() < 0.01) {  // 1% chance
          const order = generateOrder();
          socket.send(JSON.stringify({
            type: 'order',
            data: order.params,
          }));
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    });
    
    socket.on('error', (e) => {
      console.error(`VU ${__VU}: WebSocket error:`, e);
    });
    
    socket.on('close', () => {
      console.log(`VU ${__VU}: WebSocket closed`);
    });
    
    // Keep connection alive for duration
    socket.setTimeout(() => {
      socket.close();
    }, 60000);  // 60 seconds
  });
  
  check(response, {
    'ws connection successful': (r) => r && r.status === 101,
  });
}

// Teardown function
export function teardown(data) {
  console.log('Load test completed');
  console.log('Summary:');
  console.log(`  Total orders placed: ${data.metrics.order_latency.values.count}`);
  console.log(`  Order errors: ${data.metrics.order_errors.values.count}`);
  console.log(`  WebSocket messages: ${data.metrics.ws_messages.values.count}`);
}