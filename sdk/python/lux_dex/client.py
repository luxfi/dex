"""
LX DEX Python Client
"""

import json
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from decimal import Decimal

import requests
import websocket

from .types import (
    OrderType, OrderSide, OrderStatus, TimeInForce,
    Order, Trade, OrderBook, OrderBookLevel, NodeInfo
)
from .exceptions import LXDexException, ConnectionError, OrderError
from .market_data import (
    MarketDataClient, LiquidationMonitor,
    MarketDataSource, LiquidationInfo, SettlementBatch, MarginInfo
)


class JSONRPCClient:
    """JSON-RPC 2.0 client for LX DEX"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.id_counter = 1
    
    def call(self, method: str, params: Dict = None) -> Any:
        """Make a JSON-RPC call"""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.id_counter
        }
        self.id_counter += 1
        
        try:
            response = self.session.post(
                f"{self.base_url}/rpc",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            if "error" in data:
                raise LXDexException(f"RPC Error: {data['error']['message']}")
            
            return data.get("result")
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Connection failed: {str(e)}")


class LXDexClient:
    """Main client for interacting with LX DEX"""
    
    def __init__(
        self,
        json_rpc_url: str = "http://localhost:8080",
        ws_url: str = "ws://localhost:8081",
        api_key: Optional[str] = None
    ):
        """
        Initialize LX DEX client
        
        Args:
            json_rpc_url: URL for JSON-RPC API
            ws_url: URL for WebSocket connection
            api_key: Optional API key for authentication
        """
        self.json_rpc = JSONRPCClient(json_rpc_url)
        self.ws_url = ws_url
        self.api_key = api_key
        self.ws = None
        self.ws_thread = None
        self.ws_callbacks = {}
        self.ws_running = False
        
        # Initialize sub-clients
        self.market_data = MarketDataClient(self.json_rpc)
        self.liquidation_monitor = LiquidationMonitor(None)  # Will set WS later
    
    # Connection Management
    def connect_websocket(self) -> None:
        """Connect to WebSocket for real-time data"""
        if self.ws:
            return
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_ws_message(data)
            except json.JSONDecodeError:
                print(f"Failed to parse WebSocket message: {message}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
            self.ws = None
            self.ws_running = False
        
        def on_open(ws):
            print("WebSocket connected")
            self.ws_running = True
            # Set WebSocket for liquidation monitor
            self.liquidation_monitor.ws = ws
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # Wait for connection
        timeout = 5
        start = time.time()
        while not self.ws_running and time.time() - start < timeout:
            time.sleep(0.1)
        
        if not self.ws_running:
            raise ConnectionError("Failed to connect to WebSocket")
    
    def disconnect(self) -> None:
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()
            self.ws = None
            self.ws_running = False
    
    # Order Management
    def place_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        price: float,
        size: float,
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        post_only: bool = False,
        reduce_only: bool = False
    ) -> Order:
        """
        Place a new order
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            order_type: Type of order
            side: Buy or sell
            price: Order price
            size: Order size
            user_id: User identifier
            client_id: Client order ID
            time_in_force: Order time in force
            post_only: Post-only order flag
            reduce_only: Reduce-only order flag
            
        Returns:
            Order object with order details
        """
        params = {
            "symbol": symbol,
            "type": order_type.value,
            "side": side.value,
            "price": price,
            "size": size,
            "userID": user_id or "default",
            "clientID": client_id,
            "timeInForce": time_in_force.value,
            "postOnly": post_only,
            "reduceOnly": reduce_only
        }
        
        result = self.json_rpc.call("lx_placeOrder", params)
        
        if not result.get("orderId"):
            raise OrderError(f"Order rejected: {result.get('message', 'Unknown error')}")
        
        return Order(
            order_id=result["orderId"],
            symbol=symbol,
            type=order_type,
            side=side,
            price=price,
            size=size,
            status=OrderStatus(result.get("status", "open")),
            user_id=user_id,
            client_id=client_id
        )
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        result = self.json_rpc.call("lx_cancelOrder", {"orderId": order_id})
        
        if not result.get("success"):
            raise OrderError(f"Failed to cancel order: {result.get('message')}")
        
        return True
    
    def get_order(self, order_id: int) -> Order:
        """
        Get order details
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object
        """
        result = self.json_rpc.call("lx_getOrder", {"orderId": order_id})
        
        if not result:
            raise OrderError(f"Order {order_id} not found")
        
        return Order(**result)
    
    # Market Data
    def get_order_book(self, symbol: str = "BTC-USD", depth: int = 10) -> OrderBook:
        """
        Get order book snapshot
        
        Args:
            symbol: Trading pair symbol
            depth: Number of price levels
            
        Returns:
            OrderBook object
        """
        result = self.json_rpc.call("lx_getOrderBook", {
            "symbol": symbol,
            "depth": depth
        })
        
        return OrderBook(
            symbol=result["Symbol"],
            bids=[OrderBookLevel(price=b["Price"], size=b["Size"]) for b in result["Bids"]],
            asks=[OrderBookLevel(price=a["Price"], size=a["Size"]) for a in result["Asks"]],
            timestamp=result["Timestamp"]
        )
    
    def get_best_bid(self, symbol: str = "BTC-USD") -> float:
        """Get best bid price"""
        result = self.json_rpc.call("lx_getBestBid", {"symbol": symbol})
        return result["price"]
    
    def get_best_ask(self, symbol: str = "BTC-USD") -> float:
        """Get best ask price"""
        result = self.json_rpc.call("lx_getBestAsk", {"symbol": symbol})
        return result["price"]
    
    def get_trades(self, symbol: str = "BTC-USD", limit: int = 100) -> List[Trade]:
        """
        Get recent trades
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of trades
            
        Returns:
            List of Trade objects
        """
        result = self.json_rpc.call("lx_getTrades", {
            "symbol": symbol,
            "limit": limit
        })
        
        return [Trade(**trade) for trade in result]
    
    # Node Information
    def get_info(self) -> NodeInfo:
        """Get node information"""
        result = self.json_rpc.call("lx_getInfo")
        return NodeInfo(**result)
    
    def ping(self) -> str:
        """Ping the server"""
        return self.json_rpc.call("lx_ping")
    
    # WebSocket Subscriptions
    def subscribe(self, channel: str, callback: Callable) -> None:
        """
        Subscribe to a WebSocket channel
        
        Args:
            channel: Channel name
            callback: Function to call with updates
        """
        if not self.ws:
            self.connect_websocket()
        
        self.ws_callbacks[channel] = callback
        
        # Send subscription message
        self.ws.send(json.dumps({
            "type": "subscribe",
            "channel": channel
        }))
    
    def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel"""
        if channel in self.ws_callbacks:
            del self.ws_callbacks[channel]
        
        if self.ws:
            self.ws.send(json.dumps({
                "type": "unsubscribe",
                "channel": channel
            }))
    
    def subscribe_order_book(self, symbol: str, callback: Callable[[OrderBook], None]) -> None:
        """Subscribe to order book updates"""
        self.subscribe(f"orderbook:{symbol}", callback)
    
    def subscribe_trades(self, symbol: str, callback: Callable[[Trade], None]) -> None:
        """Subscribe to trade updates"""
        self.subscribe(f"trades:{symbol}", callback)
    
    # Private methods
    def _handle_ws_message(self, msg: Dict) -> None:
        """Handle incoming WebSocket message"""
        channel = msg.get("channel")
        data = msg.get("data")
        
        if channel in self.ws_callbacks:
            callback = self.ws_callbacks[channel]
            try:
                callback(data)
            except Exception as e:
                print(f"Error in WebSocket callback: {e}")
    
    # Utility methods
    @staticmethod
    def format_price(price: float, decimals: int = 2) -> str:
        """Format price with specified decimal places"""
        return f"{price:.{decimals}f}"
    
    @staticmethod
    def format_size(size: float, decimals: int = 8) -> str:
        """Format size with specified decimal places"""
        return f"{size:.{decimals}f}"
    
    @staticmethod
    def calculate_total(price: float, size: float) -> float:
        """Calculate total value"""
        return price * size