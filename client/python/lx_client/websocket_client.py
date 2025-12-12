"""
LX Trading Client - WebSocket Implementation

Async WebSocket client for LX trading API.
"""

import asyncio
import json
import uuid
from typing import Optional, List, Dict, Any, AsyncIterator

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    websockets = None
    WebSocketClientProtocol = None

from lx_client.base import (
    TradingClient,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    OrderBook,
    PriceLevel,
    Position,
    Balance,
    Trade,
)


class WebSocketClient(TradingClient):
    """
    WebSocket-based trading client for LX.

    Uses async/await for all operations with websockets library.

    Example:
        async with WebSocketClient("ws://localhost:8081") as client:
            order = await client.place_order("BTC-USD", "buy", "limit", 50000, 0.1)
            print(f"Order placed: {order.order_id}")
    """

    def __init__(
        self,
        url: str = "ws://localhost:8081",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize WebSocket client.

        Args:
            url: WebSocket server URL
            api_key: API key for authentication
            api_secret: API secret for authentication
            verbose: Enable verbose logging
        """
        super().__init__(api_key, api_secret)

        if websockets is None:
            raise ImportError(
                "websockets library required. Install with: pip install websockets"
            )

        self.url = url
        self.verbose = verbose
        self._ws: Optional[WebSocketClientProtocol] = None
        self._responses: Dict[str, asyncio.Future] = {}
        self._recv_task: Optional[asyncio.Task] = None
        self._subscriptions: Dict[str, asyncio.Queue] = {}

    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            self._ws = await websockets.connect(
                self.url,
                ping_interval=30,
                ping_timeout=10,
            )
            self._recv_task = asyncio.create_task(self._recv_loop())

            # Wait for connected message
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                data = json.loads(msg)
                if self.verbose:
                    print(f"<< {json.dumps(data, indent=2)}")
                if data.get("type") == "connected":
                    self._connected = True
                    return True
            except asyncio.TimeoutError:
                pass

            self._connected = True
            return True
        except Exception as e:
            if self.verbose:
                print(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self._connected = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        # Cancel pending responses
        for future in self._responses.values():
            if not future.done():
                future.cancel()
        self._responses.clear()

    async def _recv_loop(self) -> None:
        """Background task to receive messages."""
        while self._connected and self._ws:
            try:
                data = await self._ws.recv()
                msg = json.loads(data)

                if self.verbose:
                    print(f"<< {json.dumps(msg, indent=2)}")

                req_id = msg.get("request_id", "")
                msg_type = msg.get("type", "")

                # Handle auth response
                if msg_type == "auth_success":
                    self._authenticated = True

                # Route response to waiting caller
                if req_id and req_id in self._responses:
                    future = self._responses.pop(req_id)
                    if not future.done():
                        future.set_result(msg)

                # Route subscription updates
                elif msg_type in ("orderbook", "trade", "order_update"):
                    symbol = msg.get("data", {}).get("symbol", "")
                    key = f"{msg_type}:{symbol}"
                    if key in self._subscriptions:
                        await self._subscriptions[key].put(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._connected and self.verbose:
                    print(f"Receive error: {e}")
                break

    async def _send(self, msg_type: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Send message and wait for response."""
        if not self._ws:
            raise ConnectionError("Not connected")

        req_id = str(uuid.uuid4())[:8]
        msg = {"type": msg_type, "request_id": req_id}
        if data:
            msg.update(data)

        if self.verbose:
            print(f">> {json.dumps(msg, indent=2)}")

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._responses[req_id] = future

        try:
            await self._ws.send(json.dumps(msg))
            return await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            self._responses.pop(req_id, None)
            raise TimeoutError(f"Request {msg_type} timed out")

    async def authenticate(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> bool:
        """Authenticate with API credentials."""
        key = api_key or self.api_key
        secret = api_secret or self.api_secret

        if not key or not secret:
            raise ValueError("API key and secret required")

        resp = await self._send("auth", {"apiKey": key, "apiSecret": secret})
        if resp.get("type") == "auth_success":
            self._authenticated = True
            return True
        return False

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        size: float,
        client_id: Optional[str] = None,
        time_in_force: str = "gtc",
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> Order:
        """Place a new order."""
        order_data = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "price": price,
            "size": size,
            "time_in_force": time_in_force,
            "post_only": post_only,
            "reduce_only": reduce_only,
        }
        if client_id:
            order_data["client_id"] = client_id

        resp = await self._send("place_order", {"order": order_data})

        if resp.get("error"):
            raise ValueError(resp["error"])

        data = resp.get("data", resp)
        return Order(
            order_id=data.get("order_id", data.get("orderID", 0)),
            symbol=symbol,
            order_type=OrderType(order_type),
            side=OrderSide(side),
            price=price,
            size=size,
            status=OrderStatus(data.get("status", "open")),
            client_id=client_id or "",
            time_in_force=TimeInForce(time_in_force),
            post_only=post_only,
            reduce_only=reduce_only,
        )

    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an existing order."""
        resp = await self._send("cancel_order", {"orderID": order_id})
        if resp.get("error"):
            return False
        return resp.get("success", True)

    async def get_order(self, order_id: int) -> Optional[Order]:
        """Get order details."""
        resp = await self._send("get_order", {"orderID": order_id})
        if resp.get("error"):
            return None
        return self._parse_order(resp.get("data", resp))

    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get list of orders."""
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if status:
            params["status"] = status

        resp = await self._send("get_orders", params)
        if resp.get("error"):
            return []

        orders_data = resp.get("data", {}).get("orders", [])
        return [self._parse_order(o) for o in orders_data]

    async def get_orderbook(
        self,
        symbol: str,
        depth: int = 20,
    ) -> OrderBook:
        """Get order book snapshot."""
        resp = await self._send("subscribe", {"symbols": [symbol], "depth": depth})

        # Wait briefly for orderbook data
        await asyncio.sleep(0.5)

        data = resp.get("data", {})
        return OrderBook(
            symbol=symbol,
            bids=[
                PriceLevel(price=b.get("price", 0), size=b.get("size", 0))
                for b in data.get("bids", [])[:depth]
            ],
            asks=[
                PriceLevel(price=a.get("price", 0), size=a.get("size", 0))
                for a in data.get("asks", [])[:depth]
            ],
            timestamp=data.get("timestamp", 0),
        )

    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        resp = await self._send("get_positions", {})
        if resp.get("error"):
            return []

        positions_data = resp.get("data", {}).get("positions", [])
        return [
            Position(
                symbol=p.get("symbol", ""),
                size=p.get("size", 0),
                entry_price=p.get("entry_price", 0),
                mark_price=p.get("mark_price", 0),
                pnl=p.get("pnl", 0),
                margin=p.get("margin", 0),
            )
            for p in positions_data
        ]

    async def get_balance(self, asset: str) -> Optional[Balance]:
        """Get balance for an asset."""
        resp = await self._send("get_balances", {"asset": asset})
        if resp.get("error"):
            return None

        data = resp.get("data", {})
        return Balance(
            asset=asset,
            available=data.get("available", 0),
            locked=data.get("locked", 0),
            total=data.get("total", 0),
        )

    async def stream_orderbook(
        self,
        symbol: str,
        depth: int = 20,
    ) -> AsyncIterator[OrderBook]:
        """Stream order book updates."""
        key = f"orderbook:{symbol}"
        self._subscriptions[key] = asyncio.Queue()

        try:
            await self._send("subscribe", {"symbols": [symbol], "depth": depth})

            while self._connected:
                try:
                    msg = await asyncio.wait_for(
                        self._subscriptions[key].get(),
                        timeout=30.0,
                    )
                    data = msg.get("data", {})
                    yield OrderBook(
                        symbol=symbol,
                        bids=[
                            PriceLevel(price=b.get("price", 0), size=b.get("size", 0))
                            for b in data.get("bids", [])[:depth]
                        ],
                        asks=[
                            PriceLevel(price=a.get("price", 0), size=a.get("size", 0))
                            for a in data.get("asks", [])[:depth]
                        ],
                        timestamp=data.get("timestamp", 0),
                    )
                except asyncio.TimeoutError:
                    continue
        finally:
            self._subscriptions.pop(key, None)
            await self._send("unsubscribe", {"symbols": [symbol]})

    async def stream_trades(
        self,
        symbol: str,
    ) -> AsyncIterator[Trade]:
        """Stream trades."""
        key = f"trade:{symbol}"
        self._subscriptions[key] = asyncio.Queue()

        try:
            await self._send("subscribe_trades", {"symbol": symbol})

            while self._connected:
                try:
                    msg = await asyncio.wait_for(
                        self._subscriptions[key].get(),
                        timeout=30.0,
                    )
                    data = msg.get("data", {})
                    yield Trade(
                        trade_id=data.get("trade_id", 0),
                        symbol=symbol,
                        price=data.get("price", 0),
                        size=data.get("size", 0),
                        side=OrderSide(data.get("side", "buy")),
                        timestamp=data.get("timestamp", 0),
                    )
                except asyncio.TimeoutError:
                    continue
        finally:
            self._subscriptions.pop(key, None)
            await self._send("unsubscribe_trades", {"symbol": symbol})

    def _parse_order(self, data: Dict) -> Order:
        """Parse order from response data."""
        return Order(
            order_id=data.get("order_id", data.get("orderID", 0)),
            symbol=data.get("symbol", ""),
            order_type=OrderType(data.get("type", "limit")),
            side=OrderSide(data.get("side", "buy")),
            price=data.get("price", 0),
            size=data.get("size", 0),
            filled=data.get("filled", 0),
            remaining=data.get("remaining", data.get("size", 0)),
            status=OrderStatus(data.get("status", "open")),
            user_id=data.get("user_id", ""),
            client_id=data.get("client_id", ""),
            timestamp=data.get("timestamp", 0),
            time_in_force=TimeInForce(data.get("time_in_force", "gtc")),
            post_only=data.get("post_only", False),
            reduce_only=data.get("reduce_only", False),
        )
