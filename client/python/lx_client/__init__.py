"""
LX DEX Trading Client

Multi-protocol Python trading client for LX DEX.
Supports WebSocket and gRPC protocols.

Usage:
    from lx_client import WebSocketClient, GrpcClient, create_client

    # WebSocket client
    ws = WebSocketClient("ws://localhost:8081")
    await ws.connect()
    order = await ws.place_order("BTC-USD", "buy", "limit", 50000, 0.1)

    # gRPC client
    grpc = GrpcClient("localhost:50051")
    await grpc.connect()
    order = await grpc.place_order("BTC-USD", "buy", "limit", 50000, 0.1)

    # Factory function
    client = create_client("ws", url="ws://localhost:8081")
"""

from lx_client.base import (
    TradingClient,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    Order,
    Position,
    Balance,
    OrderBook,
    PriceLevel,
    Trade,
)
from lx_client.websocket_client import WebSocketClient
from lx_client.grpc_client import GrpcClient

__version__ = "0.1.0"
__all__ = [
    "TradingClient",
    "WebSocketClient",
    "GrpcClient",
    "create_client",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    "Order",
    "Position",
    "Balance",
    "OrderBook",
    "PriceLevel",
    "Trade",
]


def create_client(protocol: str, **kwargs) -> TradingClient:
    """
    Factory function to create a trading client.

    Args:
        protocol: Protocol type - "ws" or "grpc"
        **kwargs: Protocol-specific arguments
            For WebSocket: url, api_key, api_secret, verbose
            For gRPC: host, port, credentials, api_key, api_secret

    Returns:
        TradingClient instance

    Raises:
        ValueError: Unknown protocol
    """
    if protocol == "ws":
        return WebSocketClient(
            url=kwargs.get("url", "ws://localhost:8081"),
            api_key=kwargs.get("api_key"),
            api_secret=kwargs.get("api_secret"),
            verbose=kwargs.get("verbose", False),
        )
    elif protocol == "grpc":
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 50051)
        return GrpcClient(
            host=host,
            port=port,
            api_key=kwargs.get("api_key"),
            api_secret=kwargs.get("api_secret"),
            use_tls=kwargs.get("use_tls", False),
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}. Use 'ws' or 'grpc'")
