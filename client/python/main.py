#!/usr/bin/env python3
"""
LX DEX Trading Client

Multi-protocol trading client for LX DEX.
Supports WebSocket and gRPC protocols.

Usage:
    # Interactive mode (WebSocket)
    python main.py -i

    # Interactive mode (gRPC)
    python main.py --protocol grpc -i

    # Single command
    python main.py place_order BTC-USD buy limit 50000 0.1

    # As a library
    from lx_client import WebSocketClient, GrpcClient, create_client
"""

import argparse
import asyncio
import json
import sys
from typing import Optional

# Import client library
try:
    from lx_client import (
        create_client,
        TradingClient,
        WebSocketClient,
        GrpcClient,
    )
except ImportError:
    # Add parent to path for direct execution
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lx_client import (
        create_client,
        TradingClient,
        WebSocketClient,
        GrpcClient,
    )


def print_help():
    """Print interactive help."""
    print("""
LX DEX Trading Client Commands:

  place_order <symbol> <side> <type> <price> <size>
    Place a new order
    Example: place_order BTC-USD buy limit 50000 0.1

  cancel_order <order_id>
    Cancel an existing order
    Example: cancel_order 12345

  get_order <order_id>
    Get order details
    Example: get_order 12345

  get_orders [symbol] [status]
    List orders (optionally filtered)
    Example: get_orders BTC-USD open

  get_orderbook <symbol> [depth]
    Get order book snapshot
    Example: get_orderbook BTC-USD 20

  get_positions
    List all positions

  get_balance <asset>
    Get balance for an asset
    Example: get_balance USD

  auth <api_key> <api_secret>
    Authenticate with credentials

  ping
    Ping server (gRPC only)

  info
    Get node info (gRPC only)

  help
    Show this help message

  quit / exit
    Exit the client
""")


async def run_interactive(client: TradingClient):
    """Run interactive mode."""
    print("LX DEX Trading Client - Type 'help' for commands")
    print(f"Protocol: {'WebSocket' if isinstance(client, WebSocketClient) else 'gRPC'}")
    print(f"Connected: {client.connected}")
    print()

    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("> ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "help":
                print_help()

            elif cmd in ("quit", "exit"):
                print("Goodbye")
                break

            elif cmd == "auth":
                if len(parts) < 3:
                    print("Usage: auth <api_key> <api_secret>")
                else:
                    if await client.authenticate(parts[1], parts[2]):
                        print("Authenticated successfully")
                    else:
                        print("Authentication failed")

            elif cmd == "place_order":
                if len(parts) < 6:
                    print("Usage: place_order <symbol> <side> <type> <price> <size>")
                else:
                    order = await client.place_order(
                        symbol=parts[1],
                        side=parts[2],
                        order_type=parts[3],
                        price=float(parts[4]),
                        size=float(parts[5]),
                    )
                    print(f"Order placed: {json.dumps(order.to_dict(), indent=2)}")

            elif cmd == "cancel_order":
                if len(parts) < 2:
                    print("Usage: cancel_order <order_id>")
                else:
                    success = await client.cancel_order(int(parts[1]))
                    print(f"Cancel {'successful' if success else 'failed'}")

            elif cmd == "get_order":
                if len(parts) < 2:
                    print("Usage: get_order <order_id>")
                else:
                    order = await client.get_order(int(parts[1]))
                    if order:
                        print(json.dumps(order.to_dict(), indent=2))
                    else:
                        print("Order not found")

            elif cmd == "get_orders":
                symbol = parts[1] if len(parts) > 1 else None
                status = parts[2] if len(parts) > 2 else None
                orders = await client.get_orders(symbol=symbol, status=status)
                if orders:
                    for order in orders:
                        print(json.dumps(order.to_dict(), indent=2))
                else:
                    print("No orders found")

            elif cmd == "get_orderbook":
                if len(parts) < 2:
                    print("Usage: get_orderbook <symbol> [depth]")
                else:
                    depth = int(parts[2]) if len(parts) > 2 else 20
                    book = await client.get_orderbook(parts[1], depth)
                    print(f"OrderBook [{book.symbol}]:")
                    print(f"  Bids ({len(book.bids)} levels):")
                    for bid in book.bids[:5]:
                        print(f"    {bid.price:.2f} @ {bid.size:.4f}")
                    print(f"  Asks ({len(book.asks)} levels):")
                    for ask in book.asks[:5]:
                        print(f"    {ask.price:.2f} @ {ask.size:.4f}")

            elif cmd == "get_positions":
                positions = await client.get_positions()
                if positions:
                    for pos in positions:
                        print(f"  {pos.symbol}: {pos.size} @ {pos.entry_price:.2f} "
                              f"(PnL: {pos.pnl:.2f})")
                else:
                    print("No positions")

            elif cmd == "get_balance":
                if len(parts) < 2:
                    print("Usage: get_balance <asset>")
                else:
                    balance = await client.get_balance(parts[1])
                    if balance:
                        print(f"  {balance.asset}:")
                        print(f"    Available: {balance.available:.4f}")
                        print(f"    Locked: {balance.locked:.4f}")
                        print(f"    Total: {balance.total:.4f}")
                    else:
                        print("Balance not found")

            elif cmd == "ping":
                if isinstance(client, GrpcClient):
                    latency = await client.ping()
                    print(f"Pong! Latency: {latency}ms")
                else:
                    print("Ping only available with gRPC protocol")

            elif cmd == "info":
                if isinstance(client, GrpcClient):
                    info = await client.get_node_info()
                    print(json.dumps(info, indent=2))
                else:
                    print("Node info only available with gRPC protocol")

            else:
                print(f"Unknown command: {cmd}. Type 'help' for commands.")

        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")


async def run_command(client: TradingClient, args: argparse.Namespace):
    """Run a single command."""
    cmd = args.command

    if cmd == "place_order":
        order = await client.place_order(
            symbol=args.symbol,
            side=args.side,
            order_type=args.type,
            price=args.price,
            size=args.size,
        )
        print(json.dumps(order.to_dict(), indent=2))

    elif cmd == "cancel_order":
        success = await client.cancel_order(args.order_id)
        if success:
            print(json.dumps({"success": True, "order_id": args.order_id}, indent=2))
        else:
            print(json.dumps({"success": False, "order_id": args.order_id}, indent=2))
            sys.exit(1)

    elif cmd == "get_order":
        order = await client.get_order(args.order_id)
        if order:
            print(json.dumps(order.to_dict(), indent=2))
        else:
            print(json.dumps({"error": "Order not found"}, indent=2))
            sys.exit(1)

    elif cmd == "get_orders":
        orders = await client.get_orders(
            symbol=getattr(args, "symbol", None),
            status=getattr(args, "status", None),
        )
        print(json.dumps([o.to_dict() for o in orders], indent=2))

    elif cmd == "get_orderbook":
        depth = getattr(args, "depth", 20)
        book = await client.get_orderbook(args.symbol, depth)
        result = {
            "symbol": book.symbol,
            "bids": [{"price": b.price, "size": b.size} for b in book.bids],
            "asks": [{"price": a.price, "size": a.size} for a in book.asks],
            "timestamp": book.timestamp,
        }
        print(json.dumps(result, indent=2))

    elif cmd == "get_positions":
        positions = await client.get_positions()
        result = [
            {
                "symbol": p.symbol,
                "size": p.size,
                "entry_price": p.entry_price,
                "mark_price": p.mark_price,
                "pnl": p.pnl,
                "margin": p.margin,
            }
            for p in positions
        ]
        print(json.dumps(result, indent=2))

    elif cmd == "get_balance":
        balance = await client.get_balance(args.asset)
        if balance:
            print(json.dumps({
                "asset": balance.asset,
                "available": balance.available,
                "locked": balance.locked,
                "total": balance.total,
            }, indent=2))
        else:
            print(json.dumps({"error": "Balance not found"}, indent=2))
            sys.exit(1)

    elif cmd == "ping":
        if isinstance(client, GrpcClient):
            latency = await client.ping()
            print(json.dumps({"latency_ms": latency}, indent=2))
        else:
            print(json.dumps({"error": "Ping only available with gRPC"}, indent=2))
            sys.exit(1)

    elif cmd == "info":
        if isinstance(client, GrpcClient):
            info = await client.get_node_info()
            print(json.dumps(info, indent=2))
        else:
            print(json.dumps({"error": "Info only available with gRPC"}, indent=2))
            sys.exit(1)


async def main_async():
    """Async main entry point."""
    parser = argparse.ArgumentParser(
        description="LX DEX Trading Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i                                    # Interactive mode (WebSocket)
  %(prog)s --protocol grpc -i                    # Interactive mode (gRPC)
  %(prog)s place_order BTC-USD buy limit 50000 0.1
  %(prog)s cancel_order 12345
  %(prog)s get_orderbook BTC-USD
  %(prog)s get_positions

Environment:
  LX_API_KEY       API key for authentication
  LX_API_SECRET    API secret for authentication
""",
    )

    # Connection options
    parser.add_argument(
        "--protocol", "-p",
        choices=["ws", "grpc"],
        default="ws",
        help="Protocol to use (default: ws)",
    )
    parser.add_argument(
        "-u", "--url",
        help="WebSocket URL (default: ws://localhost:8081)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="gRPC host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="gRPC port (default: 50051)",
    )
    parser.add_argument(
        "--tls",
        action="store_true",
        help="Enable TLS for gRPC",
    )

    # Authentication
    parser.add_argument(
        "-k", "--key",
        help="API key for authentication",
    )
    parser.add_argument(
        "-s", "--secret",
        help="API secret for authentication",
    )

    # Mode options
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # place_order
    p_order = subparsers.add_parser("place_order", help="Place a new order")
    p_order.add_argument("symbol", help="Trading pair (e.g., BTC-USD)")
    p_order.add_argument("side", choices=["buy", "sell"], help="Order side")
    p_order.add_argument(
        "type",
        choices=["limit", "market", "stop", "stop_limit"],
        help="Order type",
    )
    p_order.add_argument("price", type=float, help="Order price")
    p_order.add_argument("size", type=float, help="Order size")

    # cancel_order
    p_cancel = subparsers.add_parser("cancel_order", help="Cancel an order")
    p_cancel.add_argument("order_id", type=int, help="Order ID")

    # get_order
    p_get = subparsers.add_parser("get_order", help="Get order details")
    p_get.add_argument("order_id", type=int, help="Order ID")

    # get_orders
    p_orders = subparsers.add_parser("get_orders", help="List orders")
    p_orders.add_argument("symbol", nargs="?", help="Filter by symbol")
    p_orders.add_argument("status", nargs="?", help="Filter by status")

    # get_orderbook
    p_book = subparsers.add_parser("get_orderbook", help="Get order book")
    p_book.add_argument("symbol", help="Trading pair")
    p_book.add_argument("depth", type=int, nargs="?", default=20, help="Depth")

    # get_positions
    subparsers.add_parser("get_positions", help="List positions")

    # get_balance
    p_balance = subparsers.add_parser("get_balance", help="Get balance")
    p_balance.add_argument("asset", help="Asset symbol")

    # ping (gRPC only)
    subparsers.add_parser("ping", help="Ping server (gRPC only)")

    # info (gRPC only)
    subparsers.add_parser("info", help="Get node info (gRPC only)")

    args = parser.parse_args()

    # Get API credentials from args or environment
    import os
    api_key = args.key or os.environ.get("LX_API_KEY")
    api_secret = args.secret or os.environ.get("LX_API_SECRET")

    # Create client based on protocol
    if args.protocol == "ws":
        url = args.url or "ws://localhost:8081"
        client = WebSocketClient(
            url=url,
            api_key=api_key,
            api_secret=api_secret,
            verbose=args.verbose,
        )
    else:
        client = GrpcClient(
            host=args.host,
            port=args.port,
            api_key=api_key,
            api_secret=api_secret,
            use_tls=args.tls,
        )

    # Connect
    try:
        if not await client.connect():
            print("Failed to connect to server", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Connection error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Authenticate if credentials provided
        if api_key and api_secret:
            try:
                if not await client.authenticate():
                    print("Authentication failed", file=sys.stderr)
                    sys.exit(1)
                if args.verbose:
                    print("Authenticated")
            except Exception as e:
                print(f"Authentication error: {e}", file=sys.stderr)
                sys.exit(1)

        # Run interactive or command mode
        if args.interactive or not args.command:
            await run_interactive(client)
        else:
            await run_command(client, args)

    finally:
        await client.disconnect()


def main():
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
