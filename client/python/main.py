#!/usr/bin/env python3
"""
LX DEX CLI Client

Command-line trading interface for LX DEX WebSocket API.
"""

import argparse
import json
import sys
import threading
import time
from typing import Optional, Dict, Any
import uuid

try:
    import websocket
except ImportError:
    print("Error: websocket-client required. Install with: pip install websocket-client")
    sys.exit(1)


class LXClient:
    """WebSocket client for LX DEX."""

    def __init__(self, url: str, verbose: bool = False):
        self.url = url
        self.verbose = verbose
        self.ws: Optional[websocket.WebSocket] = None
        self.connected = False
        self.authenticated = False
        self.responses: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            self.ws = websocket.create_connection(
                self.url,
                timeout=10
            )
            self._running = True
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()

            # Wait for connected message
            timeout = time.time() + 5
            while not self.connected and time.time() < timeout:
                time.sleep(0.1)

            return self.connected
        except Exception as e:
            print(f"Connection failed: {e}", file=sys.stderr)
            return False

    def _recv_loop(self):
        """Background thread to receive messages."""
        while self._running and self.ws:
            try:
                data = self.ws.recv()
                if not data:
                    continue

                msg = json.loads(data)
                if self.verbose:
                    print(f"<< {json.dumps(msg, indent=2)}")

                msg_type = msg.get("type", "")
                req_id = msg.get("request_id", "")

                if msg_type == "connected":
                    self.connected = True
                elif msg_type == "auth_success":
                    self.authenticated = True

                if req_id:
                    with self.lock:
                        self.responses[req_id] = msg
                else:
                    # Print unsolicited messages
                    self._print_message(msg)

            except websocket.WebSocketConnectionClosedException:
                break
            except Exception as e:
                if self._running:
                    print(f"Receive error: {e}", file=sys.stderr)
                break

        self.connected = False

    def send(self, msg_type: str, data: Optional[Dict] = None) -> str:
        """Send a message and return request ID."""
        req_id = str(uuid.uuid4())[:8]
        msg = {
            "type": msg_type,
            "request_id": req_id,
        }
        if data:
            msg.update(data)

        if self.verbose:
            print(f">> {json.dumps(msg, indent=2)}")

        self.ws.send(json.dumps(msg))
        return req_id

    def wait_response(self, req_id: str, timeout: float = 5.0) -> Optional[Dict]:
        """Wait for a response with matching request ID."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if req_id in self.responses:
                    return self.responses.pop(req_id)
            time.sleep(0.05)
        return None

    def auth(self, api_key: str, api_secret: str) -> bool:
        """Authenticate with API credentials."""
        req_id = self.send("auth", {
            "apiKey": api_key,
            "apiSecret": api_secret
        })
        resp = self.wait_response(req_id)
        if resp and resp.get("type") == "auth_success":
            self.authenticated = True
            return True
        return False

    def place_order(self, symbol: str, side: str, order_type: str,
                    price: float, size: float) -> Optional[Dict]:
        """Place a new order."""
        req_id = self.send("place_order", {
            "order": {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "price": price,
                "size": size
            }
        })
        return self.wait_response(req_id)

    def cancel_order(self, order_id: int) -> Optional[Dict]:
        """Cancel an existing order."""
        req_id = self.send("cancel_order", {"orderID": order_id})
        return self.wait_response(req_id)

    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Subscribe to orderbook updates."""
        req_id = self.send("subscribe", {"symbols": [symbol]})
        return self.wait_response(req_id, timeout=3.0)

    def get_positions(self) -> Optional[Dict]:
        """Get all positions."""
        req_id = self.send("get_positions", {})
        return self.wait_response(req_id)

    def get_orders(self) -> Optional[Dict]:
        """Get all open orders."""
        req_id = self.send("get_orders", {})
        return self.wait_response(req_id)

    def close(self):
        """Close the connection."""
        self._running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass

    def _print_message(self, msg: Dict):
        """Print a message in human-readable format."""
        msg_type = msg.get("type", "unknown")
        data = msg.get("data", {})
        error = msg.get("error", "")

        if error:
            print(f"Error: {error}")
            return

        if msg_type == "orderbook":
            symbol = data.get("symbol", "")
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            print(f"OrderBook [{symbol}]:")
            print(f"  Bids: {len(bids)} levels")
            for bid in bids[:5]:
                print(f"    {bid.get('price', 0):.2f} @ {bid.get('size', 0):.4f}")
            print(f"  Asks: {len(asks)} levels")
            for ask in asks[:5]:
                print(f"    {ask.get('price', 0):.2f} @ {ask.get('size', 0):.4f}")
        elif msg_type in ("order_update", "position_update"):
            print(f"{msg_type}: {json.dumps(data, indent=2)}")
        elif msg_type != "connected":
            print(json.dumps(msg, indent=2))


def print_help():
    """Print interactive help."""
    print("""
LX DEX CLI Commands:

  place_order <symbol> <side> <type> <price> <size>
    Example: place_order BTC-USD buy limit 50000 0.1

  cancel_order <order_id>
    Example: cancel_order 12345

  get_orderbook <symbol>
    Example: get_orderbook BTC-USD

  get_positions
    Show all open positions

  get_orders
    Show all open orders

  auth <api_key> <api_secret>
    Authenticate with credentials

  help
    Show this help message

  quit / exit
    Exit the CLI
""")


def run_interactive(client: LXClient):
    """Run interactive mode."""
    print("LX DEX CLI - Type 'help' for commands")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "help":
            print_help()

        elif cmd in ("quit", "exit"):
            print("Goodbye")
            break

        elif cmd == "auth":
            if len(parts) < 3:
                print("Usage: auth <api_key> <api_secret>")
            else:
                if client.auth(parts[1], parts[2]):
                    print("Authenticated successfully")
                else:
                    print("Authentication failed")

        elif cmd == "place_order":
            if len(parts) < 6:
                print("Usage: place_order <symbol> <side> <type> <price> <size>")
            else:
                try:
                    price = float(parts[4])
                    size = float(parts[5])
                    resp = client.place_order(parts[1], parts[2], parts[3], price, size)
                    if resp:
                        print(json.dumps(resp, indent=2))
                    else:
                        print("No response received")
                except ValueError:
                    print("Invalid price or size")

        elif cmd == "cancel_order":
            if len(parts) < 2:
                print("Usage: cancel_order <order_id>")
            else:
                try:
                    order_id = int(parts[1])
                    resp = client.cancel_order(order_id)
                    if resp:
                        print(json.dumps(resp, indent=2))
                    else:
                        print("No response received")
                except ValueError:
                    print("Invalid order ID")

        elif cmd == "get_orderbook":
            if len(parts) < 2:
                print("Usage: get_orderbook <symbol>")
            else:
                resp = client.get_orderbook(parts[1])
                if resp:
                    print(json.dumps(resp, indent=2))
                print(f"Subscribed to {parts[1]} orderbook")

        elif cmd == "get_positions":
            resp = client.get_positions()
            if resp:
                print(json.dumps(resp, indent=2))
            else:
                print("No response received")

        elif cmd == "get_orders":
            resp = client.get_orders()
            if resp:
                print(json.dumps(resp, indent=2))
            else:
                print("No response received")

        else:
            print(f"Unknown command: {cmd}. Type 'help' for commands.")


def run_command(client: LXClient, args: argparse.Namespace):
    """Run a single command."""
    cmd = args.command

    if cmd == "place_order":
        resp = client.place_order(
            args.symbol, args.side, args.type,
            args.price, args.size
        )
        if resp:
            print(json.dumps(resp, indent=2))
        else:
            print("No response received", file=sys.stderr)
            sys.exit(1)

    elif cmd == "cancel_order":
        resp = client.cancel_order(args.order_id)
        if resp:
            print(json.dumps(resp, indent=2))
        else:
            print("No response received", file=sys.stderr)
            sys.exit(1)

    elif cmd == "get_orderbook":
        resp = client.get_orderbook(args.symbol)
        if resp:
            print(json.dumps(resp, indent=2))
        # Wait a bit for orderbook data
        time.sleep(1)

    elif cmd == "get_positions":
        resp = client.get_positions()
        if resp:
            print(json.dumps(resp, indent=2))
        else:
            print("No response received", file=sys.stderr)
            sys.exit(1)

    elif cmd == "get_orders":
        resp = client.get_orders()
        if resp:
            print(json.dumps(resp, indent=2))
        else:
            print("No response received", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="LX DEX CLI Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i                                    # Interactive mode
  %(prog)s place_order BTC-USD buy limit 50000 0.1
  %(prog)s cancel_order 12345
  %(prog)s get_orderbook BTC-USD
  %(prog)s get_positions
  %(prog)s get_orders
"""
    )

    parser.add_argument(
        "-u", "--url",
        default="ws://localhost:8081",
        help="WebSocket server URL (default: ws://localhost:8081)"
    )
    parser.add_argument(
        "-k", "--key",
        help="API key for authentication"
    )
    parser.add_argument(
        "-s", "--secret",
        help="API secret for authentication"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # place_order
    p_order = subparsers.add_parser("place_order", help="Place a new order")
    p_order.add_argument("symbol", help="Trading pair symbol (e.g., BTC-USD)")
    p_order.add_argument("side", choices=["buy", "sell"], help="Order side")
    p_order.add_argument("type", choices=["limit", "market", "stop", "stop_limit"],
                         help="Order type")
    p_order.add_argument("price", type=float, help="Order price")
    p_order.add_argument("size", type=float, help="Order size")

    # cancel_order
    p_cancel = subparsers.add_parser("cancel_order", help="Cancel an order")
    p_cancel.add_argument("order_id", type=int, help="Order ID to cancel")

    # get_orderbook
    p_book = subparsers.add_parser("get_orderbook", help="Get orderbook")
    p_book.add_argument("symbol", help="Trading pair symbol")

    # get_positions
    subparsers.add_parser("get_positions", help="Get all positions")

    # get_orders
    subparsers.add_parser("get_orders", help="Get all open orders")

    args = parser.parse_args()

    # Create client and connect
    client = LXClient(args.url, args.verbose)
    if not client.connect():
        print("Failed to connect to server", file=sys.stderr)
        sys.exit(1)

    try:
        # Authenticate if credentials provided
        if args.key and args.secret:
            if not client.auth(args.key, args.secret):
                print("Authentication failed", file=sys.stderr)
                sys.exit(1)
            if args.verbose:
                print("Authenticated")

        # Run interactive or command mode
        if args.interactive or not args.command:
            run_interactive(client)
        else:
            run_command(client, args)

    finally:
        client.close()


if __name__ == "__main__":
    main()
