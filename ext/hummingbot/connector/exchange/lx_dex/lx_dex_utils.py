"""
LX DEX Utilities

Configuration and utility functions for the LX DEX connector.
"""

from decimal import Decimal
from typing import Any, Dict, Optional

from pydantic import Field, SecretStr

from hummingbot.client.config.config_data_types import BaseConnectorConfigMap, ClientFieldData
from hummingbot.connector.exchange.lx_dex import lx_dex_constants as CONSTANTS


class LxDexConfigMap(BaseConnectorConfigMap):
    """Configuration for LX DEX connector."""

    connector: str = Field(default="lx_dex", const=True, client_data=None)

    lx_dex_api_key: SecretStr = Field(
        default=...,
        client_data=ClientFieldData(
            prompt=lambda cm: "Enter your LX DEX API key",
            is_secure=True,
            is_connect_key=True,
            prompt_on_new=True,
        ),
    )

    lx_dex_api_secret: SecretStr = Field(
        default=...,
        client_data=ClientFieldData(
            prompt=lambda cm: "Enter your LX DEX API secret",
            is_secure=True,
            is_connect_key=True,
            prompt_on_new=True,
        ),
    )

    lx_dex_wallet_address: str = Field(
        default=...,
        client_data=ClientFieldData(
            prompt=lambda cm: "Enter your wallet address",
            is_connect_key=True,
            prompt_on_new=True,
        ),
    )

    lx_dex_network: str = Field(
        default="mainnet",
        client_data=ClientFieldData(
            prompt=lambda cm: "Enter network (mainnet/testnet)",
            is_connect_key=False,
            prompt_on_new=False,
        ),
    )

    class Config:
        title = "lx_dex"


KEYS = LxDexConfigMap.construct()


def get_base_url(network: str = "mainnet") -> str:
    """Get the base URL for the specified network."""
    if network == "testnet":
        return CONSTANTS.TESTNET_BASE_URL
    return CONSTANTS.MAINNET_BASE_URL


def get_ws_url(network: str = "mainnet") -> str:
    """Get the WebSocket URL for the specified network."""
    if network == "testnet":
        return CONSTANTS.TESTNET_WS_URL
    return CONSTANTS.MAINNET_WS_URL


def convert_to_exchange_symbol(hb_symbol: str) -> str:
    """Convert Hummingbot trading pair format to exchange format.

    Example: BTC-USDT -> BTC/USDT
    """
    return hb_symbol.replace("-", "/")


def convert_from_exchange_symbol(exchange_symbol: str) -> str:
    """Convert exchange trading pair format to Hummingbot format.

    Example: BTC/USDT -> BTC-USDT
    """
    return exchange_symbol.replace("/", "-")


def decimal_to_str(value: Decimal, precision: int = 8) -> str:
    """Convert Decimal to string with specified precision."""
    return f"{value:.{precision}f}"


def str_to_decimal(value: str) -> Decimal:
    """Convert string to Decimal."""
    return Decimal(str(value))


def build_order_id(client_order_id: str) -> str:
    """Build exchange order ID from client order ID."""
    return f"hb-{client_order_id}"


def parse_order_id(exchange_order_id: str) -> str:
    """Parse client order ID from exchange order ID."""
    if exchange_order_id.startswith("hb-"):
        return exchange_order_id[3:]
    return exchange_order_id


def get_order_status(exchange_status: str) -> str:
    """Map exchange order status to Hummingbot status."""
    return CONSTANTS.ORDER_STATUS.get(exchange_status.lower(), "UNKNOWN")


def calculate_fee(
    is_maker: bool,
    base_currency: str,
    quote_currency: str,
    order_type: str,
    amount: Decimal,
    price: Decimal,
    fee_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate trading fee for an order."""
    fee_rate = (
        CONSTANTS.DEFAULT_MAKER_FEE if is_maker else CONSTANTS.DEFAULT_TAKER_FEE
    )

    if fee_override and "fee_rate" in fee_override:
        fee_rate = Decimal(str(fee_override["fee_rate"]))

    fee_amount = amount * price * Decimal(str(fee_rate))

    return {
        "percent": fee_rate * 100,
        "flat_fees": [{"asset": quote_currency, "amount": fee_amount}],
    }
