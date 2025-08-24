"""
Market data, liquidation, and settlement features for LX DEX Python Client
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .exceptions import LXDexException


@dataclass
class MarketDataSource:
    """Market data from a specific provider"""
    name: str
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    latency_ns: int
    provider: str


@dataclass
class LiquidationInfo:
    """Liquidation information"""
    user_id: str
    position_id: str
    symbol: str
    size: float
    liquidation_price: float
    mark_price: float
    status: str
    timestamp: datetime


@dataclass
class SettlementBatch:
    """Settlement batch information"""
    batch_id: int
    order_ids: List[int]
    status: str
    tx_hash: Optional[str]
    gas_used: Optional[int]
    timestamp: datetime


@dataclass
class MarginInfo:
    """Margin information for a user"""
    user_id: str
    initial_margin: float
    maintenance_margin: float
    margin_ratio: float
    free_margin: float
    margin_level: float


class MarketDataClient:
    """Market data and advanced features client"""
    
    def __init__(self, json_rpc_client):
        """
        Initialize market data client
        
        Args:
            json_rpc_client: JSON-RPC client instance
        """
        self.rpc = json_rpc_client
    
    def get_market_data(self, symbol: str, source: str) -> MarketDataSource:
        """
        Get market data from a specific source
        
        Args:
            symbol: Trading pair symbol
            source: Data source (e.g., 'alpaca', 'polygon', 'iex')
            
        Returns:
            MarketDataSource object
        """
        result = self.rpc.call("market_data.get", {
            "symbol": symbol,
            "source": source
        })
        
        return MarketDataSource(
            name=result["name"],
            symbol=result["symbol"],
            price=result["price"],
            bid=result["bid"],
            ask=result["ask"],
            volume=result["volume"],
            latency_ns=result["latency_ns"],
            provider=result["provider"]
        )
    
    def get_aggregated_market_data(self, symbol: str) -> List[MarketDataSource]:
        """
        Get aggregated market data from all sources
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of MarketDataSource objects
        """
        result = self.rpc.call("market_data.aggregate", {
            "symbol": symbol
        })
        
        return [MarketDataSource(**data) for data in result]
    
    def get_liquidations(self, symbol: str, limit: int = 100) -> List[LiquidationInfo]:
        """
        Get recent liquidations
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of liquidations to return
            
        Returns:
            List of LiquidationInfo objects
        """
        result = self.rpc.call("liquidations.get", {
            "symbol": symbol,
            "limit": limit
        })
        
        liquidations = []
        for liq in result:
            liquidations.append(LiquidationInfo(
                user_id=liq["user_id"],
                position_id=liq["position_id"],
                symbol=liq["symbol"],
                size=liq["size"],
                liquidation_price=liq["liquidation_price"],
                mark_price=liq["mark_price"],
                status=liq["status"],
                timestamp=datetime.fromisoformat(liq["timestamp"])
            ))
        
        return liquidations
    
    def get_settlement_batch(self, batch_id: int) -> SettlementBatch:
        """
        Get settlement batch information
        
        Args:
            batch_id: Settlement batch ID
            
        Returns:
            SettlementBatch object
        """
        result = self.rpc.call("settlement.batch", {
            "batch_id": batch_id
        })
        
        return SettlementBatch(
            batch_id=result["batch_id"],
            order_ids=result["order_ids"],
            status=result["status"],
            tx_hash=result.get("tx_hash"),
            gas_used=result.get("gas_used"),
            timestamp=datetime.fromisoformat(result["timestamp"])
        )
    
    def get_margin_info(self, user_id: str) -> MarginInfo:
        """
        Get margin information for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            MarginInfo object
        """
        result = self.rpc.call("margin.info", {
            "user_id": user_id
        })
        
        return MarginInfo(
            user_id=result["user_id"],
            initial_margin=result["initial_margin"],
            maintenance_margin=result["maintenance_margin"],
            margin_ratio=result["margin_ratio"],
            free_margin=result["free_margin"],
            margin_level=result["margin_level"]
        )
    
    def check_liquidation_risk(self, user_id: str) -> Dict[str, Any]:
        """
        Check liquidation risk for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with risk metrics
        """
        result = self.rpc.call("margin.liquidation_risk", {
            "user_id": user_id
        })
        
        return result
    
    def get_insurance_fund_status(self) -> Dict[str, Any]:
        """
        Get insurance fund status
        
        Returns:
            Dictionary with fund status
        """
        result = self.rpc.call("insurance_fund.status")
        return result
    
    def get_market_data_sources(self) -> List[str]:
        """
        Get list of available market data sources
        
        Returns:
            List of source names
        """
        result = self.rpc.call("market_data.sources")
        return result
    
    def get_market_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive market statistics
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with market statistics
        """
        result = self.rpc.call("market.stats", {
            "symbol": symbol
        })
        
        return result


class LiquidationMonitor:
    """Monitor and manage liquidations"""
    
    def __init__(self, ws_client):
        """
        Initialize liquidation monitor
        
        Args:
            ws_client: WebSocket client for real-time updates
        """
        self.ws = ws_client
        self.callbacks = {}
    
    def subscribe_liquidations(self, callback) -> None:
        """
        Subscribe to liquidation events
        
        Args:
            callback: Function to call on liquidation events
        """
        self.callbacks["liquidations"] = callback
        
        if self.ws:
            self.ws.send({
                "type": "subscribe",
                "channel": "liquidations"
            })
    
    def subscribe_settlements(self, callback) -> None:
        """
        Subscribe to settlement events
        
        Args:
            callback: Function to call on settlement events
        """
        self.callbacks["settlements"] = callback
        
        if self.ws:
            self.ws.send({
                "type": "subscribe",
                "channel": "settlements"
            })
    
    def subscribe_margin_calls(self, user_id: str, callback) -> None:
        """
        Subscribe to margin call events for a user
        
        Args:
            user_id: User identifier
            callback: Function to call on margin call events
        """
        self.callbacks[f"margin_calls:{user_id}"] = callback
        
        if self.ws:
            self.ws.send({
                "type": "subscribe",
                "channel": f"margin_calls:{user_id}"
            })
    
    def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from a channel
        
        Args:
            channel: Channel name
        """
        if channel in self.callbacks:
            del self.callbacks[channel]
        
        if self.ws:
            self.ws.send({
                "type": "unsubscribe",
                "channel": channel
            })