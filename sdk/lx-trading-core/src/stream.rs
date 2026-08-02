//! Async streams for order updates, fills, and market data.
//!
//! Provides unified streaming interface across all venues.

use futures::Stream;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{broadcast, mpsc};

use crate::types::*;
use crate::ws::{Fill, OrderUpdate, OrderbookUpdate, WsEvent};

/// Stream of order updates across all venues
pub struct OrderStream {
    rx: mpsc::Receiver<OrderUpdate>,
}

impl OrderStream {
    /// Create a new order stream from multiple venue event receivers
    pub fn new(mut event_rxs: Vec<broadcast::Receiver<WsEvent>>) -> Self {
        let (tx, rx) = mpsc::channel(1024);

        for mut event_rx in event_rxs.drain(..) {
            let tx = tx.clone();
            tokio::spawn(async move {
                while let Ok(event) = event_rx.recv().await {
                    if let WsEvent::OrderUpdate(update) = event {
                        if tx.send(update).await.is_err() {
                            break;
                        }
                    }
                }
            });
        }

        Self { rx }
    }
}

impl Stream for OrderStream {
    type Item = OrderUpdate;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_recv(cx)
    }
}

/// Stream of fills across all venues
pub struct FillStream {
    rx: mpsc::Receiver<Fill>,
}

impl FillStream {
    /// Create a new fill stream from multiple venue event receivers
    pub fn new(mut event_rxs: Vec<broadcast::Receiver<WsEvent>>) -> Self {
        let (tx, rx) = mpsc::channel(1024);

        for mut event_rx in event_rxs.drain(..) {
            let tx = tx.clone();
            tokio::spawn(async move {
                while let Ok(event) = event_rx.recv().await {
                    if let WsEvent::Fill(fill) = event {
                        if tx.send(fill).await.is_err() {
                            break;
                        }
                    }
                }
            });
        }

        Self { rx }
    }
}

impl Stream for FillStream {
    type Item = Fill;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_recv(cx)
    }
}

/// Stream of orderbook updates for a specific symbol across all venues
pub struct OrderbookStream {
    rx: mpsc::Receiver<OrderbookUpdate>,
    symbol: String,
}

impl OrderbookStream {
    /// Create a new orderbook stream for a specific symbol
    pub fn new(
        symbol: impl Into<String>,
        mut event_rxs: Vec<broadcast::Receiver<WsEvent>>,
    ) -> Self {
        let symbol = symbol.into();
        let (tx, rx) = mpsc::channel(1024);

        let symbol_filter = symbol.clone();
        for mut event_rx in event_rxs.drain(..) {
            let tx = tx.clone();
            let symbol_filter = symbol_filter.clone();
            tokio::spawn(async move {
                while let Ok(event) = event_rx.recv().await {
                    if let WsEvent::OrderbookUpdate(update) = event {
                        if update.symbol == symbol_filter && tx.send(update).await.is_err() {
                            break;
                        }
                    }
                }
            });
        }

        Self { rx, symbol }
    }

    /// Get the symbol this stream is for
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

impl Stream for OrderbookStream {
    type Item = OrderbookUpdate;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_recv(cx)
    }
}

/// Stream of trades for a specific symbol across all venues
pub struct TradeStream {
    rx: mpsc::Receiver<Trade>,
    symbol: String,
}

impl TradeStream {
    /// Create a new trade stream for a specific symbol
    pub fn new(
        symbol: impl Into<String>,
        mut event_rxs: Vec<broadcast::Receiver<WsEvent>>,
    ) -> Self {
        let symbol = symbol.into();
        let (tx, rx) = mpsc::channel(1024);

        let symbol_filter = symbol.clone();
        for mut event_rx in event_rxs.drain(..) {
            let tx = tx.clone();
            let symbol_filter = symbol_filter.clone();
            tokio::spawn(async move {
                while let Ok(event) = event_rx.recv().await {
                    if let WsEvent::Trade(trade) = event {
                        if trade.symbol == symbol_filter && tx.send(trade).await.is_err() {
                            break;
                        }
                    }
                }
            });
        }

        Self { rx, symbol }
    }

    /// Get the symbol this stream is for
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

impl Stream for TradeStream {
    type Item = Trade;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_recv(cx)
    }
}

/// Stream of tickers for a specific symbol across all venues
pub struct TickerStream {
    rx: mpsc::Receiver<Ticker>,
    symbol: String,
}

impl TickerStream {
    /// Create a new ticker stream for a specific symbol
    pub fn new(
        symbol: impl Into<String>,
        mut event_rxs: Vec<broadcast::Receiver<WsEvent>>,
    ) -> Self {
        let symbol = symbol.into();
        let (tx, rx) = mpsc::channel(1024);

        let symbol_filter = symbol.clone();
        for mut event_rx in event_rxs.drain(..) {
            let tx = tx.clone();
            let symbol_filter = symbol_filter.clone();
            tokio::spawn(async move {
                while let Ok(event) = event_rx.recv().await {
                    if let WsEvent::Ticker(ticker) = event {
                        if ticker.symbol == symbol_filter && tx.send(ticker).await.is_err() {
                            break;
                        }
                    }
                }
            });
        }

        Self { rx, symbol }
    }

    /// Get the symbol this stream is for
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

impl Stream for TickerStream {
    type Item = Ticker;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_recv(cx)
    }
}

/// Aggregated stream that maintains best bid/ask across all venues
pub struct AggregatedTickerStream {
    rx: mpsc::Receiver<AggregatedTicker>,
    symbol: String,
}

/// Aggregated ticker with best prices across venues
#[derive(Debug, Clone)]
pub struct AggregatedTicker {
    pub symbol: String,
    pub best_bid: Option<(rust_decimal::Decimal, String)>, // (price, venue)
    pub best_ask: Option<(rust_decimal::Decimal, String)>,
    pub mid_price: Option<rust_decimal::Decimal>,
    pub spread: Option<rust_decimal::Decimal>,
    pub by_venue: HashMap<String, Ticker>,
    pub timestamp: i64,
}

impl AggregatedTickerStream {
    /// Create a new aggregated ticker stream
    pub fn new(
        symbol: impl Into<String>,
        mut event_rxs: Vec<broadcast::Receiver<WsEvent>>,
    ) -> Self {
        let symbol = symbol.into();
        let (tx, rx) = mpsc::channel(1024);

        let symbol_filter = symbol.clone();
        let tickers: Arc<RwLock<HashMap<String, Ticker>>> = Arc::new(RwLock::new(HashMap::new()));

        for mut event_rx in event_rxs.drain(..) {
            let tx = tx.clone();
            let symbol_filter = symbol_filter.clone();
            let tickers = tickers.clone();

            tokio::spawn(async move {
                while let Ok(event) = event_rx.recv().await {
                    if let WsEvent::Ticker(ticker) = event {
                        if ticker.symbol == symbol_filter {
                            // Update venue ticker
                            {
                                let mut t = tickers.write();
                                t.insert(ticker.venue.clone(), ticker);
                            }

                            // Calculate aggregated ticker
                            let agg = {
                                let t = tickers.read();
                                calculate_aggregated_ticker(&symbol_filter, &t)
                            };

                            if tx.send(agg).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            });
        }

        Self { rx, symbol }
    }

    /// Get the symbol this stream is for
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

impl Stream for AggregatedTickerStream {
    type Item = AggregatedTicker;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_recv(cx)
    }
}

fn calculate_aggregated_ticker(
    symbol: &str,
    tickers: &HashMap<String, Ticker>,
) -> AggregatedTicker {
    use rust_decimal::Decimal;

    let mut best_bid: Option<(Decimal, String)> = None;
    let mut best_ask: Option<(Decimal, String)> = None;

    for (venue, ticker) in tickers {
        if let Some(bid) = ticker.bid {
            if best_bid.is_none() || bid > best_bid.as_ref().unwrap().0 {
                best_bid = Some((bid, venue.clone()));
            }
        }
        if let Some(ask) = ticker.ask {
            if best_ask.is_none() || ask < best_ask.as_ref().unwrap().0 {
                best_ask = Some((ask, venue.clone()));
            }
        }
    }

    let mid_price = match (&best_bid, &best_ask) {
        (Some((bid, _)), Some((ask, _))) => Some((*bid + *ask) / Decimal::from(2)),
        _ => None,
    };

    let spread = match (&best_bid, &best_ask) {
        (Some((bid, _)), Some((ask, _))) => Some(*ask - *bid),
        _ => None,
    };

    AggregatedTicker {
        symbol: symbol.to_string(),
        best_bid,
        best_ask,
        mid_price,
        spread,
        by_venue: tickers.clone(),
        timestamp: chrono::Utc::now().timestamp_millis(),
    }
}

/// Builder for creating multiple streams from venue connections
pub struct StreamBuilder {
    event_receivers: Vec<broadcast::Receiver<WsEvent>>,
}

impl StreamBuilder {
    /// Create a new stream builder
    pub fn new() -> Self {
        Self {
            event_receivers: Vec::new(),
        }
    }

    /// Add an event receiver from a WebSocket connection
    pub fn add_receiver(mut self, rx: broadcast::Receiver<WsEvent>) -> Self {
        self.event_receivers.push(rx);
        self
    }

    /// Build an order stream
    pub fn order_stream(self) -> OrderStream {
        OrderStream::new(self.event_receivers)
    }

    /// Build a fill stream
    pub fn fill_stream(self) -> FillStream {
        FillStream::new(self.event_receivers)
    }

    /// Build an orderbook stream for a symbol
    pub fn orderbook_stream(self, symbol: impl Into<String>) -> OrderbookStream {
        OrderbookStream::new(symbol, self.event_receivers)
    }

    /// Build a trade stream for a symbol
    pub fn trade_stream(self, symbol: impl Into<String>) -> TradeStream {
        TradeStream::new(symbol, self.event_receivers)
    }

    /// Build a ticker stream for a symbol
    pub fn ticker_stream(self, symbol: impl Into<String>) -> TickerStream {
        TickerStream::new(symbol, self.event_receivers)
    }

    /// Build an aggregated ticker stream for a symbol
    pub fn aggregated_ticker_stream(self, symbol: impl Into<String>) -> AggregatedTickerStream {
        AggregatedTickerStream::new(symbol, self.event_receivers)
    }
}

impl Default for StreamBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stream_builder() {
        let builder = StreamBuilder::new();
        // Just verify it compiles and can be created
        assert!(builder.event_receivers.is_empty());
    }

    #[test]
    fn test_aggregated_ticker_calculation() {
        use rust_decimal::Decimal;

        let mut tickers = HashMap::new();

        tickers.insert(
            "venue1".to_string(),
            Ticker {
                symbol: "BTC-USDC".to_string(),
                venue: "venue1".to_string(),
                bid: Some(Decimal::from(50000)),
                ask: Some(Decimal::from(50100)),
                last: None,
                volume_24h: None,
                high_24h: None,
                low_24h: None,
                change_24h: None,
                timestamp: 0,
            },
        );

        tickers.insert(
            "venue2".to_string(),
            Ticker {
                symbol: "BTC-USDC".to_string(),
                venue: "venue2".to_string(),
                bid: Some(Decimal::from(50050)),
                ask: Some(Decimal::from(50080)),
                last: None,
                volume_24h: None,
                high_24h: None,
                low_24h: None,
                change_24h: None,
                timestamp: 0,
            },
        );

        let agg = calculate_aggregated_ticker("BTC-USDC", &tickers);

        // Best bid should be from venue2 (50050 > 50000)
        assert_eq!(agg.best_bid.as_ref().unwrap().0, Decimal::from(50050));
        assert_eq!(agg.best_bid.as_ref().unwrap().1, "venue2");

        // Best ask should be from venue2 (50080 < 50100)
        assert_eq!(agg.best_ask.as_ref().unwrap().0, Decimal::from(50080));
        assert_eq!(agg.best_ask.as_ref().unwrap().1, "venue2");

        // Spread should be 50080 - 50050 = 30
        assert_eq!(agg.spread, Some(Decimal::from(30)));
    }
}
