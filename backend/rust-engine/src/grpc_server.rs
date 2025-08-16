use crate::engine::Engine;
use crate::types::*;
use rust_decimal::prelude::*;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

pub mod lx_engine {
    tonic::include_proto!("lx_engine");
}

use lx_engine::lx_engine_server::LxEngine;
use lx_engine::*;

pub struct LxEngineService {
    engine: Arc<Engine>,
}

impl LxEngineService {
    pub fn new(engine: Arc<Engine>) -> Self {
        LxEngineService { engine }
    }
}

#[tonic::async_trait]
impl LxEngine for LxEngineService {
    async fn submit_order(
        &self,
        request: Request<SubmitOrderRequest>,
    ) -> Result<Response<SubmitOrderResponse>, Status> {
        let req = request.into_inner();
        
        let side = match req.side.as_str() {
            "BUY" => Side::Buy,
            "SELL" => Side::Sell,
            _ => return Err(Status::invalid_argument("Invalid side")),
        };

        let order_type = match req.order_type.as_str() {
            "MARKET" => OrderType::Market,
            "LIMIT" => OrderType::Limit,
            _ => return Err(Status::invalid_argument("Invalid order type")),
        };

        let price = Decimal::from_f64(req.price)
            .ok_or_else(|| Status::invalid_argument("Invalid price"))?;
        let quantity = Decimal::from_f64(req.quantity)
            .ok_or_else(|| Status::invalid_argument("Invalid quantity"))?;

        let (order, trades) = self.engine.submit_order(
            req.user_id,
            req.symbol,
            side,
            order_type,
            price,
            quantity,
        );

        let response = SubmitOrderResponse {
            order_id: order.id.0,
            status: format!("{:?}", order.status),
            trades: trades
                .into_iter()
                .map(|t| Trade {
                    id: t.id,
                    symbol: t.symbol,
                    price: t.price.to_f64().unwrap_or(0.0),
                    quantity: t.quantity.to_f64().unwrap_or(0.0),
                    timestamp: t.timestamp,
                })
                .collect(),
        };

        Ok(Response::new(response))
    }

    async fn cancel_order(
        &self,
        request: Request<CancelOrderRequest>,
    ) -> Result<Response<CancelOrderResponse>, Status> {
        let req = request.into_inner();
        
        let order = self.engine.cancel_order(&req.symbol, req.order_id);
        
        let response = CancelOrderResponse {
            success: order.is_some(),
            message: if order.is_some() {
                "Order cancelled".to_string()
            } else {
                "Order not found".to_string()
            },
        };

        Ok(Response::new(response))
    }

    async fn get_orderbook(
        &self,
        request: Request<GetOrderbookRequest>,
    ) -> Result<Response<GetOrderbookResponse>, Status> {
        let req = request.into_inner();
        let depth = req.depth as usize;
        
        let snapshot = self.engine.get_orderbook_snapshot(&req.symbol, depth);
        
        let response = GetOrderbookResponse {
            symbol: snapshot.symbol,
            bids: snapshot
                .bids
                .into_iter()
                .map(|level| PriceLevel {
                    price: level.price.to_f64().unwrap_or(0.0),
                    quantity: level.quantity.to_f64().unwrap_or(0.0),
                    count: level.order_count as u32,
                })
                .collect(),
            asks: snapshot
                .asks
                .into_iter()
                .map(|level| PriceLevel {
                    price: level.price.to_f64().unwrap_or(0.0),
                    quantity: level.quantity.to_f64().unwrap_or(0.0),
                    count: level.order_count as u32,
                })
                .collect(),
            timestamp: snapshot.timestamp,
        };

        Ok(Response::new(response))
    }

    async fn stream_market_data(
        &self,
        request: Request<StreamMarketDataRequest>,
    ) -> Result<Response<Self::StreamMarketDataStream>, Status> {
        // This would typically implement a streaming response
        // For now, return unimplemented
        Err(Status::unimplemented("Streaming not yet implemented"))
    }

    async fn get_trades(
        &self,
        request: Request<GetTradesRequest>,
    ) -> Result<Response<GetTradesResponse>, Status> {
        // This would typically query historical trades
        // For now, return empty
        Ok(Response::new(GetTradesResponse {
            trades: Vec::new(),
        }))
    }
}