use lx_engine::{engine::Engine, grpc_server::LxEngineService};
use lx_engine::grpc_server::lx_engine::lx_engine_server::LxEngineServer;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::{info, error};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting LX Rust Engine");

    // Create engine
    let engine = Arc::new(Engine::new());
    
    // Create gRPC service
    let service = LxEngineService::new(engine);
    
    // Start gRPC server
    let addr = "0.0.0.0:50054".parse()?;
    info!("LX Rust Engine listening on {}", addr);
    
    Server::builder()
        .add_service(LxEngineServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}