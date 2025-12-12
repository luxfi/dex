//! LX DEX CLI Client
//!
//! Command-line trading interface for LX DEX WebSocket API.

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tungstenite::{connect, Message as WsMessage};
use url::Url;

/// Message from/to the WebSocket server
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamp: Option<i64>,
}

/// Order for placement
#[derive(Debug, Clone, Serialize)]
struct Order {
    symbol: String,
    side: String,
    #[serde(rename = "type")]
    order_type: String,
    price: f64,
    size: f64,
}

/// WebSocket client for LX DEX
struct Client {
    sender: Sender<String>,
    responses: Arc<Mutex<HashMap<String, Message>>>,
    receiver: Receiver<Message>,
    verbose: bool,
    req_counter: Arc<Mutex<u64>>,
}

impl Client {
    /// Create a new client and connect to the server
    fn new(url: &str, verbose: bool) -> Result<Self> {
        let url = Url::parse(url).context("Invalid WebSocket URL")?;
        let (mut socket, _response) = connect(url).context("Failed to connect")?;

        let (tx_send, rx_send) = channel::<String>();
        let (tx_recv, rx_recv) = channel::<Message>();
        let responses: Arc<Mutex<HashMap<String, Message>>> = Arc::new(Mutex::new(HashMap::new()));
        let responses_clone = Arc::clone(&responses);

        // Read thread
        let verbose_clone = verbose;
        thread::spawn(move || {
            loop {
                match socket.read() {
                    Ok(WsMessage::Text(text)) => {
                        if verbose_clone {
                            eprintln!("<< {}", text);
                        }
                        if let Ok(msg) = serde_json::from_str::<Message>(&text) {
                            if let Some(ref req_id) = msg.request_id {
                                let mut resp = responses_clone.lock().unwrap();
                                resp.insert(req_id.clone(), msg.clone());
                            }
                            let _ = tx_recv.send(msg);
                        }
                    }
                    Ok(WsMessage::Close(_)) => break,
                    Err(e) => {
                        eprintln!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }

                // Check for outgoing messages
                if let Ok(msg) = rx_send.try_recv() {
                    if verbose_clone {
                        eprintln!(">> {}", msg);
                    }
                    if socket.send(WsMessage::Text(msg)).is_err() {
                        break;
                    }
                }
            }
        });

        Ok(Client {
            sender: tx_send,
            responses,
            receiver: rx_recv,
            verbose,
            req_counter: Arc::new(Mutex::new(0)),
        })
    }

    /// Generate a unique request ID
    fn next_req_id(&self) -> String {
        let mut counter = self.req_counter.lock().unwrap();
        *counter += 1;
        format!("req-{}", *counter)
    }

    /// Send a message to the server
    fn send(&self, msg_type: &str, data: Option<Value>) -> Result<String> {
        let req_id = self.next_req_id();
        let mut msg = json!({
            "type": msg_type,
            "request_id": req_id,
        });
        if let Some(d) = data {
            if let Value::Object(map) = d {
                for (k, v) in map {
                    msg[k] = v;
                }
            }
        }
        self.sender
            .send(serde_json::to_string(&msg)?)
            .context("Failed to send message")?;
        Ok(req_id)
    }

    /// Wait for a response with matching request ID
    fn wait_response(&self, req_id: &str, timeout: Duration) -> Result<Message> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            // Check stored responses
            {
                let mut resp = self.responses.lock().unwrap();
                if let Some(msg) = resp.remove(req_id) {
                    return Ok(msg);
                }
            }

            // Check for new messages
            if let Ok(msg) = self.receiver.recv_timeout(Duration::from_millis(50)) {
                if msg.request_id.as_deref() == Some(req_id) {
                    return Ok(msg);
                }
                // Print other messages
                if msg.msg_type != "connected" {
                    print_message(&msg);
                }
            }

            if std::time::Instant::now() > deadline {
                return Err(anyhow!("Timeout waiting for response"));
            }
        }
    }

    /// Wait for connected message
    fn wait_connected(&self) -> Result<()> {
        let timeout = Duration::from_secs(5);
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Ok(msg) = self.receiver.recv_timeout(Duration::from_millis(100)) {
                if msg.msg_type == "connected" {
                    return Ok(());
                }
            }
            if std::time::Instant::now() > deadline {
                return Err(anyhow!("Timeout waiting for connection"));
            }
        }
    }

    /// Authenticate with API credentials
    fn auth(&self, api_key: &str, api_secret: &str) -> Result<()> {
        let req_id = self.send(
            "auth",
            Some(json!({
                "apiKey": api_key,
                "apiSecret": api_secret
            })),
        )?;
        let resp = self.wait_response(&req_id, Duration::from_secs(5))?;
        if resp.error.is_some() {
            return Err(anyhow!("Auth failed: {:?}", resp.error));
        }
        Ok(())
    }

    /// Place an order
    fn place_order(&self, order: &Order) -> Result<Message> {
        let req_id = self.send("place_order", Some(json!({ "order": order })))?;
        self.wait_response(&req_id, Duration::from_secs(5))
    }

    /// Cancel an order
    fn cancel_order(&self, order_id: u64) -> Result<Message> {
        let req_id = self.send("cancel_order", Some(json!({ "orderID": order_id })))?;
        self.wait_response(&req_id, Duration::from_secs(5))
    }

    /// Get positions
    fn get_positions(&self) -> Result<Message> {
        let req_id = self.send("get_positions", None)?;
        self.wait_response(&req_id, Duration::from_secs(5))
    }

    /// Get orders
    fn get_orders(&self) -> Result<Message> {
        let req_id = self.send("get_orders", None)?;
        self.wait_response(&req_id, Duration::from_secs(5))
    }

    /// Subscribe to orderbook
    fn subscribe(&self, symbol: &str) -> Result<()> {
        self.send("subscribe", Some(json!({ "symbols": [symbol] })))?;
        Ok(())
    }
}

fn print_message(msg: &Message) {
    if let Some(ref error) = msg.error {
        eprintln!("Error: {}", error);
        return;
    }

    match msg.msg_type.as_str() {
        "orderbook" => {
            if let Some(ref data) = msg.data {
                let symbol = data.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
                println!("OrderBook [{}]:", symbol);
                if let Some(bids) = data.get("bids").and_then(|v| v.as_array()) {
                    println!("  Bids: {} levels", bids.len());
                    for bid in bids.iter().take(5) {
                        let price = bid.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let size = bid.get("size").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        println!("    {:.2} @ {:.4}", price, size);
                    }
                }
                if let Some(asks) = data.get("asks").and_then(|v| v.as_array()) {
                    println!("  Asks: {} levels", asks.len());
                    for ask in asks.iter().take(5) {
                        let price = ask.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let size = ask.get("size").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        println!("    {:.2} @ {:.4}", price, size);
                    }
                }
            }
        }
        "order_update" | "position_update" => {
            println!(
                "{}: {}",
                msg.msg_type,
                serde_json::to_string_pretty(&msg.data).unwrap_or_default()
            );
        }
        _ => {
            println!(
                "{}",
                serde_json::to_string_pretty(msg).unwrap_or_default()
            );
        }
    }
}

fn print_help() {
    println!(
        r#"
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
"#
    );
}

fn run_interactive(client: &Client) {
    println!("LX DEX CLI - Type 'help' for commands");
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        let cmd = parts[0].to_lowercase();

        match cmd.as_str() {
            "help" => print_help(),
            "quit" | "exit" => {
                println!("Goodbye");
                break;
            }
            "auth" => {
                if parts.len() < 3 {
                    println!("Usage: auth <api_key> <api_secret>");
                } else {
                    match client.auth(parts[1], parts[2]) {
                        Ok(_) => println!("Authenticated successfully"),
                        Err(e) => eprintln!("Auth failed: {}", e),
                    }
                }
            }
            "place_order" => {
                if parts.len() < 6 {
                    println!("Usage: place_order <symbol> <side> <type> <price> <size>");
                } else {
                    let price: Result<f64, _> = parts[4].parse();
                    let size: Result<f64, _> = parts[5].parse();
                    match (price, size) {
                        (Ok(price), Ok(size)) => {
                            let order = Order {
                                symbol: parts[1].to_string(),
                                side: parts[2].to_string(),
                                order_type: parts[3].to_string(),
                                price,
                                size,
                            };
                            match client.place_order(&order) {
                                Ok(resp) => print_message(&resp),
                                Err(e) => eprintln!("Failed: {}", e),
                            }
                        }
                        _ => println!("Invalid price or size"),
                    }
                }
            }
            "cancel_order" => {
                if parts.len() < 2 {
                    println!("Usage: cancel_order <order_id>");
                } else {
                    match parts[1].parse::<u64>() {
                        Ok(order_id) => match client.cancel_order(order_id) {
                            Ok(resp) => print_message(&resp),
                            Err(e) => eprintln!("Failed: {}", e),
                        },
                        Err(_) => println!("Invalid order ID"),
                    }
                }
            }
            "get_orderbook" => {
                if parts.len() < 2 {
                    println!("Usage: get_orderbook <symbol>");
                } else {
                    match client.subscribe(parts[1]) {
                        Ok(_) => println!("Subscribed to {} orderbook", parts[1]),
                        Err(e) => eprintln!("Failed: {}", e),
                    }
                }
            }
            "get_positions" => match client.get_positions() {
                Ok(resp) => print_message(&resp),
                Err(e) => eprintln!("Failed: {}", e),
            },
            "get_orders" => match client.get_orders() {
                Ok(resp) => print_message(&resp),
                Err(e) => eprintln!("Failed: {}", e),
            },
            "subscribe" => {
                if parts.len() < 2 {
                    println!("Usage: subscribe <symbol>");
                } else {
                    match client.subscribe(parts[1]) {
                        Ok(_) => println!("Subscribed to {}", parts[1]),
                        Err(e) => eprintln!("Failed: {}", e),
                    }
                }
            }
            _ => println!("Unknown command: {}. Type 'help' for commands.", cmd),
        }
    }
}

#[derive(Parser)]
#[command(name = "lx-cli")]
#[command(about = "LX DEX CLI Client", long_about = None)]
struct Cli {
    /// WebSocket server URL
    #[arg(short, long, default_value = "ws://localhost:8081")]
    url: String,

    /// API key for authentication
    #[arg(short, long)]
    key: Option<String>,

    /// API secret for authentication
    #[arg(short, long)]
    secret: Option<String>,

    /// Interactive mode
    #[arg(short, long)]
    interactive: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Place a new order
    PlaceOrder {
        /// Trading pair symbol (e.g., BTC-USD)
        symbol: String,
        /// Order side (buy/sell)
        side: String,
        /// Order type (limit/market/stop/stop_limit)
        #[arg(name = "type")]
        order_type: String,
        /// Order price
        price: f64,
        /// Order size
        size: f64,
    },
    /// Cancel an order
    CancelOrder {
        /// Order ID to cancel
        order_id: u64,
    },
    /// Get orderbook for a symbol
    GetOrderbook {
        /// Trading pair symbol
        symbol: String,
    },
    /// Get all positions
    GetPositions,
    /// Get all open orders
    GetOrders,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Connect
    let client = Client::new(&cli.url, cli.verbose)?;
    client.wait_connected()?;

    if cli.verbose {
        eprintln!("Connected to LX DEX");
    }

    // Authenticate if credentials provided
    if let (Some(ref key), Some(ref secret)) = (&cli.key, &cli.secret) {
        client.auth(key, secret)?;
        if cli.verbose {
            eprintln!("Authenticated");
        }
    }

    // Run command or interactive mode
    if cli.interactive || cli.command.is_none() {
        run_interactive(&client);
    } else if let Some(cmd) = cli.command {
        match cmd {
            Commands::PlaceOrder {
                symbol,
                side,
                order_type,
                price,
                size,
            } => {
                let order = Order {
                    symbol,
                    side,
                    order_type,
                    price,
                    size,
                };
                let resp = client.place_order(&order)?;
                println!("{}", serde_json::to_string_pretty(&resp)?);
            }
            Commands::CancelOrder { order_id } => {
                let resp = client.cancel_order(order_id)?;
                println!("{}", serde_json::to_string_pretty(&resp)?);
            }
            Commands::GetOrderbook { symbol } => {
                client.subscribe(&symbol)?;
                // Wait a bit for orderbook data
                thread::sleep(Duration::from_secs(1));
            }
            Commands::GetPositions => {
                let resp = client.get_positions()?;
                println!("{}", serde_json::to_string_pretty(&resp)?);
            }
            Commands::GetOrders => {
                let resp = client.get_orders()?;
                println!("{}", serde_json::to_string_pretty(&resp)?);
            }
        }
    }

    Ok(())
}
