package client

import (
	"context"
	"fmt"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	
	"github.com/luxfi/dex/backend/pkg/engine"
	pb "github.com/luxfi/dex/backend/pkg/proto/engine"
)

// Mode specifies how the client connects to the engine
type Mode string

const (
	ModeLocal  Mode = "local"
	ModeRemote Mode = "remote"
	ModeHybrid Mode = "hybrid"
)

// ClientConfig configures the LX client
type ClientConfig struct {
	Mode          Mode
	ServerAddress string        // For remote mode
	EngineConfig  engine.Config // For local mode
	PreferLocal   bool          // For hybrid mode
}

// LXClient provides a unified interface to LX engines
type LXClient struct {
	mode   Mode
	local  *engine.LXEngine
	remote pb.EngineServiceClient
	conn   *grpc.ClientConn
	mu     sync.RWMutex
}

// NewLXClient creates a new LX client
func NewLXClient(config ClientConfig) (*LXClient, error) {
	client := &LXClient{
		mode: config.Mode,
	}

	switch config.Mode {
	case ModeLocal:
		// Create local embedded engine
		eng, err := engine.New(config.EngineConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create local engine: %w", err)
		}
		client.local = eng

	case ModeRemote:
		// Connect to remote engine
		conn, err := grpc.Dial(config.ServerAddress, 
			grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			return nil, fmt.Errorf("failed to connect to remote engine: %w", err)
		}
		client.conn = conn
		client.remote = pb.NewEngineServiceClient(conn)

	case ModeHybrid:
		// Create both local and remote
		if config.PreferLocal {
			eng, err := engine.New(config.EngineConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to create local engine: %w", err)
			}
			client.local = eng
		}
		
		// Always create remote as fallback
		conn, err := grpc.Dial(config.ServerAddress, 
			grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			// In hybrid mode, remote failure is not fatal if we have local
			if client.local == nil {
				return nil, fmt.Errorf("failed to connect to remote engine: %w", err)
			}
		} else {
			client.conn = conn
			client.remote = pb.NewEngineServiceClient(conn)
		}

	default:
		return nil, fmt.Errorf("invalid client mode: %s", config.Mode)
	}

	return client, nil
}

// SubmitOrder submits a new order
func (c *LXClient) SubmitOrder(ctx context.Context, req *pb.SubmitOrderRequest) (*pb.SubmitOrderResponse, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Use local engine if available
	if c.local != nil {
		return c.local.SubmitOrder(ctx, req)
	}

	// Otherwise use remote
	if c.remote != nil {
		return c.remote.SubmitOrder(ctx, req)
	}

	return nil, fmt.Errorf("no engine available")
}

// CancelOrder cancels an existing order
func (c *LXClient) CancelOrder(ctx context.Context, req *pb.CancelOrderRequest) (*pb.CancelOrderResponse, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Use local engine if available
	if c.local != nil {
		return c.local.CancelOrder(ctx, req)
	}

	// Otherwise use remote
	if c.remote != nil {
		return c.remote.CancelOrder(ctx, req)
	}

	return nil, fmt.Errorf("no engine available")
}

// GetOrderBook gets the current order book
func (c *LXClient) GetOrderBook(ctx context.Context, req *pb.GetOrderBookRequest) (*pb.GetOrderBookResponse, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Use local engine if available
	if c.local != nil {
		return c.local.GetOrderBook(ctx, req)
	}

	// Otherwise use remote
	if c.remote != nil {
		return c.remote.GetOrderBook(ctx, req)
	}

	return nil, fmt.Errorf("no engine available")
}

// StreamOrderBook streams order book updates
func (c *LXClient) StreamOrderBook(ctx context.Context, req *pb.StreamOrderBookRequest) (pb.EngineService_StreamOrderBookClient, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// For streaming, prefer remote if available
	if c.remote != nil {
		return c.remote.StreamOrderBook(ctx, req)
	}

	// Local streaming not implemented yet
	return nil, fmt.Errorf("streaming not available for local engine")
}

// Close closes the client and any connections
func (c *LXClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// GetMode returns the current client mode
func (c *LXClient) GetMode() Mode {
	return c.mode
}

// IsLocal returns true if using local engine
func (c *LXClient) IsLocal() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.local != nil
}

// IsRemote returns true if using remote engine
func (c *LXClient) IsRemote() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.remote != nil
}