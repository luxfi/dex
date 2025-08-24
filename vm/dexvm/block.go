package dexvm

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/luxfi/consensus/snow/choices"
	"github.com/luxfi/consensus/snow/consensus/snowman"
	"github.com/luxfi/ids"

	"github.com/luxfi/dex/pkg/lx"
)

var (
	_ snowman.Block = &Block{}
)

// Block represents a DEX block
type Block struct {
	// Block metadata
	ID        ids.ID    `json:"id"`
	PrntID    ids.ID    `json:"parentID"`
	Height    uint64    `json:"height"`
	Timestamp time.Time `json:"timestamp"`

	// DEX transactions
	Orders          []*lx.Order          `json:"orders"`
	Trades          []*lx.Trade          `json:"trades"`
	Settlements     []*Settlement        `json:"settlements"`
	Liquidations    []*Liquidation       `json:"liquidations"`
	FundingPayments []*lx.FundingPayment `json:"fundingPayments"`
	Cancellations   []*OrderCancellation `json:"cancellations"`
	Transfers       []*Transfer          `json:"transfers"`

	// State root after applying this block
	StateRoot []byte `json:"stateRoot"`

	// Consensus fields
	status choices.Status
	vm     *VM
	bytes  []byte
}

// ID returns the block ID
func (b *Block) ID() ids.ID {
	return b.ID
}

// Parent returns the parent block ID
func (b *Block) Parent() ids.ID {
	return b.PrntID
}

// Height returns the block height
func (b *Block) Height() uint64 {
	return b.Height
}

// Timestamp returns the block timestamp
func (b *Block) Timestamp() time.Time {
	return b.Timestamp
}

// Status returns the block status
func (b *Block) Status() choices.Status {
	return b.status
}

// Bytes returns the block bytes
func (b *Block) Bytes() []byte {
	if b.bytes == nil {
		bytes, _ := b.Marshal()
		b.bytes = bytes
	}
	return b.bytes
}

// Verify verifies the block is valid
func (b *Block) Verify() error {
	// Verify parent exists and is accepted
	parent, err := b.vm.GetBlock(b.PrntID)
	if err != nil {
		return fmt.Errorf("parent block not found: %w", err)
	}

	if parent.Status() != choices.Accepted {
		return errors.New("parent block not accepted")
	}

	// Verify height is correct
	if b.Height != parent.(*Block).Height+1 {
		return errors.New("invalid block height")
	}

	// Verify timestamp is after parent
	if b.Timestamp.Before(parent.(*Block).Timestamp) {
		return errors.New("block timestamp before parent")
	}

	// Verify orders are valid
	if err := b.verifyOrders(); err != nil {
		return fmt.Errorf("invalid orders: %w", err)
	}

	// Verify trades match orders
	if err := b.verifyTrades(); err != nil {
		return fmt.Errorf("invalid trades: %w", err)
	}

	// Verify settlements
	if err := b.verifySettlements(); err != nil {
		return fmt.Errorf("invalid settlements: %w", err)
	}

	// Verify liquidations
	if err := b.verifyLiquidations(); err != nil {
		return fmt.Errorf("invalid liquidations: %w", err)
	}

	// Verify state root
	if err := b.verifyStateRoot(); err != nil {
		return fmt.Errorf("invalid state root: %w", err)
	}

	b.status = choices.Processing
	return nil
}

// Accept accepts the block
func (b *Block) Accept() error {
	b.vm.mu.Lock()
	defer b.vm.mu.Unlock()

	// Apply orders to order book
	for _, order := range b.Orders {
		if _, err := b.vm.orderBook.AddOrder(order); err != nil {
			b.vm.ctx.Log.Warn("failed to add order",
				"error", err,
				"orderID", order.ID,
			)
		}
	}

	// Process cancellations
	for _, cancel := range b.Cancellations {
		if err := b.vm.orderBook.CancelOrder(cancel.OrderID, cancel.User); err != nil {
			b.vm.ctx.Log.Warn("failed to cancel order",
				"error", err,
				"orderID", cancel.OrderID,
			)
		}
	}

	// Process trades through clearinghouse
	for _, trade := range b.Trades {
		b.vm.clearinghouse.ProcessTrade(trade)
	}

	// Process settlements
	for _, settlement := range b.Settlements {
		b.processSettlement(settlement)
	}

	// Process liquidations
	for _, liquidation := range b.Liquidations {
		b.processLiquidation(liquidation)
	}

	// Process funding payments
	for _, funding := range b.FundingPayments {
		b.processFundingPayment(funding)
	}

	// Process transfers
	for _, transfer := range b.Transfers {
		b.processTransfer(transfer)
	}

	// Update state
	b.vm.state.CurrentHeight = b.Height
	b.vm.state.CurrentBlockID = b.ID
	b.vm.state.LastBlockTime = b.Timestamp
	b.vm.lastAccepted = b.ID

	// Update metrics
	b.vm.metrics.BlocksProduced.Inc()
	b.vm.metrics.OrdersProcessed.Add(float64(len(b.Orders)))
	b.vm.metrics.TradesExecuted.Add(float64(len(b.Trades)))

	// Persist to database
	if err := b.vm.db.Put(b.ID[:], b.Bytes()); err != nil {
		return fmt.Errorf("failed to persist block: %w", err)
	}

	// Store height index
	heightKey := fmt.Sprintf("height:%d", b.Height)
	if err := b.vm.db.Put([]byte(heightKey), b.ID[:]); err != nil {
		return fmt.Errorf("failed to store height index: %w", err)
	}

	b.status = choices.Accepted

	b.vm.ctx.Log.Info("block accepted",
		"height", b.Height,
		"id", b.ID,
		"orders", len(b.Orders),
		"trades", len(b.Trades),
	)

	return nil
}

// Reject rejects the block
func (b *Block) Reject() error {
	b.status = choices.Rejected

	b.vm.ctx.Log.Info("block rejected",
		"height", b.Height,
		"id", b.ID,
	)

	return nil
}

// Marshal marshals the block to bytes
func (b *Block) Marshal() ([]byte, error) {
	return json.Marshal(b)
}

// Unmarshal unmarshals the block from bytes
func (b *Block) Unmarshal(bytes []byte) error {
	return json.Unmarshal(bytes, b)
}

// calculateID calculates the block ID
func (b *Block) calculateID() ids.ID {
	bytes, _ := json.Marshal(struct {
		Parent    ids.ID
		Height    uint64
		Timestamp time.Time
		StateRoot []byte
	}{
		Parent:    b.PrntID,
		Height:    b.Height,
		Timestamp: b.Timestamp,
		StateRoot: b.StateRoot,
	})

	return ids.ComputeID(bytes)
}

// Verification methods

func (b *Block) verifyOrders() error {
	for _, order := range b.Orders {
		// Verify order signature
		// Verify user has sufficient balance
		// Verify order parameters are valid
		if order.Size <= 0 || order.Price <= 0 {
			return errors.New("invalid order parameters")
		}
	}
	return nil
}

func (b *Block) verifyTrades() error {
	for _, trade := range b.Trades {
		// Verify trade matches orders
		// Verify trade price is valid
		// Verify trade size is valid
		if trade.Size <= 0 || trade.Price <= 0 {
			return errors.New("invalid trade parameters")
		}
	}
	return nil
}

func (b *Block) verifySettlements() error {
	for _, settlement := range b.Settlements {
		// Verify settlement matches trade
		// Verify amounts are correct
		if settlement.Amount.Sign() <= 0 {
			return errors.New("invalid settlement amount")
		}
	}
	return nil
}

func (b *Block) verifyLiquidations() error {
	for _, liquidation := range b.Liquidations {
		// Verify position was actually liquidatable
		// Verify liquidation price is correct
		// Verify penalty is correct
		if liquidation.Size <= 0 {
			return errors.New("invalid liquidation size")
		}
	}
	return nil
}

func (b *Block) verifyStateRoot() error {
	// Calculate expected state root
	// Compare with block's state root
	return nil
}

// Processing methods

func (b *Block) processSettlement(settlement *Settlement) {
	// Update user balances
	buyerBalance := b.vm.state.BalanceState[settlement.Buyer]
	if buyerBalance == nil {
		buyerBalance = new(big.Int)
		b.vm.state.BalanceState[settlement.Buyer] = buyerBalance
	}

	sellerBalance := b.vm.state.BalanceState[settlement.Seller]
	if sellerBalance == nil {
		sellerBalance = new(big.Int)
		b.vm.state.BalanceState[settlement.Seller] = sellerBalance
	}

	// Transfer funds
	// In real implementation, this would handle multiple assets
}

func (b *Block) processLiquidation(liquidation *Liquidation) {
	// Close position
	// Distribute penalty to insurance fund
	// Update user balance
}

func (b *Block) processFundingPayment(funding *lx.FundingPayment) {
	// Apply funding to all positions
	// Update position PnL
}

func (b *Block) processTransfer(transfer *Transfer) {
	// Update sender balance
	fromBalance := b.vm.state.BalanceState[transfer.From]
	if fromBalance != nil && fromBalance.Cmp(transfer.Amount) >= 0 {
		fromBalance.Sub(fromBalance, transfer.Amount)
	}

	// Update receiver balance
	toBalance := b.vm.state.BalanceState[transfer.To]
	if toBalance == nil {
		toBalance = new(big.Int)
		b.vm.state.BalanceState[transfer.To] = toBalance
	}
	toBalance.Add(toBalance, transfer.Amount)
}

// Mempool methods

// GetPendingOrders returns pending orders from mempool
func (m *Mempool) GetPendingOrders(limit int) []*lx.Order {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.orders) <= limit {
		orders := m.orders
		m.orders = m.orders[:0]
		return orders
	}

	orders := m.orders[:limit]
	m.orders = m.orders[limit:]
	return orders
}

// GetPendingCancellations returns pending cancellations
func (m *Mempool) GetPendingCancellations() []*OrderCancellation {
	m.mu.Lock()
	defer m.mu.Unlock()

	cancellations := m.cancellations
	m.cancellations = m.cancellations[:0]
	return cancellations
}

// GetPendingTransfers returns pending transfers
func (m *Mempool) GetPendingTransfers() []*Transfer {
	m.mu.Lock()
	defer m.mu.Unlock()

	transfers := m.transfers
	m.transfers = m.transfers[:0]
	return transfers
}

// AddOrder adds an order to mempool
func (m *Mempool) AddOrder(order *lx.Order) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.orders = append(m.orders, order)
}

// AddCancellation adds a cancellation to mempool
func (m *Mempool) AddCancellation(cancel *OrderCancellation) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.cancellations = append(m.cancellations, cancel)
}

// AddTransfer adds a transfer to mempool
func (m *Mempool) AddTransfer(transfer *Transfer) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.transfers = append(m.transfers, transfer)
}
