package gateway

import (
	"math/big"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// OrderStore is a thread-safe in-memory order store.
// Keyed by orderId (UUID). Supports lookup by owner address.
type OrderStore struct {
	mu     sync.RWMutex
	orders map[string]*GatewayOrder       // orderId -> order
	byAddr map[string]map[string]struct{} // lowercase(owner) -> set of orderIds
}

// NewOrderStore creates an empty order store.
func NewOrderStore() *OrderStore {
	return &OrderStore{
		orders: make(map[string]*GatewayOrder),
		byAddr: make(map[string]map[string]struct{}),
	}
}

// Create inserts a new order derived from a validated CreateOrderRequest.
// Returns the created order. The caller must have validated the request first.
func (s *OrderStore) Create(req *CreateOrderRequest) *GatewayOrder {
	id := uuid.New().String()
	now := time.Now().UTC()

	order := &GatewayOrder{
		ID:           id,
		Kind:         req.Type,
		Status:       OrderStatusOpen,
		Owner:        strings.ToLower(req.Recipient),
		TokenIn:      strings.ToLower(req.TokenIn),
		TokenOut:     strings.ToLower(req.TokenOut),
		AmountIn:     parseBigIntStr(req.AmountIn),
		ChainID:      ChainID(req.ChainID),
		Deadline:     req.Deadline,
		Recipient:    strings.ToLower(req.Recipient),
		CreatedAt:    now,
		UpdatedAt:    now,
		FilledAmount: big.NewInt(0),
		Fills:        make([]Fill, 0),
	}

	switch req.Type {
	case OrderKindLimit:
		order.LimitPrice = parseBigIntStr(req.LimitPrice)
	case OrderKindDutch:
		order.StartAmount = parseBigIntStr(req.StartAmount)
		order.EndAmount = parseBigIntStr(req.EndAmount)
		order.DecayStartTime = req.DecayStartTime
		order.DecayEndTime = req.DecayEndTime
	}

	s.mu.Lock()
	s.orders[id] = order
	ownerKey := strings.ToLower(req.Recipient)
	if s.byAddr[ownerKey] == nil {
		s.byAddr[ownerKey] = make(map[string]struct{})
	}
	s.byAddr[ownerKey][id] = struct{}{}
	s.mu.Unlock()

	return order
}

// Get retrieves an order by ID. Returns nil if not found.
func (s *OrderStore) Get(orderID string) *GatewayOrder {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.orders[orderID]
}

// ListByOwner returns all orders for an owner address, optionally filtered by status.
func (s *OrderStore) ListByOwner(owner string, statusFilter OrderStatus) []GatewayOrder {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ownerKey := strings.ToLower(owner)
	ids, ok := s.byAddr[ownerKey]
	if !ok {
		return nil
	}

	result := make([]GatewayOrder, 0, len(ids))
	for id := range ids {
		order := s.orders[id]
		if order == nil {
			continue
		}
		if statusFilter != "" && order.Status != statusFilter {
			continue
		}
		result = append(result, *order)
	}
	return result
}

// Cancel transitions an order to cancelled state.
// Returns the updated order or an error if the transition is invalid.
func (s *OrderStore) Cancel(orderID string) (*GatewayOrder, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	order, ok := s.orders[orderID]
	if !ok {
		return nil, ErrOrderNotFoundGateway
	}

	if err := ValidateTransition(order.Status, OrderStatusCancelled); err != nil {
		return nil, err
	}

	order.Status = OrderStatusCancelled
	order.UpdatedAt = time.Now().UTC()
	return order, nil
}

// RecordFill records a fill against an order, updating status accordingly.
// This is used by solvers/settlement to report fills.
func (s *OrderStore) RecordFill(orderID string, amount *big.Int, price *big.Int, txHash string) (*GatewayOrder, error) {
	if amount == nil || amount.Sign() <= 0 {
		return nil, ErrMissingAmountIn
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	order, ok := s.orders[orderID]
	if !ok {
		return nil, ErrOrderNotFoundGateway
	}

	if order.Status.IsTerminal() {
		return nil, ErrOrderTerminal
	}

	remaining := new(big.Int).Sub(order.AmountIn, order.FilledAmount)
	if amount.Cmp(remaining) > 0 {
		return nil, ErrFillExceedsRemaining
	}

	now := time.Now().UTC()

	order.FilledAmount.Add(order.FilledAmount, amount)
	order.Fills = append(order.Fills, Fill{
		Amount:    new(big.Int).Set(amount),
		Price:     price,
		TxHash:    txHash,
		Timestamp: now,
	})

	// Determine new status
	var newStatus OrderStatus
	if order.FilledAmount.Cmp(order.AmountIn) >= 0 {
		newStatus = OrderStatusFilled
	} else {
		newStatus = OrderStatusPartiallyFilled
	}

	if err := ValidateTransition(order.Status, newStatus); err != nil {
		return nil, err
	}

	order.Status = newStatus
	order.UpdatedAt = now
	return order, nil
}

// ExpireStale transitions all non-terminal orders past their deadline to expired.
// Returns the number of orders expired.
func (s *OrderStore) ExpireStale() int {
	now := time.Now().Unix()

	s.mu.Lock()
	defer s.mu.Unlock()

	count := 0
	for _, order := range s.orders {
		if order.Status.IsTerminal() {
			continue
		}
		if order.Deadline > 0 && now > order.Deadline {
			if err := ValidateTransition(order.Status, OrderStatusExpired); err == nil {
				order.Status = OrderStatusExpired
				order.UpdatedAt = time.Now().UTC()
				count++
			}
		}
	}
	return count
}

// Count returns the total number of orders in the store.
func (s *OrderStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.orders)
}
