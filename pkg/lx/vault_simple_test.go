package lx

import (
	"context"
	"math/big"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestNewSimpleVaultManager tests the constructor for SimpleVaultManager
func TestNewSimpleVaultManager(t *testing.T) {
	ch := NewClearingHouse(nil, nil)

	vm := NewSimpleVaultManager(ch)

	if vm == nil {
		t.Fatal("NewSimpleVaultManager returned nil")
	}

	if vm.vaults == nil {
		t.Error("vaults map is nil")
	}

	if vm.clearinghouse != ch {
		t.Error("clearinghouse not set correctly")
	}

	if vm.strategyEngine == nil {
		t.Error("strategyEngine is nil")
	}
}

// TestCreateVaultWithStrategy_AIStrategy tests vault creation with AI strategy
func TestCreateVaultWithStrategy_AIStrategy(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6) // 1000 USDC
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	if vault == nil {
		t.Fatal("vault is nil")
	}

	if vault.Leader != leader {
		t.Errorf("expected leader %s, got %s", leader, vault.Leader)
	}

	if vault.TotalValue.Cmp(deposit) != 0 {
		t.Errorf("expected total value %s, got %s", deposit.String(), vault.TotalValue.String())
	}

	if vault.ShareSupply.Cmp(big.NewInt(1000000)) != 0 {
		t.Errorf("expected share supply 1000000, got %s", vault.ShareSupply.String())
	}

	// Leader should have all initial shares
	leaderShares := vault.MemberShares[leader]
	if leaderShares == nil || leaderShares.Cmp(vault.ShareSupply) != 0 {
		t.Error("leader should have all initial shares")
	}

	if vault.ProfitShare != 0.10 {
		t.Errorf("expected profit share 0.10, got %f", vault.ProfitShare)
	}

	if vault.Strategy == nil {
		t.Error("vault strategy is nil")
	}
}

// TestCreateVaultWithStrategy_SimpleMMStrategy tests vault creation with simple MM strategy
func TestCreateVaultWithStrategy_SimpleMMStrategy(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	leader := "leader_87654321_address"
	deposit := big.NewInt(500 * 1e6) // 500 USDC
	config := map[string]interface{}{
		"spread_bps": 10,
		"order_size": 0.1,
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "simple_mm", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	if vault == nil {
		t.Fatal("vault is nil")
	}

	if vault.Strategy == nil {
		t.Error("vault strategy is nil")
	}
}

// TestCreateVaultWithStrategy_MinimumDeposit tests minimum deposit requirement
func TestCreateVaultWithStrategy_MinimumDeposit(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	leader := "leader_12345678_address"
	deposit := big.NewInt(50 * 1e6) // 50 USDC - below minimum
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	_, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err == nil {
		t.Error("expected error for deposit below minimum")
	}

	if !strings.Contains(err.Error(), "minimum deposit") {
		t.Errorf("expected 'minimum deposit' error, got: %v", err)
	}
}

// TestCreateVaultWithStrategy_UnknownStrategy tests unknown strategy type
func TestCreateVaultWithStrategy_UnknownStrategy(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{}

	_, err := vm.CreateVaultWithStrategy(leader, deposit, "unknown_strategy", config)
	if err == nil {
		t.Error("expected error for unknown strategy")
	}

	if !strings.Contains(err.Error(), "unknown strategy") {
		t.Errorf("expected 'unknown strategy' error, got: %v", err)
	}
}

// TestDeposit tests depositing funds into a vault
func TestDeposit(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault first
	leader := "leader_12345678_address"
	initialDeposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, initialDeposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// New user deposits
	user := "user_abcdefgh_address"
	depositAmount := big.NewInt(500 * 1e6)

	err = vm.Deposit(vault.ID, user, depositAmount)
	if err != nil {
		t.Fatalf("Deposit failed: %v", err)
	}

	// Check vault totals updated
	expectedTotal := new(big.Int).Add(initialDeposit, depositAmount)
	if vault.TotalValue.Cmp(expectedTotal) != 0 {
		t.Errorf("expected total value %s, got %s", expectedTotal.String(), vault.TotalValue.String())
	}

	// Check user has shares
	userShares := vault.MemberShares[user]
	if userShares == nil || userShares.Sign() == 0 {
		t.Error("user should have shares after deposit")
	}

	// Check share supply increased
	if vault.ShareSupply.Cmp(big.NewInt(1000000)) <= 0 {
		t.Error("share supply should have increased")
	}
}

// TestDeposit_VaultNotFound tests deposit to non-existent vault
func TestDeposit_VaultNotFound(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	err := vm.Deposit("non_existent_vault", "user", big.NewInt(100*1e6))
	if err == nil {
		t.Error("expected error for non-existent vault")
	}

	if !strings.Contains(err.Error(), "vault not found") {
		t.Errorf("expected 'vault not found' error, got: %v", err)
	}
}

// TestWithdraw tests withdrawing funds from a vault
func TestWithdraw(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	initialDeposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, initialDeposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Leader withdraws 50% of their shares
	withdrawn, err := vm.Withdraw(vault.ID, leader, 0.5)
	if err != nil {
		t.Fatalf("Withdraw failed: %v", err)
	}

	if withdrawn == nil || withdrawn.Sign() == 0 {
		t.Error("withdrawn amount should be positive")
	}

	// Check shares reduced
	leaderShares := vault.MemberShares[leader]
	if leaderShares == nil {
		t.Error("leader should still have shares")
	}

	// Original was 1M shares, after 50% withdrawal should be ~500K
	expectedShares := big.NewInt(500000)
	if leaderShares.Cmp(expectedShares) != 0 {
		t.Errorf("expected ~%s shares, got %s", expectedShares.String(), leaderShares.String())
	}
}

// TestWithdraw_InvalidSharePercent tests invalid share percent
func TestWithdraw_InvalidSharePercent(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Test zero percent
	_, err := vm.Withdraw("vault_id", "user", 0)
	if err == nil {
		t.Error("expected error for 0 share percent")
	}

	// Test negative percent
	_, err = vm.Withdraw("vault_id", "user", -0.5)
	if err == nil {
		t.Error("expected error for negative share percent")
	}

	// Test over 100%
	_, err = vm.Withdraw("vault_id", "user", 1.5)
	if err == nil {
		t.Error("expected error for >100% share percent")
	}
}

// TestWithdraw_VaultNotFound tests withdrawal from non-existent vault
func TestWithdraw_VaultNotFound(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	_, err := vm.Withdraw("non_existent_vault", "user", 0.5)
	if err == nil {
		t.Error("expected error for non-existent vault")
	}

	if !strings.Contains(err.Error(), "vault not found") {
		t.Errorf("expected 'vault not found' error, got: %v", err)
	}
}

// TestWithdraw_NoSharesOwned tests withdrawal with no shares
func TestWithdraw_NoSharesOwned(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Non-member tries to withdraw
	_, err = vm.Withdraw(vault.ID, "non_member", 0.5)
	if err == nil {
		t.Error("expected error for user with no shares")
	}

	if !strings.Contains(err.Error(), "no shares owned") {
		t.Errorf("expected 'no shares owned' error, got: %v", err)
	}
}

// TestWithdraw_FullWithdrawal tests 100% withdrawal removes user from members
func TestWithdraw_FullWithdrawal(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault and add second member
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Add another user
	user := "user_abcdefgh_address"
	err = vm.Deposit(vault.ID, user, big.NewInt(500*1e6))
	if err != nil {
		t.Fatalf("Deposit failed: %v", err)
	}

	// User withdraws 100%
	_, err = vm.Withdraw(vault.ID, user, 1.0)
	if err != nil {
		t.Fatalf("Withdraw failed: %v", err)
	}

	// User should be removed from members
	if _, exists := vault.MemberShares[user]; exists {
		t.Error("user should be removed from members after full withdrawal")
	}
}

// TestWithdraw_ProfitShare tests profit share calculation for non-leaders
func TestWithdraw_ProfitShare(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Add another user
	user := "user_abcdefgh_address"
	userDeposit := big.NewInt(1000 * 1e6) // Equal deposit
	err = vm.Deposit(vault.ID, user, userDeposit)
	if err != nil {
		t.Fatalf("Deposit failed: %v", err)
	}

	// Simulate profit by increasing vault value above high water mark
	vault.mu.Lock()
	vault.TotalValue = big.NewInt(3000 * 1e6) // 50% profit
	vault.mu.Unlock()

	// User withdraws - should have profit share deducted
	withdrawn, err := vm.Withdraw(vault.ID, user, 1.0)
	if err != nil {
		t.Fatalf("Withdraw failed: %v", err)
	}

	// Withdrawn amount should be less than proportional share due to profit share
	// User had 50% of shares, so would get 1500 USDC minus profit share fee
	maxExpected := big.NewInt(1500 * 1e6)
	if withdrawn.Cmp(maxExpected) >= 0 {
		t.Error("withdrawn should be less than max due to profit share")
	}
}

// TestSimpleVault_UpdateVaultValue tests updating vault value based on strategy PnL
func TestSimpleVault_UpdateVaultValue(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	initialValue := new(big.Int).Set(vault.TotalValue)

	err = vm.UpdateVaultValue(vault.ID)
	if err != nil {
		t.Fatalf("UpdateVaultValue failed: %v", err)
	}

	// Value should be updated (AI strategy returns 0 PnL by default, so unchanged)
	// But high water mark should match total value
	if vault.HighWaterMark.Cmp(initialValue) != 0 {
		t.Error("high water mark should equal initial value")
	}
}

// TestSimpleVault_UpdateVaultValue_VaultNotFound tests update for non-existent vault
func TestSimpleVault_UpdateVaultValue_VaultNotFound(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	err := vm.UpdateVaultValue("non_existent_vault")
	if err == nil {
		t.Error("expected error for non-existent vault")
	}

	if !strings.Contains(err.Error(), "vault not found") {
		t.Errorf("expected 'vault not found' error, got: %v", err)
	}
}

// TestGetVaultInfo tests retrieving vault information
func TestGetVaultInfo(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	info, err := vm.GetVaultInfo(vault.ID)
	if err != nil {
		t.Fatalf("GetVaultInfo failed: %v", err)
	}

	if info == nil {
		t.Fatal("info is nil")
	}

	// Check required fields
	if info["id"] != vault.ID {
		t.Errorf("expected id %s, got %s", vault.ID, info["id"])
	}

	if info["leader"] != leader {
		t.Errorf("expected leader %s, got %s", leader, info["leader"])
	}

	if info["total_value"] != deposit.String() {
		t.Errorf("expected total_value %s, got %s", deposit.String(), info["total_value"])
	}

	if info["member_count"] != 1 {
		t.Errorf("expected member_count 1, got %v", info["member_count"])
	}

	// Check strategy metrics are present
	if _, exists := info["strategy_pnl"]; !exists {
		t.Error("strategy_pnl should be present")
	}

	if _, exists := info["trade_count"]; !exists {
		t.Error("trade_count should be present")
	}

	if _, exists := info["win_rate"]; !exists {
		t.Error("win_rate should be present")
	}

	if _, exists := info["sharpe_ratio"]; !exists {
		t.Error("sharpe_ratio should be present")
	}
}

// TestGetVaultInfo_VaultNotFound tests GetVaultInfo for non-existent vault
func TestGetVaultInfo_VaultNotFound(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	_, err := vm.GetVaultInfo("non_existent_vault")
	if err == nil {
		t.Error("expected error for non-existent vault")
	}

	if !strings.Contains(err.Error(), "vault not found") {
		t.Errorf("expected 'vault not found' error, got: %v", err)
	}
}

// TestStartStrategyExecution tests starting strategy execution
func TestStartStrategyExecution(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	_, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Start strategy execution in background
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	vm.StartStrategyExecution(ctx)

	// Let it run briefly
	time.Sleep(50 * time.Millisecond)

	// Should not panic or error - strategy engine should be running
}

// TestExecuteVaultOrder tests executing an order for a vault
func TestExecuteVaultOrder(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Create an order
	order := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000.0,
		Size:   0.1,
	}

	err = vm.ExecuteVaultOrder(vault.ID, order)
	if err != nil {
		t.Fatalf("ExecuteVaultOrder failed: %v", err)
	}

	// Order should be assigned to vault's subaccount
	if order.User != vault.SubaccountID {
		t.Errorf("expected order user %s, got %s", vault.SubaccountID, order.User)
	}
}

// TestExecuteVaultOrder_VaultNotFound tests order execution for non-existent vault
func TestExecuteVaultOrder_VaultNotFound(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	order := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000.0,
		Size:   0.1,
	}

	err := vm.ExecuteVaultOrder("non_existent_vault", order)
	if err == nil {
		t.Error("expected error for non-existent vault")
	}

	if !strings.Contains(err.Error(), "vault not found") {
		t.Errorf("expected 'vault not found' error, got: %v", err)
	}
}

// TestExecuteVaultOrder_NoMembers tests order execution for vault with no members
func TestExecuteVaultOrder_NoMembers(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Remove all members (simulating empty vault)
	vault.mu.Lock()
	vault.MemberShares = make(map[string]*big.Int)
	vault.mu.Unlock()

	order := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000.0,
		Size:   0.1,
	}

	err = vm.ExecuteVaultOrder(vault.ID, order)
	if err == nil {
		t.Error("expected error for vault with no members")
	}

	if !strings.Contains(err.Error(), "no members") {
		t.Errorf("expected 'no members' error, got: %v", err)
	}
}

// TestCreateStrategy_AIStrategy tests AI strategy creation
func TestCreateStrategy_AIStrategy(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	strategy, err := vm.createStrategy("ai_hft", config)
	if err != nil {
		t.Fatalf("createStrategy failed: %v", err)
	}

	if strategy == nil {
		t.Fatal("strategy is nil")
	}

	// Check it's an AI strategy
	_, ok := strategy.(*AIStrategy)
	if !ok {
		t.Error("expected AIStrategy type")
	}
}

// TestCreateStrategy_SimpleMMStrategy tests simple MM strategy creation
func TestCreateStrategy_SimpleMMStrategy(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	config := map[string]interface{}{
		"spread_bps": 10,
		"order_size": 0.1,
	}

	strategy, err := vm.createStrategy("simple_mm", config)
	if err != nil {
		t.Fatalf("createStrategy failed: %v", err)
	}

	if strategy == nil {
		t.Fatal("strategy is nil")
	}

	// Check it's a SimpleMMStrategy
	_, ok := strategy.(*SimpleMMStrategy)
	if !ok {
		t.Error("expected SimpleMMStrategy type")
	}
}

// TestCreateStrategy_UnknownStrategy tests unknown strategy creation
func TestCreateStrategy_UnknownStrategy(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	_, err := vm.createStrategy("unknown_strategy", map[string]interface{}{})
	if err == nil {
		t.Error("expected error for unknown strategy")
	}

	if !strings.Contains(err.Error(), "unknown strategy") {
		t.Errorf("expected 'unknown strategy' error, got: %v", err)
	}
}

// TestMultipleVaults tests creating multiple vaults
func TestMultipleVaults(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create first vault
	vault1, err := vm.CreateVaultWithStrategy(
		"leader1_12345678",
		big.NewInt(1000*1e6),
		"ai_hft",
		map[string]interface{}{"model_endpoint": "localhost:8080"},
	)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy vault1 failed: %v", err)
	}

	// Create second vault
	vault2, err := vm.CreateVaultWithStrategy(
		"leader2_87654321",
		big.NewInt(2000*1e6),
		"simple_mm",
		map[string]interface{}{"spread_bps": 10, "order_size": 0.1},
	)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy vault2 failed: %v", err)
	}

	// Verify vaults are distinct
	if vault1.ID == vault2.ID {
		t.Error("vault IDs should be different")
	}

	// Verify each vault is in the map
	if len(vm.vaults) != 2 {
		t.Errorf("expected 2 vaults, got %d", len(vm.vaults))
	}
}

// TestConcurrentDeposits tests concurrent deposits to same vault
func TestConcurrentDeposits(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Concurrent deposits
	var wg sync.WaitGroup
	numDeposits := 10
	depositAmount := big.NewInt(100 * 1e6)

	for i := 0; i < numDeposits; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			user := "user_" + string(rune('a'+idx)) + "_address"
			err := vm.Deposit(vault.ID, user, depositAmount)
			if err != nil {
				t.Errorf("Deposit %d failed: %v", idx, err)
			}
		}(i)
	}

	wg.Wait()

	// Check total value
	expectedTotal := new(big.Int).Add(deposit, new(big.Int).Mul(depositAmount, big.NewInt(int64(numDeposits))))
	if vault.TotalValue.Cmp(expectedTotal) != 0 {
		t.Errorf("expected total value %s, got %s", expectedTotal.String(), vault.TotalValue.String())
	}
}

// TestVaultIDFormat tests that vault IDs follow expected format
func TestVaultIDFormat(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// ID should start with "v_" and contain leader prefix
	if !strings.HasPrefix(vault.ID, "v_") {
		t.Errorf("vault ID should start with 'v_', got %s", vault.ID)
	}

	// Should contain first 8 chars of leader
	if !strings.Contains(vault.ID, leader[:8]) {
		t.Errorf("vault ID should contain leader prefix, got %s", vault.ID)
	}
}

// TestSubaccountIDFormat tests that subaccount IDs follow expected format
func TestSubaccountIDFormat(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Subaccount ID should start with "vault_"
	if !strings.HasPrefix(vault.SubaccountID, "vault_") {
		t.Errorf("subaccount ID should start with 'vault_', got %s", vault.SubaccountID)
	}
}

// TestHighWaterMarkUpdates tests that high water mark updates correctly
func TestHighWaterMarkUpdates(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	// Initial HWM should equal deposit
	if vault.HighWaterMark.Cmp(deposit) != 0 {
		t.Errorf("initial HWM should equal deposit")
	}

	// Simulate profit
	vault.mu.Lock()
	vault.TotalValue = big.NewInt(1500 * 1e6)
	vault.mu.Unlock()

	// Update vault value should update HWM
	err = vm.UpdateVaultValue(vault.ID)
	if err != nil {
		t.Fatalf("UpdateVaultValue failed: %v", err)
	}

	// HWM should be updated (at least to current value)
	if vault.HighWaterMark.Cmp(deposit) <= 0 {
		t.Error("HWM should have increased with profit")
	}
}

// TestShareCalculation tests share minting calculation
func TestShareCalculation(t *testing.T) {
	ch := NewClearingHouse(nil, nil)
	vm := NewSimpleVaultManager(ch)

	// Create vault
	leader := "leader_12345678_address"
	deposit := big.NewInt(1000 * 1e6)
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080",
	}

	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		t.Fatalf("CreateVaultWithStrategy failed: %v", err)
	}

	initialShares := new(big.Int).Set(vault.ShareSupply)

	// User deposits same amount
	user := "user_abcdefgh_address"
	err = vm.Deposit(vault.ID, user, big.NewInt(1000*1e6))
	if err != nil {
		t.Fatalf("Deposit failed: %v", err)
	}

	// User should get same number of shares as initial
	userShares := vault.MemberShares[user]
	if userShares.Cmp(initialShares) != 0 {
		t.Errorf("user should get same shares as initial, expected %s, got %s",
			initialShares.String(), userShares.String())
	}
}
