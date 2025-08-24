package lx

import (
	"context"
	"errors"
	"fmt"
	"math/big"
	"sync"
	"time"
)

// StakingPool represents a staking pool for validators and delegators
type StakingPool struct {
	// Pool identification
	ID          string
	Name        string
	Description string

	// Staking parameters
	MinStake           *big.Int      // Minimum stake amount
	MaxStake           *big.Int      // Maximum stake per user
	LockupPeriod       time.Duration // Lock period for staked tokens
	UnbondingPeriod    time.Duration // Time to wait for withdrawal
	CompoundingEnabled bool          // Auto-compound rewards

	// Rewards configuration
	APY            float64       // Annual percentage yield
	RewardToken    string        // Token used for rewards
	RewardPool     *big.Int      // Total rewards available
	RewardInterval time.Duration // How often rewards are distributed
	LastRewardTime time.Time     // Last reward distribution

	// Pool state
	TotalStaked      *big.Int                  // Total amount staked
	ActiveStakers    map[string]*StakePosition // address -> position
	PendingUnbonding map[string]*UnbondingRequest

	// Slashing parameters
	SlashingEnabled    bool
	SlashingRate       float64 // Percentage to slash for misbehavior
	SlashingConditions []SlashingCondition

	// Pool status
	Active    bool
	Paused    bool
	CreatedAt time.Time

	mu sync.RWMutex
}

// StakePosition represents a user's staking position
type StakePosition struct {
	Owner              string
	Amount             *big.Int
	Shares             *big.Int // Share tokens representing stake
	RewardsEarned      *big.Int
	RewardsClaimed     *big.Int
	StakedAt           time.Time
	LastClaimTime      time.Time
	LockedUntil        time.Time
	CompoundingEnabled bool
	Delegations        map[string]*big.Int // validator -> amount
}

// UnbondingRequest represents a withdrawal request
type UnbondingRequest struct {
	Owner       string
	Amount      *big.Int
	Shares      *big.Int
	RequestedAt time.Time
	AvailableAt time.Time
	Completed   bool
	CompletedAt time.Time
}

// SlashingCondition defines when slashing occurs
type SlashingCondition struct {
	Type        string  // "downtime", "double_sign", "invalid_block"
	Threshold   float64 // Threshold for triggering
	SlashAmount float64 // Percentage to slash
}

// ValidatorPool represents staking for validators specifically
type ValidatorPool struct {
	*StakingPool

	// Validator-specific parameters
	MaxValidators     int
	MinSelfStake      *big.Int // Minimum self-stake for validators
	CommissionRate    float64  // Commission on delegator rewards
	MaxCommissionRate float64  // Maximum allowed commission

	// Active validators
	Validators     map[string]*ValidatorInfo
	ValidatorQueue []*ValidatorCandidate // Waiting to become active

	// Performance tracking
	ValidatorMetrics map[string]*ValidatorMetrics

	// Delegation
	MaxDelegations int      // Max delegations per delegator
	MinDelegation  *big.Int // Minimum delegation amount
}

// ValidatorInfo represents an active validator
type ValidatorInfo struct {
	Address        string
	PublicKey      []byte
	SelfStake      *big.Int
	TotalStake     *big.Int // Self + delegated
	Commission     float64
	Active         bool
	Jailed         bool // Temporarily removed for misbehavior
	JailedUntil    time.Time
	Performance    float64 // Uptime percentage
	BlocksProduced uint64
	BlocksMissed   uint64
	LastActive     time.Time
	Delegators     map[string]*big.Int
}

// ValidatorCandidate represents someone waiting to become a validator
type ValidatorCandidate struct {
	Address         string
	SelfStake       *big.Int
	ApplicationTime time.Time
	Approved        bool
}

// ValidatorMetrics tracks validator performance
type ValidatorMetrics struct {
	Uptime          float64
	BlocksProduced  uint64
	BlocksMissed    uint64
	SlashingEvents  int
	RewardsEarned   *big.Int
	DelegatorsCount int
	AverageAPY      float64
}

// StakingManager manages all staking pools
type StakingManager struct {
	Pools          map[string]*StakingPool
	ValidatorPools map[string]*ValidatorPool

	// Global parameters
	NativeToken      string
	TotalValueLocked *big.Int

	// Rewards distribution
	RewardDistributor *RewardDistributor

	// Governance integration
	GovernanceVoting map[string]*VotingPower

	mu sync.RWMutex
}

// RewardDistributor handles reward calculations and distribution
type RewardDistributor struct {
	RewardSchedule   map[string]*RewardSchedule // poolID -> schedule
	PendingRewards   map[string]*big.Int        // user -> pending
	CompoundingUsers map[string]bool            // users with auto-compound

	mu sync.RWMutex
}

// RewardSchedule defines how rewards are distributed
type RewardSchedule struct {
	PoolID             string
	RewardRate         *big.Int // Rewards per second
	StartTime          time.Time
	EndTime            time.Time
	TotalRewards       *big.Int
	DistributedRewards *big.Int
}

// VotingPower represents governance voting power from staking
type VotingPower struct {
	Address         string
	StakedAmount    *big.Int
	VotingPower     *big.Int // Can be different from staked amount
	Multiplier      float64  // Based on lock duration
	ActiveProposals []string // Proposal IDs user is voting on
}

// NewStakingManager creates a new staking manager
func NewStakingManager(nativeToken string) *StakingManager {
	return &StakingManager{
		Pools:            make(map[string]*StakingPool),
		ValidatorPools:   make(map[string]*ValidatorPool),
		NativeToken:      nativeToken,
		TotalValueLocked: big.NewInt(0),
		RewardDistributor: &RewardDistributor{
			RewardSchedule:   make(map[string]*RewardSchedule),
			PendingRewards:   make(map[string]*big.Int),
			CompoundingUsers: make(map[string]bool),
		},
		GovernanceVoting: make(map[string]*VotingPower),
	}
}

// CreateStakingPool creates a new staking pool
func (sm *StakingManager) CreateStakingPool(
	name string,
	minStake *big.Int,
	lockupPeriod time.Duration,
	apy float64,
) (*StakingPool, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	poolID := fmt.Sprintf("pool_%s_%d", name, time.Now().Unix())

	pool := &StakingPool{
		ID:               poolID,
		Name:             name,
		MinStake:         minStake,
		MaxStake:         new(big.Int).Mul(minStake, big.NewInt(10000)), // 10000x min
		LockupPeriod:     lockupPeriod,
		UnbondingPeriod:  7 * 24 * time.Hour, // 7 days default
		APY:              apy,
		RewardToken:      sm.NativeToken,
		RewardPool:       big.NewInt(0),
		RewardInterval:   1 * time.Hour,
		TotalStaked:      big.NewInt(0),
		ActiveStakers:    make(map[string]*StakePosition),
		PendingUnbonding: make(map[string]*UnbondingRequest),
		Active:           true,
		CreatedAt:        time.Now(),
		LastRewardTime:   time.Now(),
	}

	sm.Pools[poolID] = pool

	// Create reward schedule
	sm.RewardDistributor.RewardSchedule[poolID] = &RewardSchedule{
		PoolID:             poolID,
		RewardRate:         sm.calculateRewardRate(apy, minStake),
		StartTime:          time.Now(),
		EndTime:            time.Now().Add(365 * 24 * time.Hour), // 1 year
		TotalRewards:       big.NewInt(0),
		DistributedRewards: big.NewInt(0),
	}

	return pool, nil
}

// CreateValidatorPool creates a pool for validators
func (sm *StakingManager) CreateValidatorPool(
	maxValidators int,
	minSelfStake *big.Int,
	commissionRate float64,
) (*ValidatorPool, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if commissionRate < 0 || commissionRate > 1 {
		return nil, errors.New("commission rate must be between 0 and 1")
	}

	// Create base staking pool
	basePool := &StakingPool{
		ID:               fmt.Sprintf("validator_pool_%d", time.Now().Unix()),
		Name:             "Validator Staking Pool",
		MinStake:         minSelfStake,
		MaxStake:         new(big.Int).Mul(minSelfStake, big.NewInt(1000)),
		LockupPeriod:     21 * 24 * time.Hour, // 21 days for validators
		UnbondingPeriod:  21 * 24 * time.Hour,
		APY:              12.0, // 12% APY for validators
		RewardToken:      sm.NativeToken,
		RewardPool:       big.NewInt(0),
		RewardInterval:   1 * time.Hour,
		TotalStaked:      big.NewInt(0),
		ActiveStakers:    make(map[string]*StakePosition),
		PendingUnbonding: make(map[string]*UnbondingRequest),
		SlashingEnabled:  true,
		SlashingRate:     0.05, // 5% default slashing
		Active:           true,
		CreatedAt:        time.Now(),
	}

	// Add slashing conditions
	basePool.SlashingConditions = []SlashingCondition{
		{Type: "downtime", Threshold: 0.1, SlashAmount: 0.01},    // 1% slash for 10% downtime
		{Type: "double_sign", Threshold: 1, SlashAmount: 0.05},   // 5% slash for double signing
		{Type: "invalid_block", Threshold: 1, SlashAmount: 0.02}, // 2% slash for invalid block
	}

	validatorPool := &ValidatorPool{
		StakingPool:       basePool,
		MaxValidators:     maxValidators,
		MinSelfStake:      minSelfStake,
		CommissionRate:    commissionRate,
		MaxCommissionRate: 0.2, // 20% max commission
		Validators:        make(map[string]*ValidatorInfo),
		ValidatorQueue:    make([]*ValidatorCandidate, 0),
		ValidatorMetrics:  make(map[string]*ValidatorMetrics),
		MaxDelegations:    100,
		MinDelegation:     new(big.Int).Div(minSelfStake, big.NewInt(100)), // 1% of min self stake
	}

	sm.ValidatorPools[basePool.ID] = validatorPool
	sm.Pools[basePool.ID] = basePool // Add to regular pools too for staking functions

	return validatorPool, nil
}

// stakeInternal is an internal function that assumes the lock is already held
func (sm *StakingManager) stakeInternal(
	poolID string,
	user string,
	amount *big.Int,
) (*StakePosition, error) {

	pool, exists := sm.Pools[poolID]
	if !exists {
		return nil, errors.New("pool not found")
	}

	if pool.Paused {
		return nil, errors.New("pool is paused")
	}

	// Validate amount
	if amount.Cmp(pool.MinStake) < 0 {
		return nil, fmt.Errorf("amount below minimum: %s", pool.MinStake)
	}

	position := pool.ActiveStakers[user]
	if position == nil {
		// New position
		position = &StakePosition{
			Owner:          user,
			Amount:         big.NewInt(0),
			Shares:         big.NewInt(0),
			RewardsEarned:  big.NewInt(0),
			RewardsClaimed: big.NewInt(0),
			StakedAt:       time.Now(),
			LastClaimTime:  time.Now(),
			LockedUntil:    time.Now().Add(pool.LockupPeriod),
			Delegations:    make(map[string]*big.Int),
		}
		pool.ActiveStakers[user] = position
	}

	// Check max stake
	newAmount := new(big.Int).Add(position.Amount, amount)
	if newAmount.Cmp(pool.MaxStake) > 0 {
		return nil, fmt.Errorf("exceeds maximum stake: %s", pool.MaxStake)
	}

	// Calculate shares (1:1 for first staker, proportional after)
	var shares *big.Int
	if pool.TotalStaked.Sign() == 0 {
		shares = amount
	} else {
		totalShares := sm.getTotalShares(pool)
		shares = new(big.Int).Mul(amount, totalShares)
		shares.Div(shares, pool.TotalStaked)
	}

	// Update position
	position.Amount.Add(position.Amount, amount)
	position.Shares.Add(position.Shares, shares)
	position.LockedUntil = time.Now().Add(pool.LockupPeriod)

	// Update pool
	pool.TotalStaked.Add(pool.TotalStaked, amount)
	sm.TotalValueLocked.Add(sm.TotalValueLocked, amount)

	// Update voting power
	sm.updateVotingPower(user, position)

	return position, nil
}

// Stake stakes tokens in a pool (public interface)
func (sm *StakingManager) Stake(
	poolID string,
	user string,
	amount *big.Int,
) (*StakePosition, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	return sm.stakeInternal(poolID, user, amount)
}

// Unstake initiates unstaking process
func (sm *StakingManager) Unstake(
	poolID string,
	user string,
	shares *big.Int,
) (*UnbondingRequest, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	pool, exists := sm.Pools[poolID]
	if !exists {
		return nil, errors.New("pool not found")
	}

	position := pool.ActiveStakers[user]
	if position == nil {
		return nil, errors.New("no staking position found")
	}

	if shares.Cmp(position.Shares) > 0 {
		return nil, errors.New("insufficient shares")
	}

	// Check lockup period
	if time.Now().Before(position.LockedUntil) {
		return nil, fmt.Errorf("tokens locked until %s", position.LockedUntil)
	}

	// Calculate amount to unstake
	amount := new(big.Int).Mul(shares, pool.TotalStaked)
	totalShares := sm.getTotalShares(pool)
	amount.Div(amount, totalShares)

	// Claim pending rewards first
	rewards := sm.calculateRewards(pool, position)
	position.RewardsEarned.Add(position.RewardsEarned, rewards)

	// Create unbonding request
	unbonding := &UnbondingRequest{
		Owner:       user,
		Amount:      amount,
		Shares:      shares,
		RequestedAt: time.Now(),
		AvailableAt: time.Now().Add(pool.UnbondingPeriod),
	}

	// Update position
	position.Shares.Sub(position.Shares, shares)
	position.Amount.Sub(position.Amount, amount)

	// Remove position if fully unstaked
	if position.Shares.Sign() == 0 {
		delete(pool.ActiveStakers, user)
	}

	// Update pool
	pool.TotalStaked.Sub(pool.TotalStaked, amount)
	sm.TotalValueLocked.Sub(sm.TotalValueLocked, amount)

	// Store unbonding request
	requestID := fmt.Sprintf("%s_%d", user, time.Now().Unix())
	pool.PendingUnbonding[requestID] = unbonding

	// Update voting power
	sm.updateVotingPower(user, position)

	return unbonding, nil
}

// ClaimRewards claims accumulated rewards
func (sm *StakingManager) ClaimRewards(poolID string, user string) (*big.Int, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	pool, exists := sm.Pools[poolID]
	if !exists {
		return nil, errors.New("pool not found")
	}

	position := pool.ActiveStakers[user]
	if position == nil {
		return nil, errors.New("no staking position found")
	}

	// Calculate rewards
	rewards := sm.calculateRewards(pool, position)
	position.RewardsEarned.Add(position.RewardsEarned, rewards)

	// Get claimable amount
	claimable := new(big.Int).Sub(position.RewardsEarned, position.RewardsClaimed)
	if claimable.Sign() <= 0 {
		return big.NewInt(0), nil
	}

	// Update position
	position.RewardsClaimed.Add(position.RewardsClaimed, claimable)
	position.LastClaimTime = time.Now()

	// Handle compounding if enabled
	if position.CompoundingEnabled {
		// Stake the rewards (use internal method since we already hold the lock)
		sm.stakeInternal(poolID, user, claimable)
		return claimable, nil
	}

	return claimable, nil
}

// BecomeValidator applies to become a validator
func (sm *StakingManager) BecomeValidator(
	poolID string,
	address string,
	selfStake *big.Int,
	publicKey []byte,
	commission float64,
) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	vPool, exists := sm.ValidatorPools[poolID]
	if !exists {
		return errors.New("validator pool not found")
	}

	// Check self-stake requirement
	if selfStake.Cmp(vPool.MinSelfStake) < 0 {
		return fmt.Errorf("self-stake below minimum: %s", vPool.MinSelfStake)
	}

	// Check commission rate
	if commission > vPool.MaxCommissionRate {
		return fmt.Errorf("commission exceeds maximum: %.2f%%", vPool.MaxCommissionRate*100)
	}

	// Check if already a validator
	if _, exists := vPool.Validators[address]; exists {
		return errors.New("already a validator")
	}

	// Check validator slots
	if len(vPool.Validators) >= vPool.MaxValidators {
		// Add to queue
		candidate := &ValidatorCandidate{
			Address:         address,
			SelfStake:       selfStake,
			ApplicationTime: time.Now(),
		}
		vPool.ValidatorQueue = append(vPool.ValidatorQueue, candidate)
		return nil
	}

	// Create validator
	validator := &ValidatorInfo{
		Address:     address,
		PublicKey:   publicKey,
		SelfStake:   selfStake,
		TotalStake:  selfStake,
		Commission:  commission,
		Active:      true,
		Performance: 100.0,
		LastActive:  time.Now(),
		Delegators:  make(map[string]*big.Int),
	}

	vPool.Validators[address] = validator

	// Create metrics
	vPool.ValidatorMetrics[address] = &ValidatorMetrics{
		Uptime:        100.0,
		RewardsEarned: big.NewInt(0),
	}

	// Stake the self-stake (use internal method since we already hold the lock)
	sm.stakeInternal(poolID, address, selfStake)

	return nil
}

// Delegate delegates stake to a validator
func (sm *StakingManager) Delegate(
	poolID string,
	delegator string,
	validator string,
	amount *big.Int,
) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	vPool, exists := sm.ValidatorPools[poolID]
	if !exists {
		return errors.New("validator pool not found")
	}

	validatorInfo, exists := vPool.Validators[validator]
	if !exists {
		return errors.New("validator not found")
	}

	if !validatorInfo.Active {
		return errors.New("validator not active")
	}

	if validatorInfo.Jailed {
		return errors.New("validator is jailed")
	}

	// Check minimum delegation
	if amount.Cmp(vPool.MinDelegation) < 0 {
		return fmt.Errorf("amount below minimum delegation: %s", vPool.MinDelegation)
	}

	// Update validator's total stake
	validatorInfo.TotalStake.Add(validatorInfo.TotalStake, amount)

	// Track delegation
	if validatorInfo.Delegators[delegator] == nil {
		validatorInfo.Delegators[delegator] = big.NewInt(0)
	}
	validatorInfo.Delegators[delegator].Add(validatorInfo.Delegators[delegator], amount)

	// Stake on behalf of delegator (use internal method since we already hold the lock)
	position, err := sm.stakeInternal(poolID, delegator, amount)
	if err != nil {
		return err
	}

	// Track delegation in position
	if position.Delegations[validator] == nil {
		position.Delegations[validator] = big.NewInt(0)
	}
	position.Delegations[validator].Add(position.Delegations[validator], amount)

	// Update metrics
	vPool.ValidatorMetrics[validator].DelegatorsCount++

	return nil
}

// SlashValidator slashes a validator for misbehavior
func (sm *StakingManager) SlashValidator(
	poolID string,
	validator string,
	reason string,
	evidence []byte,
) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	vPool, exists := sm.ValidatorPools[poolID]
	if !exists {
		return errors.New("validator pool not found")
	}

	validatorInfo, exists := vPool.Validators[validator]
	if !exists {
		return errors.New("validator not found")
	}

	// Find matching slashing condition
	var slashRate float64
	for _, condition := range vPool.SlashingConditions {
		if condition.Type == reason {
			slashRate = condition.SlashAmount
			break
		}
	}

	if slashRate == 0 {
		return errors.New("invalid slashing reason")
	}

	// Calculate slash amount
	slashAmount := new(big.Float).Mul(
		new(big.Float).SetInt(validatorInfo.TotalStake),
		big.NewFloat(slashRate),
	)
	slashAmountInt, _ := slashAmount.Int(nil)

	// Apply slash to validator's stake
	validatorInfo.TotalStake.Sub(validatorInfo.TotalStake, slashAmountInt)
	validatorInfo.SelfStake.Sub(validatorInfo.SelfStake,
		new(big.Int).Div(slashAmountInt, big.NewInt(2))) // Half from self-stake

	// Jail validator
	validatorInfo.Jailed = true
	validatorInfo.JailedUntil = time.Now().Add(7 * 24 * time.Hour) // 7 days jail

	// Update metrics
	vPool.ValidatorMetrics[validator].SlashingEvents++

	// Distribute slashed amount to other validators
	sm.distributeSlashedAmount(vPool, slashAmountInt, validator)

	return nil
}

// Helper methods

func (sm *StakingManager) calculateRewardRate(apy float64, minStake *big.Int) *big.Int {
	// Calculate reward rate per second based on APY
	// rewardRate = (APY * minStake) / (365 * 24 * 3600)
	yearlyReward := new(big.Float).Mul(
		new(big.Float).SetInt(minStake),
		big.NewFloat(apy/100),
	)
	secondsPerYear := big.NewFloat(365 * 24 * 3600)
	rewardRate := new(big.Float).Quo(yearlyReward, secondsPerYear)
	rewardRateInt, _ := rewardRate.Int(nil)
	return rewardRateInt
}

func (sm *StakingManager) getTotalShares(pool *StakingPool) *big.Int {
	total := big.NewInt(0)
	for _, position := range pool.ActiveStakers {
		total.Add(total, position.Shares)
	}
	return total
}

func (sm *StakingManager) calculateRewards(pool *StakingPool, position *StakePosition) *big.Int {
	// Calculate rewards based on time staked and APY
	timeSinceLastClaim := time.Since(position.LastClaimTime)
	seconds := big.NewFloat(timeSinceLastClaim.Seconds())

	// rewards = (shares / totalShares) * rewardRate * seconds
	totalShares := sm.getTotalShares(pool)
	if totalShares.Sign() == 0 {
		return big.NewInt(0)
	}

	schedule := sm.RewardDistributor.RewardSchedule[pool.ID]
	if schedule == nil {
		return big.NewInt(0)
	}

	shareRatio := new(big.Float).Quo(
		new(big.Float).SetInt(position.Shares),
		new(big.Float).SetInt(totalShares),
	)

	rewardFloat := new(big.Float).Mul(shareRatio, new(big.Float).SetInt(schedule.RewardRate))
	rewardFloat.Mul(rewardFloat, seconds)

	rewards, _ := rewardFloat.Int(nil)
	return rewards
}

func (sm *StakingManager) updateVotingPower(user string, position *StakePosition) {
	if position == nil || position.Amount.Sign() == 0 {
		delete(sm.GovernanceVoting, user)
		return
	}

	// Calculate voting power with lock multiplier
	lockDuration := time.Until(position.LockedUntil)
	multiplier := 1.0 + (lockDuration.Hours() / (365 * 24)) // Up to 2x for 1 year lock

	votingPower := new(big.Float).Mul(
		new(big.Float).SetInt(position.Amount),
		big.NewFloat(multiplier),
	)
	votingPowerInt, _ := votingPower.Int(nil)

	sm.GovernanceVoting[user] = &VotingPower{
		Address:      user,
		StakedAmount: position.Amount,
		VotingPower:  votingPowerInt,
		Multiplier:   multiplier,
	}
}

func (sm *StakingManager) distributeSlashedAmount(
	pool *ValidatorPool,
	amount *big.Int,
	excludeValidator string,
) {
	// Distribute slashed amount to other validators proportionally
	totalStake := big.NewInt(0)
	for addr, v := range pool.Validators {
		if addr != excludeValidator && v.Active && !v.Jailed {
			totalStake.Add(totalStake, v.TotalStake)
		}
	}

	if totalStake.Sign() == 0 {
		return
	}

	for addr, v := range pool.Validators {
		if addr != excludeValidator && v.Active && !v.Jailed {
			// reward = (validatorStake / totalStake) * slashedAmount
			reward := new(big.Int).Mul(v.TotalStake, amount)
			reward.Div(reward, totalStake)

			// Add to validator's rewards
			pool.ValidatorMetrics[addr].RewardsEarned.Add(
				pool.ValidatorMetrics[addr].RewardsEarned,
				reward,
			)
		}
	}
}

// GetStakingInfo returns staking information for a user
func (sm *StakingManager) GetStakingInfo(poolID string, user string) (map[string]interface{}, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	pool, exists := sm.Pools[poolID]
	if !exists {
		return nil, errors.New("pool not found")
	}

	position := pool.ActiveStakers[user]
	if position == nil {
		return map[string]interface{}{
			"staked":          "0",
			"shares":          "0",
			"rewards_earned":  "0",
			"rewards_claimed": "0",
		}, nil
	}

	// Calculate pending rewards
	pendingRewards := sm.calculateRewards(pool, position)

	return map[string]interface{}{
		"staked":          position.Amount.String(),
		"shares":          position.Shares.String(),
		"rewards_earned":  position.RewardsEarned.String(),
		"rewards_claimed": position.RewardsClaimed.String(),
		"pending_rewards": pendingRewards.String(),
		"locked_until":    position.LockedUntil.Format(time.RFC3339),
		"staked_at":       position.StakedAt.Format(time.RFC3339),
		"delegations":     position.Delegations,
	}, nil
}

// StartRewardDistribution starts automatic reward distribution
func (sm *StakingManager) StartRewardDistribution(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				sm.distributeRewards()
			}
		}
	}()
}

// distributeRewards distributes rewards to all pools
func (sm *StakingManager) distributeRewards() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	for poolID, pool := range sm.Pools {
		if !pool.Active || pool.Paused {
			continue
		}

		// Check if it's time to distribute
		if time.Since(pool.LastRewardTime) < pool.RewardInterval {
			continue
		}

		// Distribute rewards to all stakers
		for user, position := range pool.ActiveStakers {
			rewards := sm.calculateRewards(pool, position)
			position.RewardsEarned.Add(position.RewardsEarned, rewards)

			// Auto-compound if enabled
			if position.CompoundingEnabled {
				sm.Stake(poolID, user, rewards)
			}
		}

		pool.LastRewardTime = time.Now()
	}
}
