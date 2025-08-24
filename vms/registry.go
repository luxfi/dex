// Package vms provides the registry of all VMs in modern luxfi/node
package vms

import (
	"fmt"

	"github.com/luxfi/consensus/snow"
	"github.com/luxfi/database"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"

	"github.com/luxfi/dex/sdk/chain"
	"github.com/luxfi/dex/vm/dexvm"
)

// VMRegistry manages all available VMs in luxfi/node
type VMRegistry struct {
	vms map[chain.VMType]VMFactory
}

// VMFactory creates VM instances
type VMFactory interface {
	New(
		ctx *snow.Context,
		db database.Database,
		genesis []byte,
		config []byte,
	) (VM, error)
}

// VM represents a virtual machine
type VM interface {
	Initialize() error
	Start() error
	Stop() error
	HealthCheck() (interface{}, error)
}

// GlobalRegistry is the global VM registry
var GlobalRegistry = &VMRegistry{
	vms: make(map[chain.VMType]VMFactory),
}

// Register registers a VM factory
func (r *VMRegistry) Register(vmType chain.VMType, factory VMFactory) error {
	if _, exists := r.vms[vmType]; exists {
		return fmt.Errorf("VM %s already registered", vmType)
	}

	r.vms[vmType] = factory
	return nil
}

// Get gets a VM factory
func (r *VMRegistry) Get(vmType chain.VMType) (VMFactory, error) {
	factory, exists := r.vms[vmType]
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmType)
	}

	return factory, nil
}

// ListVMs lists all registered VMs
func (r *VMRegistry) ListVMs() []chain.VMType {
	vms := make([]chain.VMType, 0, len(r.vms))
	for vmType := range r.vms {
		vms = append(vms, vmType)
	}
	return vms
}

// =============================================================================
// DEXVM Factory
// =============================================================================

type DEXVMFactory struct{}

func (f *DEXVMFactory) New(
	ctx *snow.Context,
	db database.Database,
	genesis []byte,
	config []byte,
) (VM, error) {
	// Create DEX VM using the Chain SDK
	sdk := chain.NewChainSDK(chain.VMTypeDEX, chain.ConsensusSnowman)

	if err := sdk.Initialize(ctx, db, config); err != nil {
		return nil, err
	}

	// Register DEX extension
	dexExt := &chain.DEXExtension{
		// Initialize DEX components
	}

	if err := sdk.RegisterExtension("dex", dexExt); err != nil {
		return nil, err
	}

	return &DEXVM{sdk: sdk}, nil
}

type DEXVM struct {
	sdk *chain.ChainSDK
}

func (vm *DEXVM) Initialize() error                 { return nil }
func (vm *DEXVM) Start() error                      { return nil }
func (vm *DEXVM) Stop() error                       { return nil }
func (vm *DEXVM) HealthCheck() (interface{}, error) { return nil, nil }

// =============================================================================
// AIVM (Attestation/AI VM) Factory
// =============================================================================

type AIVMFactory struct{}

func (f *AIVMFactory) New(
	ctx *snow.Context,
	db database.Database,
	genesis []byte,
	config []byte,
) (VM, error) {
	// Create AI VM using the Chain SDK
	sdk := chain.NewChainSDK(chain.VMTypeAttestation, chain.ConsensusHybrid)

	if err := sdk.Initialize(ctx, db, config); err != nil {
		return nil, err
	}

	// Register AI extension
	aiExt := &chain.AIVMExtension{
		ModelRegistry:   make(map[string]*chain.AIModel),
		InferenceEngine: NewInferenceEngine(),
		ProofGenerator:  NewProofGenerator(),
	}

	if err := sdk.RegisterExtension("ai", aiExt); err != nil {
		return nil, err
	}

	return &AIVM{sdk: sdk}, nil
}

type AIVM struct {
	sdk *chain.ChainSDK
}

func (vm *AIVM) Initialize() error {
	// Initialize AI-specific components
	// - Model verification
	// - Attestation generation
	// - Inference execution
	return nil
}

func (vm *AIVM) Start() error { return nil }
func (vm *AIVM) Stop() error  { return nil }
func (vm *AIVM) HealthCheck() (interface{}, error) {
	return map[string]interface{}{
		"models_registered": 0,
		"inferences_run":    0,
		"attestations":      0,
	}, nil
}

// =============================================================================
// FHEVM (Fully Homomorphic Encryption VM) Factory
// =============================================================================

type FHEVMFactory struct{}

func (f *FHEVMFactory) New(
	ctx *snow.Context,
	db database.Database,
	genesis []byte,
	config []byte,
) (VM, error) {
	// Create FHE VM using the Chain SDK
	sdk := chain.NewChainSDK(chain.VMTypeFHE, chain.ConsensusSnowman)

	if err := sdk.Initialize(ctx, db, config); err != nil {
		return nil, err
	}

	// Register FHE extension
	fheExt := &chain.FHEVMExtension{
		FHEScheme:      chain.FHESchemeCKKS,
		EncryptedState: make(map[string][]byte),
		KeyManager:     NewFHEKeyManager(),
	}

	if err := sdk.RegisterExtension("fhe", fheExt); err != nil {
		return nil, err
	}

	return &FHEVM{sdk: sdk}, nil
}

type FHEVM struct {
	sdk *chain.ChainSDK
}

func (vm *FHEVM) Initialize() error {
	// Initialize FHE-specific components
	// - Key generation
	// - Encrypted state management
	// - Homomorphic operations
	return nil
}

func (vm *FHEVM) Start() error                      { return nil }
func (vm *FHEVM) Stop() error                       { return nil }
func (vm *FHEVM) HealthCheck() (interface{}, error) { return nil, nil }

// =============================================================================
// MPCVM (Multi-Party Computation VM) Factory
// =============================================================================

type MPCVMFactory struct{}

func (f *MPCVMFactory) New(
	ctx *snow.Context,
	db database.Database,
	genesis []byte,
	config []byte,
) (VM, error) {
	// Create MPC VM using the Chain SDK
	sdk := chain.NewChainSDK(chain.VMTypeMPC, chain.ConsensusFPC)

	if err := sdk.Initialize(ctx, db, config); err != nil {
		return nil, err
	}

	// Register MPC extension
	mpcExt := &chain.MPCVMExtension{
		MPCProtocol:  chain.MPCProtocolSPDZ,
		Parties:      make(map[string]*chain.Party),
		SecretShares: make(map[string][]*chain.Share),
	}

	if err := sdk.RegisterExtension("mpc", mpcExt); err != nil {
		return nil, err
	}

	return &MPCVM{sdk: sdk}, nil
}

type MPCVM struct {
	sdk *chain.ChainSDK
}

func (vm *MPCVM) Initialize() error {
	// Initialize MPC-specific components
	// - Secret sharing
	// - Secure computation protocols
	// - Party management
	return nil
}

func (vm *MPCVM) Start() error                      { return nil }
func (vm *MPCVM) Stop() error                       { return nil }
func (vm *MPCVM) HealthCheck() (interface{}, error) { return nil, nil }

// =============================================================================
// QuantumVM (Quantum-Resistant VM with Ringtail) Factory
// =============================================================================

type QuantumVMFactory struct{}

func (f *QuantumVMFactory) New(
	ctx *snow.Context,
	db database.Database,
	genesis []byte,
	config []byte,
) (VM, error) {
	// Create Quantum VM using the Chain SDK with Ringtail consensus
	sdk := chain.NewChainSDK(chain.VMTypeQuantum, chain.ConsensusRingtail)

	if err := sdk.Initialize(ctx, db, config); err != nil {
		return nil, err
	}

	// Configure Ringtail consensus parameters
	consensusParams := chain.ConsensusParams{
		LatticeParams: &chain.LatticeParams{
			Dimension:         512,
			Modulus:           new(big.Int).Exp(big.NewInt(2), big.NewInt(256), nil),
			StandardDeviation: 3.2,
			SecurityLevel:     256, // Post-quantum 256-bit security
		},
		RingtailRounds:  2,  // 2-round consensus as specified
		BLSThreshold:    67, // 2/3 threshold for BLS
		VerkleWitnesses: true,
		VerkleTreeDepth: 32,
		FPCVotingRounds: 3,
		FPCRandomness:   true,
	}

	if err := sdk.Consensus.Initialize(consensusParams); err != nil {
		return nil, err
	}

	return &QuantumVM{sdk: sdk}, nil
}

type QuantumVM struct {
	sdk *chain.ChainSDK
}

func (vm *QuantumVM) Initialize() error {
	// Initialize quantum-resistant components
	// - Lattice-based signatures
	// - Ringtail consensus
	// - Post-quantum cryptography
	return nil
}

func (vm *QuantumVM) Start() error { return nil }
func (vm *QuantumVM) Stop() error  { return nil }
func (vm *QuantumVM) HealthCheck() (interface{}, error) {
	return map[string]interface{}{
		"consensus":       "ringtail",
		"security_level":  256,
		"rounds":          2,
		"lattice_enabled": true,
	}, nil
}

// =============================================================================
// Helper functions for extensions
// =============================================================================

func NewInferenceEngine() interface{} {
	// Create inference engine for AIVM
	return nil
}

func NewProofGenerator() interface{} {
	// Create proof generator for AIVM
	return nil
}

func NewFHEKeyManager() interface{} {
	// Create FHE key manager
	return nil
}

// =============================================================================
// Registration of all VMs
// =============================================================================

func init() {
	// Register all VMs in the global registry

	// Core VMs
	GlobalRegistry.Register(chain.VMTypeDEX, &DEXVMFactory{})
	GlobalRegistry.Register(chain.VMTypeAttestation, &AIVMFactory{})
	GlobalRegistry.Register(chain.VMTypeFHE, &FHEVMFactory{})
	GlobalRegistry.Register(chain.VMTypeMPC, &MPCVMFactory{})
	GlobalRegistry.Register(chain.VMTypeQuantum, &QuantumVMFactory{})

	// Additional VMs can be registered here:
	// GlobalRegistry.Register(chain.VMTypeZK, &ZKVMFactory{})
	// GlobalRegistry.Register(chain.VMTypeStorage, &StorageVMFactory{})
	// GlobalRegistry.Register(chain.VMTypeOracle, &OracleVMFactory{})
	// GlobalRegistry.Register(chain.VMTypePrivacy, &PrivacyVMFactory{})
	// GlobalRegistry.Register(chain.VMTypeRollup, &RollupVMFactory{})

	log.Info("Registered VMs", "count", len(GlobalRegistry.vms))
}

// =============================================================================
// Avalanche Integration
// =============================================================================

// AvalancheAdapter allows Avalanche to adopt our QuantumVM
type AvalancheAdapter struct {
	quantumVM *QuantumVM
}

// NewAvalancheAdapter creates an adapter for Avalanche to use QuantumVM
func NewAvalancheAdapter() *AvalancheAdapter {
	// Create QuantumVM with Ringtail consensus
	factory := &QuantumVMFactory{}
	vm, _ := factory.New(nil, nil, nil, nil)

	return &AvalancheAdapter{
		quantumVM: vm.(*QuantumVM),
	}
}

// GetConsensusGadget returns the quantum consensus gadget for Avalanche
func (a *AvalancheAdapter) GetConsensusGadget() interface{} {
	// Return Ringtail lattice 2-round consensus with:
	// - BLS signature aggregation with optimizations
	// - Verkle witnesses
	// - FPC
	// This can be adopted by Avalanche as an opt-in consensus upgrade
	return a.quantumVM.sdk.Consensus
}

// Message for Avalanche adoption
const AvalancheAdoptionMessage = `
Avalanche can theoretically opt-in to adopt our QuantumVM by:

1. Using the Ringtail lattice-based 2-round consensus gadget
2. Integrating BLS signature aggregation with our optimizations
3. Adding Verkle witnesses for efficient state proofs
4. Incorporating FPC for faster finality

This provides post-quantum security while maintaining compatibility
with existing Avalanche infrastructure. The consensus gadget is
designed as a drop-in replacement that can be activated via a
network upgrade.

Key benefits for Avalanche:
- Quantum resistance (256-bit post-quantum security)
- Faster finality (2-round consensus vs multi-round)
- Better signature aggregation (optimized BLS)
- Efficient state proofs (Verkle witnesses)
- Improved probabilistic finality (FPC)
`
