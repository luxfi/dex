package vm

import (
	"errors"
	"fmt"

	"github.com/luxfi/consensus/snow"
	"github.com/luxfi/database"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"

	"github.com/luxfi/dex/vm/dexvm"
)

const (
	// DEXVMID is the ID of the DEX VM
	DEXVMID = "dexvm"
)

var (
	// ErrUnknownVMType is returned when an unknown VM type is requested
	ErrUnknownVMType = errors.New("unknown VM type")
)

// Factory creates VMs
type Factory interface {
	New(*snow.Context, database.Manager, []byte, []byte, []byte, chan<- common.Message, []*common.Fx, common.AppSender) (interface{}, error)
}

// VMFactory implements Factory for DEX VM
type VMFactory struct{}

// New creates a new VM instance
func (f *VMFactory) New(
	ctx *snow.Context,
	dbManager database.Manager,
	genesisBytes []byte,
	upgradeBytes []byte,
	configBytes []byte,
	toEngine chan<- common.Message,
	fxs []*common.Fx,
	appSender common.AppSender,
) (interface{}, error) {

	ctx.Log.Info("Creating DEX VM",
		log.String("chainID", ctx.ChainID.String()),
		log.String("vmID", DEXVMID),
	)

	// Create DEX VM instance
	vm := dexvm.New()

	// Initialize the VM
	if err := vm.Initialize(
		ctx,
		dbManager,
		genesisBytes,
		upgradeBytes,
		configBytes,
		toEngine,
		fxs,
		appSender,
	); err != nil {
		return nil, fmt.Errorf("failed to initialize DEX VM: %w", err)
	}

	return vm, nil
}

// Manager manages VM factories
type Manager struct {
	factories map[ids.ID]Factory
}

// NewManager creates a new VM manager
func NewManager() *Manager {
	return &Manager{
		factories: make(map[ids.ID]Factory),
	}
}

// RegisterFactory registers a VM factory
func (m *Manager) RegisterFactory(vmID ids.ID, factory Factory) error {
	if _, exists := m.factories[vmID]; exists {
		return fmt.Errorf("VM %s already registered", vmID)
	}

	m.factories[vmID] = factory
	return nil
}

// GetFactory returns a factory for the given VM ID
func (m *Manager) GetFactory(vmID ids.ID) (Factory, error) {
	factory, exists := m.factories[vmID]
	if !exists {
		return nil, fmt.Errorf("%w: %s", ErrUnknownVMType, vmID)
	}

	return factory, nil
}

// RegisterDEXVM registers the DEX VM factory
func RegisterDEXVM(manager *Manager) error {
	// Create DEX VM ID
	dexVMID, err := ids.FromString(DEXVMID)
	if err != nil {
		// Use a deterministic ID for DEX VM
		dexVMID = ids.GenerateIDFromString("dexvm")
	}

	// Register factory
	return manager.RegisterFactory(dexVMID, &VMFactory{})
}
