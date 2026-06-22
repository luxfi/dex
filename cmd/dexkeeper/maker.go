// maker.go derives the keeper's maker signing key from a BIP-39 mnemonic on the
// canonical Lux BIP44 path m/44'/9000'/0'/0/i — the SAME path the genesis builder
// (genesis.LoadBIP44WalletKeysFromMnemonic), `lux key derive`, MetaMask and the Lux
// wallet use, so a key derived here has the identical EVM address everywhere.
//
// WHY REPRODUCE THE WALK INSTEAD OF IMPORTING genesis: genesis/pkg/genesis pulls the
// full validator-identity graph (ML-DSA-65, BLS, luxfi/tls) to derive NODE identities.
// The keeper needs only a user SPENDING key (secp256k1 -> EVM address). This is a small,
// frozen, standard BIP44 walk over the SAME primitives (go-bip39 + go-bip32 + secp256k1
// + keccak) the genesis builder uses; maker_test pins the derived address against the
// known treasury address so a one-sided drift from the canonical path trips a red test.
// (Same "three homes" reproduction discipline dexsession uses for the 0x9999 calldata.)
package main

import (
	"crypto/ecdsa"
	"fmt"

	"github.com/luxfi/crypto"
	"github.com/luxfi/crypto/keccak256"
	"github.com/luxfi/geth/common"
	bip32 "github.com/luxfi/go-bip32"
	bip39 "github.com/luxfi/go-bip39"
)

// luxCoinType is the BIP44 coin type Lux uses by canonical convention (the 9000' in
// m/44'/9000'/0'/0/i). Every BIP44-conformant wallet using that coin type derives the
// same key set from the same mnemonic.
const luxCoinType = 9000

// makerKey is a single derived maker identity: the STANDARD EVM address (keccak256(X||Y)
// [12:], the C tx-caller / escrow-owner format) plus the ecdsa signing key the 0x9999
// C-Chain Phase-A/B swaps are geth-signed with.
//
// There is intentionally NO secp256k1.PrivateKey wrapper here: the dexvm settling
// RelayOrderTx is sent UNSIGNED (its From is the escrow owner — see keeper.go), so the
// keeper never invokes secp256k1.PrivateKey.EVMAddress(), which is NON-standard (it hashes
// the COMPRESSED pubkey, giving an address that does not equal the standard escrow owner).
type makerKey struct {
	Index   int
	Address common.Address
	// ecdsa is the standard ecdsa.PrivateKey used by geth types.SignTx for the C-Chain
	// Phase-A (intent) and Phase-B (settlement) swap transactions.
	ecdsa *ecdsa.PrivateKey
}

// deriveMakerKey scans m/44'/9000'/0'/0/i for i in [0, scan) and returns the first key
// whose EVM address equals want. scan bounds the search (the canonical funded set is the
// first 1000 wallet keys). A nil want is invalid — the caller always knows the maker
// address it is driving.
func deriveMakerKey(mnemonic string, want common.Address, scan int) (*makerKey, error) {
	if !bip39.IsMnemonicValid(mnemonic) {
		return nil, fmt.Errorf("invalid mnemonic")
	}
	change, err := bip44ChangeNode(mnemonic)
	if err != nil {
		return nil, err
	}
	for i := 0; i < scan; i++ {
		child, err := change.NewChildKey(uint32(i)) //nolint:gosec // i < scan, bounded
		if err != nil {
			return nil, fmt.Errorf("derive child %d: %w", i, err)
		}
		mk, err := makerKeyFromBytes(child.Key, i)
		if err != nil {
			return nil, fmt.Errorf("key %d: %w", i, err)
		}
		if mk.Address == want {
			return mk, nil
		}
	}
	return nil, fmt.Errorf("no key in m/44'/%d'/0'/0/0..%d derives address %s", luxCoinType, scan, want)
}

// bip44ChangeNode walks the hardened prefix m/44'/9000'/0'/0 once; the caller then
// derives the non-hardened account index i off the returned change node. This is the
// EXACT prefix genesis.LoadBIP44WalletKeysFromMnemonic walks.
func bip44ChangeNode(mnemonic string) (*bip32.Key, error) {
	seed := bip39.NewSeed(mnemonic, "")
	master, err := bip32.NewMasterKey(seed)
	if err != nil {
		return nil, fmt.Errorf("master key: %w", err)
	}
	purpose, err := master.NewChildKey(bip32.FirstHardenedChild + 44)
	if err != nil {
		return nil, fmt.Errorf("derive purpose 44': %w", err)
	}
	coin, err := purpose.NewChildKey(bip32.FirstHardenedChild + luxCoinType)
	if err != nil {
		return nil, fmt.Errorf("derive coin %d': %w", luxCoinType, err)
	}
	account, err := coin.NewChildKey(bip32.FirstHardenedChild + 0)
	if err != nil {
		return nil, fmt.Errorf("derive account 0': %w", err)
	}
	change, err := account.NewChildKey(0) // non-hardened external chain
	if err != nil {
		return nil, fmt.Errorf("derive change 0: %w", err)
	}
	return change, nil
}

// makerKeyFromBytes builds the ecdsa key + standard EVM address from raw 32-byte private
// key material. The EVM address is keccak256(pubX||pubY)[12:] — identical to
// genesis.keccakAddr and to geth crypto.PubkeyToAddress (the escrow-owner format).
func makerKeyFromBytes(raw []byte, index int) (*makerKey, error) {
	ec, err := crypto.ToECDSA(raw)
	if err != nil {
		return nil, fmt.Errorf("ecdsa: %w", err)
	}
	return &makerKey{
		Index:   index,
		Address: evmAddress(ec),
		ecdsa:   ec,
	}, nil
}

// evmAddress derives the 20-byte EVM address from an ecdsa public key as
// keccak256(X||Y)[12:]. Matches genesis.keccakAddr byte-for-byte (the funded-address
// contract) — both fill X and Y big-endian into a 64-byte buffer before hashing.
func evmAddress(key *ecdsa.PrivateKey) common.Address {
	buf := make([]byte, 64)
	key.PublicKey.X.FillBytes(buf[:32])
	key.PublicKey.Y.FillBytes(buf[32:])
	h := keccak256.Sum(buf)
	var a common.Address
	copy(a[:], h[12:])
	return a
}
