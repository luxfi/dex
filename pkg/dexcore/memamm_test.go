// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexcore

import "encoding/binary"

// memamm_test.go is an in-Store implementation of AMMPool for the proof suite — the
// exact analog of the precompile's real V2/V3 pool view, but with reserves stored as
// a dexcore Store row instead of cEVM contract storage. dexcore's constant-product
// math + routing run identically over either backend; the test proves the math and
// the journal, the precompile proves the real-pool reserve read. Reserves live
// ENTIRELY in the Store (no package-global state), so two concurrent tests over two
// stores never interfere.

const prefixAMMReserve = "ammtest:"

func ammReserveKey(poolID [32]byte) []byte {
	k := make([]byte, len(prefixAMMReserve)+32)
	copy(k, prefixAMMReserve)
	copy(k[len(prefixAMMReserve):], poolID[:])
	return k
}

// memAMM is a test AMM pool view. It holds a one-shot pending seed (the initial
// reserves) materialized into the Store on the first Reserves read, so reserves live
// ENTIRELY in the Store thereafter — no package-global, no per-pool db handle, no
// cross-test interference.
type memAMM struct {
	feeBps uint32

	// pending one-shot seed, applied on first Reserves(db) call.
	seeded              bool
	seedPID             [32]byte
	seedBase, seedQuote uint64
}

func newMemAMM(feeBps uint32) *memAMM { return &memAMM{feeBps: feeBps} }

// set stashes the initial reserves; they are written to the Store on the first
// Reserves read (which is where a real db handle is available). Keeps the terse
// pool.set(pid, base, quote) call site while keeping reserves Store-resident.
func (m *memAMM) set(pid [32]byte, base, quote uint64) {
	m.seedPID, m.seedBase, m.seedQuote, m.seeded = pid, base, quote, true
}

func (m *memAMM) materialize(db Store) error {
	if !m.seeded {
		return nil
	}
	m.seeded = false
	var v [16]byte
	binary.BigEndian.PutUint64(v[0:8], m.seedBase)
	binary.BigEndian.PutUint64(v[8:16], m.seedQuote)
	return db.Put(ammReserveKey(m.seedPID), v[:])
}

func (m *memAMM) Reserves(db Store, poolID [32]byte) (uint64, uint64, bool, error) {
	if err := m.materialize(db); err != nil {
		return 0, 0, false, err
	}
	v, err := db.Get(ammReserveKey(poolID))
	if err != nil || len(v) != 16 {
		return 0, 0, false, nil
	}
	return binary.BigEndian.Uint64(v[0:8]), binary.BigEndian.Uint64(v[8:16]), true, nil
}

func (m *memAMM) SetReserves(db Store, poolID [32]byte, base, quote uint64) error {
	var v [16]byte
	binary.BigEndian.PutUint64(v[0:8], base)
	binary.BigEndian.PutUint64(v[8:16], quote)
	return db.Put(ammReserveKey(poolID), v[:])
}

func (m *memAMM) FeeBps() uint32 { return m.feeBps }
