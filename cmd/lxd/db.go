package main

import (
	"encoding/binary"
	"sync"
)

// SimpleDB is a basic key-value database interface
type SimpleDB interface {
	Put(key, value []byte) error
	Get(key []byte) ([]byte, error)
	Delete(key []byte) error
	Close() error
}

// MemDB is an in-memory database implementation
type MemDB struct {
	mu   sync.RWMutex
	data map[string][]byte
}

func NewMemDB() *MemDB {
	return &MemDB{
		data: make(map[string][]byte),
	}
}

func (db *MemDB) Put(key, value []byte) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.data[string(key)] = append([]byte(nil), value...)
	return nil
}

func (db *MemDB) Get(key []byte) ([]byte, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	if value, exists := db.data[string(key)]; exists {
		return append([]byte(nil), value...), nil
	}
	return nil, nil
}

func (db *MemDB) Delete(key []byte) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	delete(db.data, string(key))
	return nil
}

func (db *MemDB) Close() error {
	return nil
}

// Helper functions for encoding/decoding
func EncodeUint64(n uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, n)
	return b
}

func DecodeUint64(b []byte) uint64 {
	if len(b) < 8 {
		return 0
	}
	return binary.BigEndian.Uint64(b)
}