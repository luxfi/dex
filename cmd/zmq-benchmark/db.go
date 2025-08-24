package main

import (
	"context"
	"sync"

	"github.com/luxfi/database"
)

// simpleMemDB is a simple in-memory database for benchmarking
type simpleMemDB struct {
	mu   sync.RWMutex
	data map[string][]byte
}

func newSimpleMemDB() database.Database {
	return &simpleMemDB{
		data: make(map[string][]byte),
	}
}

func (m *simpleMemDB) Get(key []byte) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	val, ok := m.data[string(key)]
	if !ok {
		return nil, database.ErrNotFound
	}
	return val, nil
}

func (m *simpleMemDB) Put(key []byte, value []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data[string(key)] = value
	return nil
}

func (m *simpleMemDB) Delete(key []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.data, string(key))
	return nil
}

func (m *simpleMemDB) Close() error {
	return nil
}

func (m *simpleMemDB) Has(key []byte) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.data[string(key)]
	return ok, nil
}

func (m *simpleMemDB) Compact(start []byte, limit []byte) error {
	return nil
}

func (m *simpleMemDB) NewBatch() database.Batch {
	return &simpleBatch{db: m, ops: make([]batchOp, 0)}
}

func (m *simpleMemDB) NewIterator() database.Iterator {
	return nil
}

func (m *simpleMemDB) NewIteratorWithStart(start []byte) database.Iterator {
	return nil
}

func (m *simpleMemDB) NewIteratorWithPrefix(prefix []byte) database.Iterator {
	return nil
}

func (m *simpleMemDB) NewIteratorWithStartAndPrefix(start, prefix []byte) database.Iterator {
	return nil
}

func (m *simpleMemDB) HealthCheck(ctx context.Context) (interface{}, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return map[string]interface{}{
		"type": "memDB",
		"size": len(m.data),
	}, nil
}

// simpleBatch implements database.Batch
type simpleBatch struct {
	db  *simpleMemDB
	ops []batchOp
}

type batchOp struct {
	delete bool
	key    []byte
	value  []byte
}

func (b *simpleBatch) Put(key, value []byte) error {
	b.ops = append(b.ops, batchOp{delete: false, key: key, value: value})
	return nil
}

func (b *simpleBatch) Delete(key []byte) error {
	b.ops = append(b.ops, batchOp{delete: true, key: key})
	return nil
}

func (b *simpleBatch) ValueSize() int {
	size := 0
	for _, op := range b.ops {
		size += len(op.value)
	}
	return size
}

func (b *simpleBatch) Size() int {
	size := 0
	for _, op := range b.ops {
		size += len(op.key) + len(op.value)
	}
	return size
}

func (b *simpleBatch) Write() error {
	b.db.mu.Lock()
	defer b.db.mu.Unlock()
	for _, op := range b.ops {
		if op.delete {
			delete(b.db.data, string(op.key))
		} else {
			b.db.data[string(op.key)] = op.value
		}
	}
	return nil
}

func (b *simpleBatch) Reset() {
	b.ops = b.ops[:0]
}

func (b *simpleBatch) Replay(w database.KeyValueWriterDeleter) error {
	for _, op := range b.ops {
		if op.delete {
			if err := w.Delete(op.key); err != nil {
				return err
			}
		} else {
			if err := w.Put(op.key, op.value); err != nil {
				return err
			}
		}
	}
	return nil
}

func (b *simpleBatch) Inner() database.Batch {
	return b
}
