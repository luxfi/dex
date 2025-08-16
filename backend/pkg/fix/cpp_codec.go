//go:build cgo
// +build cgo

package fix

/*
#cgo CXXFLAGS: -std=c++17 -O3
#cgo LDFLAGS: -lstdc++
#include "../../bridge/fix_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// CppParser wraps the C++ FIX parser
type CppParser struct {
	handle C.FixParserHandle
	mu     sync.Mutex
}

// NewCppParser creates a new C++ FIX parser
func NewCppParser(maxMessageSize int) *CppParser {
	return &CppParser{
		handle: C.fix_parser_create(C.int(maxMessageSize)),
	}
}

// Parse parses FIX message data
func (p *CppParser) Parse(data []byte) (*Message, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	var msgHandle C.FixMessageHandle
	
	result := C.fix_parser_parse(
		p.handle,
		(*C.char)(unsafe.Pointer(&data[0])),
		C.int(len(data)),
		&msgHandle,
	)
	
	if !result {
		return nil, fmt.Errorf("failed to parse FIX message")
	}
	
	return &Message{handle: msgHandle}, nil
}

// Destroy cleans up the parser
func (p *CppParser) Destroy() {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if p.handle != nil {
		C.fix_parser_destroy(p.handle)
		p.handle = nil
	}
}

// Message wraps a C++ FIX message
type Message struct {
	handle C.FixMessageHandle
}

// GetField returns a field value as string
func (m *Message) GetField(tag uint32) string {
	cStr := C.fix_message_get_field(m.handle, C.uint32_t(tag))
	if cStr == nil {
		return ""
	}
	return C.GoString(cStr)
}

// GetIntField returns a field value as int
func (m *Message) GetIntField(tag uint32) int {
	return int(C.fix_message_get_int_field(m.handle, C.uint32_t(tag)))
}

// GetDoubleField returns a field value as float64
func (m *Message) GetDoubleField(tag uint32) float64 {
	return float64(C.fix_message_get_double_field(m.handle, C.uint32_t(tag)))
}

// HasField checks if a field exists
func (m *Message) HasField(tag uint32) bool {
	return bool(C.fix_message_has_field(m.handle, C.uint32_t(tag)))
}

// GetAllFields returns all fields in the message
func (m *Message) GetAllFields() map[uint32]string {
	fields := make([]C.FixField, 256)
	count := int(C.fix_message_get_all_fields(
		m.handle,
		(*C.FixField)(unsafe.Pointer(&fields[0])),
		256,
	))
	
	result := make(map[uint32]string)
	for i := 0; i < count; i++ {
		tag := uint32(fields[i].tag)
		value := C.GoString(&fields[i].value[0])
		result[tag] = value
	}
	
	return result
}

// CppBuilder wraps the C++ FIX message builder
type CppBuilder struct {
	handle C.FixBuilderHandle
	mu     sync.Mutex
}

// NewCppBuilder creates a new C++ FIX builder
func NewCppBuilder() *CppBuilder {
	return &CppBuilder{
		handle: C.fix_builder_create(),
	}
}

// Reset clears the builder
func (b *CppBuilder) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	C.fix_builder_reset(b.handle)
}

// AddField adds a string field
func (b *CppBuilder) AddField(tag uint32, value string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	cValue := C.CString(value)
	defer C.free(unsafe.Pointer(cValue))
	
	C.fix_builder_add_field(b.handle, C.uint32_t(tag), cValue)
}

// AddIntField adds an integer field
func (b *CppBuilder) AddIntField(tag uint32, value int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	C.fix_builder_add_int_field(b.handle, C.uint32_t(tag), C.int(value))
}

// AddDoubleField adds a double field
func (b *CppBuilder) AddDoubleField(tag uint32, value float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	C.fix_builder_add_double_field(b.handle, C.uint32_t(tag), C.double(value))
}

// Build generates the FIX message
func (b *CppBuilder) Build() ([]byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	buffer := make([]byte, 4096)
	size := int(C.fix_builder_build(
		b.handle,
		(*C.char)(unsafe.Pointer(&buffer[0])),
		C.int(len(buffer)),
	))
	
	if size < 0 {
		return nil, fmt.Errorf("buffer too small for FIX message")
	}
	
	return buffer[:size], nil
}

// Destroy cleans up the builder
func (b *CppBuilder) Destroy() {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	if b.handle != nil {
		C.fix_builder_destroy(b.handle)
		b.handle = nil
	}
}