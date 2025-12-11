package log

import (
	"bytes"
	"log"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewLogger(t *testing.T) {
	logger := NewLogger("test-service")
	assert.NotNil(t, logger)

	// Check it's a SimpleLogger
	simpleLogger, ok := logger.(*SimpleLogger)
	assert.True(t, ok)
	assert.Equal(t, "test-service", simpleLogger.name)
}

func TestLoggerWithField(t *testing.T) {
	logger := NewLogger("test")

	// Add a field
	newLogger := logger.WithField("user_id", "123")
	assert.NotNil(t, newLogger)

	// Original logger should not have the field
	origLogger := logger.(*SimpleLogger)
	assert.NotContains(t, origLogger.fields, "user_id")

	// New logger should have the field
	newSimple := newLogger.(*SimpleLogger)
	assert.Equal(t, "123", newSimple.fields["user_id"])
}

func TestLoggerWithMultipleFields(t *testing.T) {
	logger := NewLogger("test")

	// Chain multiple WithField calls
	logger = logger.WithField("request_id", "req-123")
	logger = logger.WithField("user_id", "user-456")
	logger = logger.WithField("action", "trade")

	simpleLogger := logger.(*SimpleLogger)
	assert.Equal(t, "req-123", simpleLogger.fields["request_id"])
	assert.Equal(t, "user-456", simpleLogger.fields["user_id"])
	assert.Equal(t, "trade", simpleLogger.fields["action"])
}

func TestLoggerInfo(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewLogger("test-service")
	logger.Info("test message", "key1", "value1")

	output := buf.String()
	assert.Contains(t, output, "[INFO]")
	assert.Contains(t, output, "test-service")
	assert.Contains(t, output, "test message")
	assert.Contains(t, output, "key1=value1")
}

func TestLoggerError(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewLogger("error-test")
	logger.Error("something went wrong", "error_code", 500)

	output := buf.String()
	assert.Contains(t, output, "[ERROR]")
	assert.Contains(t, output, "error-test")
	assert.Contains(t, output, "something went wrong")
	assert.Contains(t, output, "error_code=500")
}

func TestLoggerWarn(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewLogger("warn-test")
	logger.Warn("warning message", "level", "high")

	output := buf.String()
	assert.Contains(t, output, "[WARN]")
	assert.Contains(t, output, "warn-test")
	assert.Contains(t, output, "warning message")
}

func TestLoggerDebug(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewLogger("debug-test")
	logger.Debug("debug info", "variable", 42)

	output := buf.String()
	assert.Contains(t, output, "[DEBUG]")
	assert.Contains(t, output, "debug-test")
	assert.Contains(t, output, "debug info")
}

func TestLoggerWithFieldsInOutput(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewLogger("field-test").
		WithField("service", "trading").
		WithField("version", "1.0")

	logger.Info("starting service")

	output := buf.String()
	assert.Contains(t, output, "service=trading")
	assert.Contains(t, output, "version=1.0")
}

func TestLoggerConcurrency(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewLogger("concurrent-test")

	// Run multiple goroutines logging concurrently
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 100; j++ {
				logger.Info("test message", "goroutine", id, "iteration", j)
			}
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// Check output has content (no race condition crashes)
	output := buf.String()
	assert.NotEmpty(t, output)
	// Should have 1000 log entries
	assert.Equal(t, 1000, strings.Count(output, "[INFO]"))
}

func TestLoggerInterface(t *testing.T) {
	// Verify SimpleLogger implements Logger interface
	var _ Logger = &SimpleLogger{}
	var _ Logger = NewLogger("test")
}

func TestLoggerOddArgs(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewLogger("odd-args")
	// Odd number of args - should handle gracefully
	logger.Info("message", "key1", "value1", "orphan")

	output := buf.String()
	assert.Contains(t, output, "message")
	assert.Contains(t, output, "key1=value1")
	// Orphan key without value should not crash
}

func TestLoggerEmptyArgs(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewLogger("empty-args")
	logger.Info("simple message")

	output := buf.String()
	assert.Contains(t, output, "simple message")
	assert.Contains(t, output, "[INFO]")
}

func TestLoggerFieldsCopiedNotShared(t *testing.T) {
	original := NewLogger("original")
	withA := original.WithField("a", "1")
	withB := original.WithField("b", "2")

	// withA should not have "b"
	simpleA := withA.(*SimpleLogger)
	simpleB := withB.(*SimpleLogger)

	assert.Contains(t, simpleA.fields, "a")
	assert.NotContains(t, simpleA.fields, "b")

	assert.Contains(t, simpleB.fields, "b")
	assert.NotContains(t, simpleB.fields, "a")
}

func TestLoggerTimestampFormat(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewLogger("timestamp-test")
	logger.Info("test")

	output := buf.String()
	// Should contain timestamp in format: YYYY-MM-DD HH:MM:SS
	assert.Regexp(t, `\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}`, output)
}
