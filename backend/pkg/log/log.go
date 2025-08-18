package log

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// Logger interface for structured logging
type Logger interface {
	Info(msg string, args ...interface{})
	Error(msg string, args ...interface{})
	Warn(msg string, args ...interface{})
	Debug(msg string, args ...interface{})
	Fatal(msg string, args ...interface{})
	WithField(key string, value interface{}) Logger
}

// SimpleLogger is a basic implementation
type SimpleLogger struct {
	name   string
	fields map[string]interface{}
	mu     sync.RWMutex
}

// NewLogger creates a new logger
func NewLogger(name string) Logger {
	return &SimpleLogger{
		name:   name,
		fields: make(map[string]interface{}),
	}
}

// WithField adds a field to the logger
func (l *SimpleLogger) WithField(key string, value interface{}) Logger {
	l.mu.Lock()
	defer l.mu.Unlock()

	newLogger := &SimpleLogger{
		name:   l.name,
		fields: make(map[string]interface{}),
	}

	// Copy existing fields
	for k, v := range l.fields {
		newLogger.fields[k] = v
	}
	newLogger.fields[key] = value

	return newLogger
}

// Info logs an info message
func (l *SimpleLogger) Info(msg string, args ...interface{}) {
	l.logMessage("INFO", msg, args...)
}

// Error logs an error message
func (l *SimpleLogger) Error(msg string, args ...interface{}) {
	l.logMessage("ERROR", msg, args...)
}

// Warn logs a warning message
func (l *SimpleLogger) Warn(msg string, args ...interface{}) {
	l.logMessage("WARN", msg, args...)
}

// Debug logs a debug message
func (l *SimpleLogger) Debug(msg string, args ...interface{}) {
	l.logMessage("DEBUG", msg, args...)
}

// Fatal logs a fatal message and exits
func (l *SimpleLogger) Fatal(msg string, args ...interface{}) {
	l.logMessage("FATAL", msg, args...)
	log.Fatal("Fatal error")
}

// logMessage is the internal logging function
func (l *SimpleLogger) logMessage(level string, msg string, args ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	timestamp := time.Now().Format("2006-01-02 15:04:05")

	// Build fields string
	fieldsStr := ""
	for k, v := range l.fields {
		fieldsStr += fmt.Sprintf(" %s=%v", k, v)
	}

	// Build args string
	argsStr := ""
	for i := 0; i < len(args); i += 2 {
		if i+1 < len(args) {
			argsStr += fmt.Sprintf(" %v=%v", args[i], args[i+1])
		}
	}

	log.Printf("%s [%s] %s: %s%s%s", timestamp, level, l.name, msg, fieldsStr, argsStr)
}
