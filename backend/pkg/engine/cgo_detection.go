//go:build cgo
// +build cgo

package engine

// isUsingCGO returns true when built with CGO_ENABLED=1
func isUsingCGO() bool {
	return true
}