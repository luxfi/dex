//go:build !cgo
// +build !cgo

package engine

// isUsingCGO returns false when built without CGO
func isUsingCGO() bool {
	return false
}