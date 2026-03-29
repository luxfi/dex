// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package gateway

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// AdminRole is the required role in X-User-Role header for admin operations.
const AdminRole = "admin"

// pauseStateGlobal is the package-level pause state singleton.
// Thread-safe: protected by internal RWMutex.
var pauseStateGlobal = &PauseState{
	poolPaused: make(map[string]*PoolPauseInfo),
	poolFrozen: make(map[string]*PoolFreezeInfo),
}

// GetPauseState returns the global pause state for external access.
func GetPauseState() *PauseState { return pauseStateGlobal }

// PauseState tracks the current pause/freeze state for the gateway layer.
type PauseState struct {
	mu          sync.RWMutex
	dexPaused   bool
	dexPausedAt time.Time
	dexPausedBy string
	poolPaused  map[string]*PoolPauseInfo
	poolFrozen  map[string]*PoolFreezeInfo
}

// PoolPauseInfo tracks when and by whom a pool was paused.
type PoolPauseInfo struct {
	PoolID   string    `json:"poolId"`
	PausedAt time.Time `json:"pausedAt"`
	PausedBy string    `json:"pausedBy"`
}

// PoolFreezeInfo tracks when and by whom a pool was permanently frozen.
type PoolFreezeInfo struct {
	PoolID   string    `json:"poolId"`
	FrozenAt time.Time `json:"frozenAt"`
	FrozenBy string    `json:"frozenBy"`
	Reason   string    `json:"reason,omitempty"`
}

// PoolStatusResponse is the response for GET /v1/admin/pool/status.
type PoolStatusResponse struct {
	PoolID     string          `json:"poolId"`
	DEXPaused  bool            `json:"dexPaused"`
	Paused     bool            `json:"paused"`
	Frozen     bool            `json:"frozen"`
	PauseInfo  *PoolPauseInfo  `json:"pauseInfo,omitempty"`
	FreezeInfo *PoolFreezeInfo `json:"freezeInfo,omitempty"`
}

// DEXStatusResponse is the response for GET /v1/admin/status.
type DEXStatusResponse struct {
	DEXPaused       bool              `json:"dexPaused"`
	DEXPausedAt     *time.Time        `json:"dexPausedAt,omitempty"`
	DEXPausedBy     string            `json:"dexPausedBy,omitempty"`
	PausedPoolCount int               `json:"pausedPoolCount"`
	FrozenPoolCount int               `json:"frozenPoolCount"`
	PausedPools     []*PoolPauseInfo  `json:"pausedPools,omitempty"`
	FrozenPools     []*PoolFreezeInfo `json:"frozenPools,omitempty"`
}

// IsDEXPaused returns whether the entire DEX is paused.
func (ps *PauseState) IsDEXPaused() bool {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	return ps.dexPaused
}

// IsPoolPaused returns whether a specific pool is paused.
func (ps *PauseState) IsPoolPaused(poolId string) bool {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	_, ok := ps.poolPaused[poolId]
	return ok
}

// IsPoolFrozen returns whether a specific pool is permanently frozen.
func (ps *PauseState) IsPoolFrozen(poolId string) bool {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	_, ok := ps.poolFrozen[poolId]
	return ok
}

// IsOperationBlocked returns an error if operations on the given pool are blocked.
func (ps *PauseState) IsOperationBlocked(poolId string) error {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	if ps.dexPaused {
		return fmt.Errorf("DEX: paused")
	}
	if _, ok := ps.poolFrozen[poolId]; ok {
		return fmt.Errorf("pool: frozen")
	}
	if _, ok := ps.poolPaused[poolId]; ok {
		return fmt.Errorf("pool: paused")
	}
	return nil
}

// PauseDEX pauses the entire DEX.
func (ps *PauseState) PauseDEX(admin string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if ps.dexPaused {
		return fmt.Errorf("DEX already paused")
	}
	ps.dexPaused = true
	ps.dexPausedAt = time.Now()
	ps.dexPausedBy = admin
	return nil
}

// ResumeDEX resumes the entire DEX.
func (ps *PauseState) ResumeDEX(admin string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if !ps.dexPaused {
		return fmt.Errorf("DEX not paused")
	}
	ps.dexPaused = false
	ps.dexPausedAt = time.Time{}
	ps.dexPausedBy = ""
	return nil
}

// PausePool pauses a specific pool.
func (ps *PauseState) PausePool(poolId, admin string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if _, ok := ps.poolFrozen[poolId]; ok {
		return fmt.Errorf("pool is frozen, cannot pause")
	}
	if _, ok := ps.poolPaused[poolId]; ok {
		return fmt.Errorf("pool already paused")
	}
	ps.poolPaused[poolId] = &PoolPauseInfo{
		PoolID: poolId, PausedAt: time.Now(), PausedBy: admin,
	}
	return nil
}

// ResumePool resumes a specific pool.
func (ps *PauseState) ResumePool(poolId, admin string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if _, ok := ps.poolFrozen[poolId]; ok {
		return fmt.Errorf("pool is frozen, cannot resume")
	}
	if _, ok := ps.poolPaused[poolId]; !ok {
		return fmt.Errorf("pool not paused")
	}
	delete(ps.poolPaused, poolId)
	return nil
}

// FreezePool permanently freezes a pool. IRREVERSIBLE without governance.
func (ps *PauseState) FreezePool(poolId, admin, reason string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if _, ok := ps.poolFrozen[poolId]; ok {
		return fmt.Errorf("pool already frozen")
	}
	delete(ps.poolPaused, poolId)
	ps.poolFrozen[poolId] = &PoolFreezeInfo{
		PoolID: poolId, FrozenAt: time.Now(), FrozenBy: admin, Reason: reason,
	}
	return nil
}

// Status returns the current DEX status.
func (ps *PauseState) Status() DEXStatusResponse {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	resp := DEXStatusResponse{
		DEXPaused: ps.dexPaused, PausedPoolCount: len(ps.poolPaused), FrozenPoolCount: len(ps.poolFrozen),
	}
	if ps.dexPaused {
		t := ps.dexPausedAt
		resp.DEXPausedAt = &t
		resp.DEXPausedBy = ps.dexPausedBy
	}
	for _, info := range ps.poolPaused {
		resp.PausedPools = append(resp.PausedPools, info)
	}
	for _, info := range ps.poolFrozen {
		resp.FrozenPools = append(resp.FrozenPools, info)
	}
	return resp
}

// PoolStatus returns the status for a specific pool.
func (ps *PauseState) PoolStatus(poolId string) PoolStatusResponse {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	resp := PoolStatusResponse{PoolID: poolId, DEXPaused: ps.dexPaused}
	if info, ok := ps.poolPaused[poolId]; ok {
		resp.Paused = true
		resp.PauseInfo = info
	}
	if info, ok := ps.poolFrozen[poolId]; ok {
		resp.Frozen = true
		resp.FreezeInfo = info
	}
	return resp
}

// RegisterAdminRoutes registers all admin routes on the given mux.
func RegisterAdminRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/admin/pause", adminAuthWrap(handleAdminPause))
	mux.HandleFunc("/v1/admin/resume", adminAuthWrap(handleAdminResume))
	mux.HandleFunc("/v1/admin/pool/pause", adminAuthWrap(handleAdminPoolPause))
	mux.HandleFunc("/v1/admin/pool/resume", adminAuthWrap(handleAdminPoolResume))
	mux.HandleFunc("/v1/admin/pool/freeze", adminAuthWrap(handleAdminPoolFreeze))
	mux.HandleFunc("/v1/admin/pool/status", handleAdminPoolStatus)
	mux.HandleFunc("/v1/admin/status", handleAdminStatus)
}

func adminAuthWrap(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		role := r.Header.Get("X-User-Role")
		if role != AdminRole {
			writeAdminError(w, http.StatusForbidden, fmt.Errorf("admin role required"))
			return
		}
		next(w, r)
	}
}

func adminCallerID(r *http.Request) string {
	if id := r.Header.Get("X-User-ID"); id != "" {
		return id
	}
	return "unknown"
}

func writeAdminJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{"success": status < 400, "data": data})
}

func writeAdminError(w http.ResponseWriter, status int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{"success": false, "error": err.Error()})
}

type poolActionRequest struct {
	PoolID string `json:"poolId"`
	Reason string `json:"reason,omitempty"`
}

func handleAdminPause(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAdminError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	admin := adminCallerID(r)
	if err := pauseStateGlobal.PauseDEX(admin); err != nil {
		writeAdminError(w, http.StatusConflict, err)
		return
	}
	log.Printf("ADMIN: DEX PAUSED by %s", admin)
	writeAdminJSON(w, http.StatusOK, map[string]string{"status": "paused", "action": "dex_pause", "admin": admin})
}

func handleAdminResume(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAdminError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	admin := adminCallerID(r)
	if err := pauseStateGlobal.ResumeDEX(admin); err != nil {
		writeAdminError(w, http.StatusConflict, err)
		return
	}
	log.Printf("ADMIN: DEX RESUMED by %s", admin)
	writeAdminJSON(w, http.StatusOK, map[string]string{"status": "resumed", "action": "dex_resume", "admin": admin})
}

func handleAdminPoolPause(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAdminError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	var req poolActionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAdminError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}
	if req.PoolID == "" {
		writeAdminError(w, http.StatusBadRequest, fmt.Errorf("poolId is required"))
		return
	}
	admin := adminCallerID(r)
	if err := pauseStateGlobal.PausePool(req.PoolID, admin); err != nil {
		writeAdminError(w, http.StatusConflict, err)
		return
	}
	log.Printf("ADMIN: POOL PAUSED pool=%s by=%s", req.PoolID, admin)
	writeAdminJSON(w, http.StatusOK, map[string]string{"status": "paused", "action": "pool_pause", "poolId": req.PoolID, "admin": admin})
}

func handleAdminPoolResume(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAdminError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	var req poolActionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAdminError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}
	if req.PoolID == "" {
		writeAdminError(w, http.StatusBadRequest, fmt.Errorf("poolId is required"))
		return
	}
	admin := adminCallerID(r)
	if err := pauseStateGlobal.ResumePool(req.PoolID, admin); err != nil {
		writeAdminError(w, http.StatusConflict, err)
		return
	}
	log.Printf("ADMIN: POOL RESUMED pool=%s by=%s", req.PoolID, admin)
	writeAdminJSON(w, http.StatusOK, map[string]string{"status": "resumed", "action": "pool_resume", "poolId": req.PoolID, "admin": admin})
}

func handleAdminPoolFreeze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAdminError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	var req poolActionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAdminError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}
	if req.PoolID == "" {
		writeAdminError(w, http.StatusBadRequest, fmt.Errorf("poolId is required"))
		return
	}
	if req.Reason == "" {
		writeAdminError(w, http.StatusBadRequest, fmt.Errorf("reason is required for freeze (regulatory audit trail)"))
		return
	}
	admin := adminCallerID(r)
	if err := pauseStateGlobal.FreezePool(req.PoolID, admin, req.Reason); err != nil {
		writeAdminError(w, http.StatusConflict, err)
		return
	}
	log.Printf("ADMIN: POOL FROZEN (IRREVERSIBLE) pool=%s by=%s reason=%q", req.PoolID, admin, req.Reason)
	writeAdminJSON(w, http.StatusOK, map[string]string{
		"status": "frozen", "action": "pool_freeze", "poolId": req.PoolID,
		"admin": admin, "reason": req.Reason, "warning": "This action is IRREVERSIBLE without governance",
	})
}

func handleAdminPoolStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeAdminError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	poolId := r.URL.Query().Get("poolId")
	if poolId == "" {
		path := strings.TrimPrefix(r.URL.Path, "/v1/admin/pool/status")
		path = strings.TrimPrefix(path, "/")
		if path != "" {
			poolId = path
		}
	}
	if poolId == "" {
		writeAdminError(w, http.StatusBadRequest, fmt.Errorf("poolId is required"))
		return
	}
	writeAdminJSON(w, http.StatusOK, pauseStateGlobal.PoolStatus(poolId))
}

func handleAdminStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeAdminError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	writeAdminJSON(w, http.StatusOK, pauseStateGlobal.Status())
}
