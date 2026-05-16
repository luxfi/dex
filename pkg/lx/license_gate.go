// Copyright (C) 2020-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Commercial license gate for the Lux DEX matching engine.
//
// The matching engine is the proprietary core of lux-private/dex (see the
// repository LICENSE file for terms). Any binary that starts the matcher
// MUST call EnforceMatchingEngineLicense exactly once at process start so
// an unlicensed install fails closed with a clear, actionable stderr
// message instead of silently running a commercial path.
//
// This mirrors the pattern that cevm v0.50.0 established at the GPU state
// backend factory (~/work/luxcpp/cevm/bin/cevm/state_backend.hpp) and that
// the strict-PQ profile already uses for SignedOrder: one canonical gate
// in one place, every entry point dispatches through it.
//
// The scope name "dex" matches the convention documented in
// github.com/luxfi/license and the `lux-license issue --scope dex` flow.
package lx

import "github.com/luxfi/license"

// LicenseScope is the canonical scope name a customer token must include
// to be allowed to start the Lux DEX matching engine.
const LicenseScope = "dex"

// EnforceMatchingEngineLicense aborts the process (os.Exit(1)) unless the
// caller's environment contains a valid Lux commercial license token whose
// Scope list includes "dex". The check consults $LUX_LICENSE, then
// $LUX_LICENSE_FILE, then $HOME/.lux/license.jwt — same lookup order as
// every other Lux commercial binary. On success the call is silent so
// production logs stay quiet.
//
// Call this once, as the very first statement of main(), before any
// matching-engine state is constructed. Calling it later means an
// unlicensed operator can still observe a partially-initialised process
// before it dies, which defeats the fail-closed contract.
func EnforceMatchingEngineLicense() {
	license.HasFeatureOrDie(LicenseScope)
}
