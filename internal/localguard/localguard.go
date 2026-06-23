// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package localguard refuses synthetic-data seeding tools from running against
// anything but a local, loopback target.
//
// WHY THIS EXISTS: the cmd/clobverify and cmd/fourpath-live harnesses seed
// markets/orders using ASCII-of-symbol SYNTHETIC asset identities (e.g. the
// poolID for "LUX/LUSD" is literally the ASCII bytes 4c55582f4c555344...). That
// is fine for an ephemeral single-process LOCAL test, but if those frames are
// sent to a SHARED/LIVE D-Chain validator set they persist into the replication
// snapshot. The native D-Chain VM (dex/pkg/dchain) then refuses to boot on that
// snapshot — assertOrderUserCoverage sees a resting order on a market with no
// orderuser row and treats the settlement path as degraded ("refusing to start
// on a degraded settlement path"), bricking the chain.
//
// The guard makes that path IMPOSSIBLE: a seeding tool must (1) pass an explicit
// opt-in flag AND (2) point only at loopback. A non-loopback target is rejected
// BEFORE any frame is sent.
package localguard

import (
	"fmt"
	"net"
	"net/url"
	"strings"
)

// OptInFlag is the single, canonical opt-in flag name both seeding tools use.
// Passing it asserts the operator understands these tools write synthetic test
// data; it does NOT relax the loopback requirement.
const OptInFlag = "i-understand-this-seeds-synthetic-test-data"

// hostOf extracts the host (no port) from a target that may be either a URL
// ("http://host:port/path") or a bare authority ("host:port" or "host").
func hostOf(target string) (string, error) {
	t := strings.TrimSpace(target)
	if t == "" {
		return "", fmt.Errorf("empty target")
	}
	// URL form: has a scheme.
	if strings.Contains(t, "://") {
		u, err := url.Parse(t)
		if err != nil {
			return "", fmt.Errorf("parse %q: %w", target, err)
		}
		if h := u.Hostname(); h != "" {
			return h, nil
		}
		return "", fmt.Errorf("no host in %q", target)
	}
	// Bare authority form. SplitHostPort needs a port; fall back to the whole
	// string when there is none.
	if h, _, err := net.SplitHostPort(t); err == nil {
		return h, nil
	}
	return t, nil
}

// IsLoopbackTarget reports whether target resolves ENTIRELY to loopback. It
// accepts URL or bare host[:port] forms. It is fail-closed: a malformed target,
// an unresolvable name, or ANY non-loopback resolved address makes it false.
// "localhost" and a literal 127.0.0.0/8 or ::1 are loopback; a Kubernetes
// service DNS name (luxd-*.svc.cluster.local), a public host, or 0.0.0.0 are not.
func IsLoopbackTarget(target string) bool {
	host, err := hostOf(target)
	if err != nil {
		return false
	}
	// A bare IP literal: classify directly (no DNS).
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	// A name: resolve it and require EVERY address to be loopback. An empty
	// result (nothing resolved) is not loopback.
	addrs, err := net.LookupHost(host)
	if err != nil || len(addrs) == 0 {
		return false
	}
	for _, a := range addrs {
		ip := net.ParseIP(a)
		if ip == nil || !ip.IsLoopback() {
			return false
		}
	}
	return true
}

// AssertLocalOnly enforces the seeding guard over every target. It returns a
// non-nil error (the tool must exit non-zero, before sending any frame) when:
//   - the opt-in flag was not passed, or
//   - any target is non-loopback.
//
// targets is the full set of write destinations the tool will touch (validator
// base URLs, a venue host:port, etc.). An empty target set is an error: there is
// nothing to prove local.
func AssertLocalOnly(targets []string, optIn bool) error {
	if len(targets) == 0 {
		return fmt.Errorf("no target supplied")
	}
	if !optIn {
		return fmt.Errorf("this tool seeds SYNTHETIC test data and refuses to run without --%s "+
			"(local-only harness; see package doc)", OptInFlag)
	}
	for _, t := range targets {
		if !IsLoopbackTarget(t) {
			return fmt.Errorf("refusing to seed synthetic test data at non-loopback target %q: "+
				"these ASCII-of-symbol markets brick the native D-Chain VM (assertOrderUserCoverage) "+
				"if they reach a shared/live chain; point only at localhost/127.0.0.1/::1", t)
		}
	}
	return nil
}
