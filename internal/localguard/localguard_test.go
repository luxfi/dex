// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package localguard

import (
	"strings"
	"testing"
)

func TestIsLoopbackTarget(t *testing.T) {
	cases := []struct {
		name   string
		target string
		want   bool
	}{
		// loopback, in every form the two tools accept.
		{"ipv4 loopback url with port", "http://127.0.0.1:9650", true},
		{"ipv4 loopback bare host:port", "127.0.0.1:9099", true},
		{"ipv4 loopback in 127/8", "http://127.0.0.5:9650", true},
		{"ipv6 loopback bare", "[::1]:9099", true},
		{"ipv6 loopback url", "http://[::1]:9650/ext/bc/C/rpc", true},
		{"localhost url", "http://localhost:9650", true},
		{"localhost bare", "localhost:9099", true},

		// NON-loopback: the cluster-poisoning shapes the guard must block.
		{"k8s service dns", "http://luxd-0.luxd-headless.lux-devnet.svc.cluster.local:9650", false},
		{"lux dev network host", "https://api.lux-dev.network", false},
		{"public ip", "http://8.8.8.8:9650", false},
		{"unspecified 0.0.0.0 is not loopback", "0.0.0.0:9099", false},
		{"private lan ip", "http://10.0.0.5:9650", false},
		{"empty", "", false},
		{"garbage", "://::::", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsLoopbackTarget(tc.target); got != tc.want {
				t.Fatalf("IsLoopbackTarget(%q) = %v, want %v", tc.target, got, tc.want)
			}
		})
	}
}

func TestAssertLocalOnly(t *testing.T) {
	loopback := []string{"http://127.0.0.1:9650", "127.0.0.1:9099"}
	cluster := []string{"http://luxd-0.luxd-headless.lux-devnet.svc.cluster.local:9650"}
	mixed := []string{"http://127.0.0.1:9650", "http://luxd-1.lux-dev.network:9650"}

	// (a) non-loopback target is rejected (even with opt-in).
	if err := AssertLocalOnly(cluster, true); err == nil {
		t.Fatal("AssertLocalOnly(cluster, optIn=true) = nil, want rejection of non-loopback target")
	}
	// a mixed set where ANY target is non-loopback must also be rejected.
	if err := AssertLocalOnly(mixed, true); err == nil {
		t.Fatal("AssertLocalOnly(mixed, optIn=true) = nil, want rejection (one target is non-loopback)")
	}

	// (b) loopback target WITHOUT the opt-in flag is rejected.
	err := AssertLocalOnly(loopback, false)
	if err == nil {
		t.Fatal("AssertLocalOnly(loopback, optIn=false) = nil, want rejection (no opt-in)")
	}
	if !strings.Contains(err.Error(), OptInFlag) {
		t.Fatalf("missing-opt-in error should name the flag %q, got: %v", OptInFlag, err)
	}

	// (c) loopback + opt-in is allowed.
	if err := AssertLocalOnly(loopback, true); err != nil {
		t.Fatalf("AssertLocalOnly(loopback, optIn=true) = %v, want nil (allowed)", err)
	}

	// empty target set is an error (nothing to prove local).
	if err := AssertLocalOnly(nil, true); err == nil {
		t.Fatal("AssertLocalOnly(nil, optIn=true) = nil, want error for empty target set")
	}
}
