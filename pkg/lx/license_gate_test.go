// Copyright (C) 2020-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/luxfi/dex/pkg/lx"
)

// TestLicenseScopeIsDex pins the scope string. If somebody renames it to
// "matcher" or "lx" the customer tokens issued under the documented
// "lux-license issue --scope dex" flow would silently stop working.
func TestLicenseScopeIsDex(t *testing.T) {
	if lx.LicenseScope != "dex" {
		t.Fatalf("LicenseScope = %q, want %q", lx.LicenseScope, "dex")
	}
}

// TestEnforceMatchingEngineLicense exercises EnforceMatchingEngineLicense
// across the three states an operator can land in. We re-exec ourselves
// because the gate calls os.Exit(1) on the unhappy path, which can only
// be observed via process exit code, never via panic recovery.
//
// Token fixtures are produced offline by `lux-license issue --dev` and
// stored in testdata/. Regenerate via testdata/regen.sh.
func TestEnforceMatchingEngineLicense(t *testing.T) {
	if os.Getenv("LX_LICENSE_GATE_SUBPROCESS") == "1" {
		// Child mode: invoke the gate and exit. The parent inspects the
		// exit code + stderr.
		lx.EnforceMatchingEngineLicense()
		// Reach here = gate accepted the license.
		os.Exit(0)
	}

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	testdata := filepath.Join(wd, "testdata", "license")

	dexTok := mustRead(t, filepath.Join(testdata, "dex.token"))
	gpuTok := mustRead(t, filepath.Join(testdata, "gpu.token"))

	cases := []struct {
		name       string
		env        map[string]string
		wantExit   int
		wantStderr string
	}{
		{
			name:       "no_license",
			env:        map[string]string{"LUX_LICENSE": "", "LUX_LICENSE_FILE": "", "HOME": t.TempDir()},
			wantExit:   1,
			wantStderr: `no valid license is present`,
		},
		{
			name:       "wrong_scope",
			env:        map[string]string{"LUX_LICENSE": gpuTok, "LUX_LICENSE_FILE": "", "HOME": t.TempDir()},
			wantExit:   1,
			wantStderr: `does not include scope "dex"`,
		},
		{
			name:     "correct_scope",
			env:      map[string]string{"LUX_LICENSE": dexTok, "LUX_LICENSE_FILE": "", "HOME": t.TempDir()},
			wantExit: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(exe, "-test.run=TestEnforceMatchingEngineLicense", "-test.v")
			cmd.Env = append(os.Environ(), "LX_LICENSE_GATE_SUBPROCESS=1")
			for k, v := range tc.env {
				cmd.Env = appendEnv(cmd.Env, k, v)
			}
			var stderr strings.Builder
			cmd.Stderr = &stderr
			cmd.Stdout = &strings.Builder{}
			err = cmd.Run()

			gotExit := 0
			if ee, ok := err.(*exec.ExitError); ok {
				gotExit = ee.ExitCode()
			} else if err != nil {
				t.Fatalf("unexpected exec error: %v", err)
			}
			if gotExit != tc.wantExit {
				t.Errorf("exit = %d, want %d\nstderr:\n%s", gotExit, tc.wantExit, stderr.String())
			}
			if tc.wantStderr != "" && !strings.Contains(stderr.String(), tc.wantStderr) {
				t.Errorf("stderr does not contain %q\ngot:\n%s", tc.wantStderr, stderr.String())
			}
		})
	}

}

func mustRead(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return strings.TrimSpace(string(b))
}

// appendEnv replaces (or appends) KEY=value in an env slice. Necessary
// because exec.Cmd.Env shadows the parent on duplicate keys: the LAST
// occurrence wins on POSIX but FIRST on some legacy builds, so we
// canonicalise to "single occurrence, last value".
func appendEnv(env []string, key, value string) []string {
	prefix := key + "="
	out := env[:0]
	for _, kv := range env {
		if !strings.HasPrefix(kv, prefix) {
			out = append(out, kv)
		}
	}
	return append(out, prefix+value)
}
