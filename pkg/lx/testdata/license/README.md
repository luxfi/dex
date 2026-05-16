# Dex license-gate test fixtures

Tokens used by `pkg/lx/license_gate_test.go`.

Generate / regenerate with:

```sh
lux-license issue --dev --customer "dex-license-gate-tests" \
  --scope "dex" --out dex.token
lux-license issue --dev --customer "dex-license-gate-tests" \
  --scope "gpu" --out gpu.token
```

- `dex.token` — a token whose Scope = `["dex"]`. The matching-engine
  gate must accept it.
- `gpu.token` — a token whose Scope = `["gpu"]`. The matching-engine
  gate must reject it with a "scope not in" message and exit 1.

Tokens are signed by the embedded development key
(`license.DevPrivateKey()`). They cannot authenticate against a
production luxfi/license build that ships a customer-issuer public key.
The dev key fires a clear stderr warning every time it's used —
that warning is expected in test output.
