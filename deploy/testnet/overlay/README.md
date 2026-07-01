# hanzozt overlay for the dexd testnet (opt-in)

**Use this only if evo and spark cannot reach each other directly.** On the
direct 2.5 GbE link, skip the overlay entirely — the venues already publish on
the LAN (`BIND_ADDR=0.0.0.0`) and `make bench` / `make verify` dial the box IPs
directly, giving you the true wire numbers with zero overlay overhead. The
overlay is for NAT / multi-site (e.g. spark off-prem): it trades throughput and
latency for connectivity through NAT.

The overlay carries **client → venue** ZAP traffic only. The venues still do not
gossip (see ../topology.md) — there is no consensus traffic to tunnel.

## What only you can provide

- your hanzozt (OpenZiti fork) **controller** address,
- an **enrollment JWT** per box (the "overlay keys"),
- a pinned tunneler image tag for `ZT_IMAGE` (don't ship `:latest`).

Everything else is in `compose.hanzozt.yml`.

## Controller-side recipe (run once, with the `zt` CLI)

Model **one Ziti service per box** that forwards the venue port range, so you
don't define a service per node. Adjust `9099-9109` to `BASE_PORT .. BASE_PORT+N-1`.

```bash
zt edge login <controller>            # your hanzozt controller

# identities: one host identity per venue box, one dialer for the bench/client
zt edge create identity dex-evo-host    -o dex-evo-host.jwt
zt edge create identity dex-spark-host  -o dex-spark-host.jwt
zt edge create identity dex-bench       -o dex-bench.jwt

# config: each box hosts its local venue port range; the dialer intercepts a name
zt edge create config dex-evo-host.v1   host.v1 \
  '{"protocol":"tcp","address":"127.0.0.1","forwardPort":true,"allowedPortRanges":[{"low":9099,"high":9109}]}'
zt edge create config dex-evo-intercept.v1 intercept.v1 \
  '{"protocols":["tcp"],"addresses":["dex-evo.zt"],"portRanges":[{"low":9099,"high":9109}]}'
# ...repeat the two configs for spark (dex-spark.zt) ...

zt edge create service dex-evo   --configs dex-evo-host.v1,dex-evo-intercept.v1
zt edge create service dex-spark --configs dex-spark-host.v1,dex-spark-intercept.v1

# policies: host identities BIND their service; the bench DIALs both
zt edge create service-policy dex-evo-bind   Bind --service-roles '@dex-evo'   --identity-roles '@dex-evo-host'
zt edge create service-policy dex-spark-bind Bind --service-roles '@dex-spark' --identity-roles '@dex-spark-host'
zt edge create service-policy dex-dial       Dial --service-roles '@dex-evo','@dex-spark' --identity-roles '@dex-bench'
# (ensure edge-router-policy / service-edge-router-policy grant all three a router)
```

## Box-side wiring

On **each venue box** (host mode — binds its service to the local venues):

```bash
cd deploy/testnet
ZT_MODE=run-host ZT_IDENTITY_BASENAME=dex-evo-host \
  ZT_ENROLL_TOKEN="$(cat dex-evo-host.jwt)" ZT_IMAGE=hanzozt/zt-edge-tunnel:<tag> \
  docker compose -f compose.gen.yml -f overlay/compose.hanzozt.yml up -d
# spark: ZT_IDENTITY_BASENAME=dex-spark-host, its own jwt
```

On the **bench box** (proxy mode — local ports forward to the overlay services):

```bash
cd deploy/testnet
ZT_MODE=run-proxy ZT_IDENTITY_BASENAME=dex-bench \
  ZT_ENROLL_TOKEN="$(cat dex-bench.jwt)" ZT_IMAGE=hanzozt/zt-edge-tunnel:<tag> \
  docker compose -f overlay/compose.hanzozt.yml up -d
```

`run-proxy` binds a local TCP port for each dialable service/port. Point the bench
at those local ports (or, in full `run` tun mode, at the intercept names directly):

```bash
# tun mode: dial the intercept names over the overlay
docker run --rm --network host --entrypoint dexbench dexd-testnet:local \
  -addrs dex-evo.zt:9099,dex-evo.zt:9100,dex-spark.zt:9099 -conns 16 -duration 30s
```

## Expectation

Overlay path latency = LAN latency + mTLS + fabric hop; throughput is bounded by
the tunneler's crypto, not the 2.5 GbE. Report overlay numbers as a separate
column from the direct-LAN baseline — don't compare them head-to-head.
