# LP-108 Archive — 2026-05-04

DEX vapor pieces removed from production paths.

## mlx_engine.go + mlx_engine_test.go

`pkg/engine/mlx_engine.go` claimed to be an MLX-accelerated matching
engine. Real body (line 58-69):

```go
// Simulate MLX GPU processing
// In production, this would use Metal Performance Shaders
processed := uint64(len(orders))
executed := processed / 10 // 10% fill rate

// Simulate 597ns per order latency
processingTime := time.Duration(processed*597) * time.Nanosecond
if processingTime > 0 {
    time.Sleep(processingTime / 1000) // Scale down for simulation
}
```

`time.Sleep` simulating GPU processing. No Metal Performance Shaders,
no MLX device, no kernel. The "597 ns/order" was a hardcoded
constant, not a measurement.

## fpga_accelerator.go + pkg/fpga/

`pkg/lx/fpga_accelerator.go` (484 LoC) and `pkg/fpga/` (8 files,
~3K LoC: amd_versal, aws_f2, fpga_engine, fpga_interface, types,
stubs, plus tests). All interface-only. No bitstream, no driver, no
real FPGA dispatch. Tests at `pkg/lx/fpga_accelerator_test.go` and
`pkg/lx/coverage_improvement_test.go::TestFPGAAccelerator*` exercised
the stub itself ("DisabledFPGA returns error"), not any hardware.

## test/benchmark/orderbook_bench_test.go

`BenchmarkMLXEngine` called the standard CPU `lx.NewOrderBook(...)`
then reported a hardcoded `b.ReportMetric(597, "ns/order")` — a
fabricated number, not a measurement.

`BenchmarkPlanetScale` reported entirely made-up metrics
(markets=5M, orders/sec=150M, ns/order=597, watts=370,
orders/watt=405405) that no code in the repo measures.

Both removed. Honest CPU bench remains at `BenchmarkOrderBook` /
`BenchmarkOrderBookParallel` above. Real C++ numbers are at
`luxcpp/dex/build/luxdex_bench` (5.51 M orders/sec single-symbol
on Apple M1 — measured).

## orderbook.go::detectBestBackend

Used to set `currentBackend` to one of {BackendGo, BackendCGO,
BackendMLX, BackendCUDA} based on env probes. No code path
branched on the enum — the matching engine ran the Go path
regardless. Collapsed to `BackendGo` only.

## When real GPU/FPGA matching ships

Add the variants back AND wire them through the match path AND add
a parity test against `BackendGo` AND a reproducible benchmark.
LP-108 production gate applies.
