# ternary-pheromone-market

Autonomous GPU load balancing via ternary pheromone markets. Agents emit demand/supply pheromones, gossip via CRDT, and use ternary strategy nets to decide work migration. No central scheduler — load balancing is emergent.

## Why This Matters

# Experiment B — Ternary Pheromone Market
Each GPU node is an autonomous agent that:
1. Emits demand/supply pheromones into a G-Counter CRDT
2. Gossips pheromone state to ring-topology neighbors (simulated A2A)
3. Runs a ternary {-1,0,+1} strategy net over the local gradient

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub type NodeId
pub struct Trit
pub fn new
pub fn val
pub struct Rng
pub fn new
pub fn next
pub fn f32
pub struct GCounter
pub fn new
pub fn increment
pub fn value
```

## Usage

```toml
[dependencies]
ternary-pheromone-market = "0.1.0"
```

```rust
use ternary_pheromone_market::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/ternary-pheromone-market.git
cd ternary-pheromone-market
cargo test    # 19 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 19 |
| Lines of Rust | 766 |
| Public API | 40 items |

## License

Apache-2.0
