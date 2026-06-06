# ternary-pheromone-market

Autonomous GPU load balancing via ternary pheromone markets. Agents emit demand/supply pheromones, gossip via CRDT, and use ternary strategy nets to decide work migration. No central scheduler — load balancing is emergent.

## Overview

# Experiment B — Ternary Pheromone Market

Each GPU node is an autonomous agent that:

## Stats

- **Tests**: 19
- **LOC**: 765
- **License**: Apache-2.0

## Part of the Oxide Stack

This crate is part of the [Flux→PTX](https://github.com/SuperInstance/cuda-oxide/blob/main/FLUX_TO_PTX.md) experimental suite, testing synergies between the five layers of the distributed GPU runtime:

1. **open-parallel** — async runtime (tokio fork)
2. **pincher** — "Vector DB as runtime, LLM as compiler"
3. **flux-core** — bytecode VM + A2A agent protocol
4. **cuda-oxide** — Flux→MIR→Pliron→NVVM→PTX compiler
5. **cudaclaw** — persistent GPU kernels, warp-level consensus, SmartCRDT

## Usage

```rust
use ternary_pheromone_market::*;
// See tests in src/lib.rs for examples
```

## License

Apache-2.0
