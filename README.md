# Ternary Pheromone Market — Emergent GPU Load Balancing via Ternary Pheromone Signals

**Ternary Pheromone Market** implements autonomous GPU load balancing where each node is an agent that emits demand/supply pheromones into a G-Counter CRDT, gossips state to ring neighbors, and runs a ternary {-1, 0, +1} strategy network to decide work migration. No central scheduler — load balancing is an emergent property of local pheromone following, with token economics ensuring migrations are mutually voluntary and conserved fleet-wide.

## Why It Matters

Centralized schedulers are the bottleneck and single point of failure in large GPU clusters. If each node instead follows a simple rule — "move work toward nodes with spare capacity, away from overloaded ones" — global load balancing emerges from local interactions. The ternary strategy network makes this computationally cheap: each node computes {-1 (migrate away), 0 (hold), +1 (accept work)} based on local pheromone gradients. The G-Counter CRDT ensures gossip convergence without coordination, and token economics (each node has a budget of migration tokens) prevents freeloading. This approach scales to thousands of nodes with O(1) per-node overhead per round.

## How It Works

### Pheromone CRDT

Each node maintains a G-Counter CRDT — a `HashMap<NodeId, u64>` where each node owns exactly one slot:

```
demand[node_i] = node_i's current demand level
supply[node_i] = node_i's current supply capacity
```

Merge is max per slot: commutative, idempotent, associative. O(N) to merge full state, but typically O(fanout) with ring topology.

### Ternary Strategy Network

Each node runs a tiny ternary network over its pheromone gradient:

```
inputs:  local_demand, neighbor_avg_demand, local_supply, neighbor_avg_supply
weights: {-1, 0, +1} ternary matrix
output:  {-1 (migrate), 0 (hold), +1 (accept)}
```

The network is a single layer: `output = sign(Σ wᵢ × inputᵢ)` in Z₃ arithmetic. In production, weights pack 32 per u64 (2-bit encoding, 16× FP32 density).

### Migration Protocol

When a node decides to migrate work (-1 output):
1. Check token balance (must have migration tokens)
2. Find neighbor with accept signal (+1)
3. Transfer work units (mutual voluntary exchange)
4. Decrement both nodes' token balances

When a node accepts (+1): increment local load, acknowledge sender.

### Ring Topology Gossip

Nodes are arranged in a ring. Each round, a node gossips with its clockwise and counterclockwise neighbors. This provides O(log N) convergence time for information to propagate across the ring.

### Conservation

Token economics ensure γ + η = C:
- γ = total work completed (tokens earned)
- η = idle capacity (tokens unspent)
- C = total tokens (conserved)

No node can accept more work than it has tokens for; no node can offload more work than it has migration tokens.

## Quick Start

```rust
use ternary_pheromone_market::{Trit, Rng, GCounter};

// Simulate 4 nodes in a ring
let mut counters: Vec<GCounter> = (0..4).map(|i| {
    let mut c = GCounter::default();
    c.0.insert(i, (i as u64) * 10); // varying demand
    c
}).collect();

// Merge counter from node 0 into node 1
counters[1].merge(&counters[0]);
```

```bash
cargo add ternary-pheromone-market
```

## API

| Type / Function | Description |
|---|---|
| `Trit` | Ternary weight {-1, 0, +1} |
| `GCounter` | CRDT counter: `merge()`, per-node slots |
| `Rng` | XorShift64 deterministic RNG for strategy init |

## Architecture Notes

This is the decentralized scheduler for **SuperInstance** GPU fleets. No central orchestrator — each GPU node autonomously decides work migration based on local pheromone signals. The γ + η = C conservation is enforced by the token economy: total work is bounded by token supply, preventing runaway migration cascades. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Dorigo, Marco & Stützle, Thomas. *Ant Colony Optimization*, MIT Press, 2004 — pheromone-based optimization.
| Bonnet, François & Raynal, Michel. "Gossip Protocols," *Distributed Computing*, 2017.
| Li, Feng et al. "Ternary Weight Networks," *arXiv:1605.04711*, 2016 — ternary strategy nets.

## License

Apache-2.0
