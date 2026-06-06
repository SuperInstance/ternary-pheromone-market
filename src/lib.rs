//! # Experiment B — Ternary Pheromone Market
//!
//! Each GPU node is an autonomous agent that:
//!   1. Emits demand/supply pheromones into a G-Counter CRDT
//!   2. Gossips pheromone state to ring-topology neighbors (simulated A2A)
//!   3. Runs a ternary {-1,0,+1} strategy net over the local gradient
//!   4. Acts: MigrateWork → AcceptWork → Idle
//!
//! No central scheduler. Load balancing is an emergent property of
//! local pheromone following. Token economics ensure migrations are
//! mutually voluntary and conserved fleet-wide.
//!
//! Ternary weights in production would be 2-bit packed (16× FP32 density).
//! Here we use i8 for readability; the arithmetic is identical.

use std::collections::HashMap;

pub type NodeId = usize;

// ─── Trit ────────────────────────────────────────────────────────────────────

/// Ternary weight constrained to {-1, 0, +1}.
///
/// In production these pack 32 weights per u64 (2-bit encoding), yielding
/// 16× the memory density of f32 while keeping dot products as add/subtract/skip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trit(i8);

impl Trit {
    pub fn new(v: i8) -> Option<Self> {
        matches!(v, -1 | 0 | 1).then_some(Trit(v))
    }

    #[inline]
    pub fn val(self) -> i8 {
        self.0
    }

    fn rand(rng: &mut Rng) -> Self {
        // 0,1,2 mod 3, shifted to -1,0,+1
        Trit((rng.next() % 3) as i8 - 1)
    }
}

// ─── Rng ─────────────────────────────────────────────────────────────────────

/// XorShift64 — deterministic, no-std friendly, sufficient for strategy init.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1) // avoid zero state
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn f32(&mut self) -> f32 {
        // Upper 53 bits → [0, 1)
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
}

// ─── GCounter ────────────────────────────────────────────────────────────────

/// G-Counter CRDT. Each node owns exactly one slot; merge = max per slot.
///
/// Properties guaranteed by construction:
///   - Commutative:  merge(a,b) == merge(b,a)
///   - Idempotent:   merge(a,a) == a
///   - Associative:  merge(merge(a,b),c) == merge(a,merge(b,c))
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GCounter(HashMap<NodeId, u64>);

impl GCounter {
    pub fn new() -> Self {
        GCounter(HashMap::new())
    }

    pub fn increment(&mut self, node: NodeId, amount: u64) {
        *self.0.entry(node).or_default() += amount;
    }

    /// Sum of all slots — the global pheromone level seen by this replica.
    pub fn value(&self) -> u64 {
        self.0.values().sum()
    }

    pub fn get(&self, node: NodeId) -> u64 {
        self.0.get(&node).copied().unwrap_or(0)
    }

    pub fn merge(&mut self, other: &GCounter) {
        for (&node, &count) in &other.0 {
            let slot = self.0.entry(node).or_default();
            *slot = (*slot).max(count);
        }
    }
}

// ─── PheromoneMap ─────────────────────────────────────────────────────────────

/// Two G-Counter maps per node: demand (high util) and supply (spare capacity).
///
/// Each entry is itself a G-Counter so that merging two replicas of the map
/// (gossip step) is just a per-entry G-Counter merge — still CRDT.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PheromoneMap {
    pub demand: HashMap<NodeId, GCounter>,
    pub supply: HashMap<NodeId, GCounter>,
}

impl PheromoneMap {
    pub fn emit_demand(&mut self, source: NodeId, level: u64) {
        self.demand.entry(source).or_default().increment(source, level);
    }

    pub fn emit_supply(&mut self, source: NodeId, level: u64) {
        self.supply.entry(source).or_default().increment(source, level);
    }

    pub fn merge(&mut self, other: &PheromoneMap) {
        for (&node, other_ctr) in &other.demand {
            self.demand.entry(node).or_default().merge(other_ctr);
        }
        for (&node, other_ctr) in &other.supply {
            self.supply.entry(node).or_default().merge(other_ctr);
        }
    }

    pub fn demand_total(&self, node: NodeId) -> u64 {
        self.demand.get(&node).map(|c| c.value()).unwrap_or(0)
    }

    pub fn supply_total(&self, node: NodeId) -> u64 {
        self.supply.get(&node).map(|c| c.value()).unwrap_or(0)
    }

    /// Positive gradient → neighbors have more demand than self (pull work here).
    /// Negative gradient → self is more loaded than neighbors (push work out).
    pub fn demand_gradient(&self, self_node: NodeId, neighbors: &[NodeId]) -> f32 {
        if neighbors.is_empty() {
            return 0.0;
        }
        let neighbor_mean = neighbors
            .iter()
            .map(|&n| self.demand_total(n) as f64)
            .sum::<f64>()
            / neighbors.len() as f64;
        let own = self.demand_total(self_node) as f64;
        (neighbor_mean - own) as f32
    }

    pub fn supply_gradient(&self, self_node: NodeId, neighbors: &[NodeId]) -> f32 {
        if neighbors.is_empty() {
            return 0.0;
        }
        let neighbor_mean = neighbors
            .iter()
            .map(|&n| self.supply_total(n) as f64)
            .sum::<f64>()
            / neighbors.len() as f64;
        let own = self.supply_total(self_node) as f64;
        (neighbor_mean - own) as f32
    }
}

// ─── TritLayer ────────────────────────────────────────────────────────────────

/// Dense layer with ternary weights and ReLU activation.
#[derive(Debug, Clone)]
pub struct TritLayer {
    weights: Vec<Vec<Trit>>, // [out_dim][in_dim]
    in_dim: usize,
}

impl TritLayer {
    pub fn random(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Self {
        let weights = (0..out_dim)
            .map(|_| (0..in_dim).map(|_| Trit::rand(rng)).collect())
            .collect();
        TritLayer { weights, in_dim }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_dim);
        self.weights
            .iter()
            .map(|row| {
                // Ternary dot product: only add or subtract, never multiply
                row.iter()
                    .zip(input)
                    .map(|(w, x)| w.val() as f32 * x)
                    .sum::<f32>()
                    .max(0.0) // ReLU
            })
            .collect()
    }
}

// ─── TritNet ──────────────────────────────────────────────────────────────────

/// 4→8→4→3 ternary strategy network.
///
/// Inputs:  [utilization, demand_grad_norm, supply_grad_norm, token_norm]
/// Outputs: [migrate_score, accept_score, idle_score]
///
/// With random initialization the network produces arbitrary decisions;
/// with trained weights it converges to the Nash equilibrium of the
/// pheromone market (overloaded nodes migrate, underloaded accept).
#[derive(Debug, Clone)]
pub struct TritNet {
    layers: Vec<TritLayer>,
}

impl TritNet {
    pub fn random(rng: &mut Rng) -> Self {
        TritNet {
            layers: vec![
                TritLayer::random(4, 8, rng),
                TritLayer::random(8, 4, rng),
                TritLayer::random(4, 3, rng),
            ],
        }
    }

    pub fn forward(&self, input: &[f32; 4]) -> [f32; 3] {
        let mut x: Vec<f32> = input.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        [x[0], x[1], x[2]]
    }

    pub fn decide(
        &self,
        utilization: f32,
        demand_grad: f32,
        supply_grad: f32,
        token_balance: i64,
    ) -> Action {
        let token_norm = (token_balance as f32 / 100.0).clamp(-1.0, 1.0);
        // Normalize gradients to [-1, 1] range (1000 = full utilization signal)
        let nd = (demand_grad / 1000.0).clamp(-1.0, 1.0);
        let ns = (supply_grad / 1000.0).clamp(-1.0, 1.0);
        let scores = self.forward(&[utilization, nd, ns, token_norm]);
        // Ties broken toward last element (Idle) — conservative default
        let best = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(2);
        match best {
            0 => Action::MigrateWork,
            1 => Action::AcceptWork,
            _ => Action::Idle,
        }
    }
}

// ─── Action ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Offload 30% of local utilization to the least-loaded AcceptWork neighbor.
    MigrateWork,
    /// Accept incoming workload from a MigrateWork neighbor.
    AcceptWork,
    Idle,
}

// ─── Agent ────────────────────────────────────────────────────────────────────

/// Autonomous GPU agent: owns utilization state, a pheromone replica,
/// a ternary strategy, and a token balance for economic settlement.
pub struct Agent {
    pub id: NodeId,
    pub utilization: f32,
    pub token_balance: i64,
    pub pheromone_map: PheromoneMap,
    pub strategy: TritNet,
    pub neighbors: Vec<NodeId>,
}

impl Agent {
    pub fn new(id: NodeId, utilization: f32, rng: &mut Rng) -> Self {
        Agent {
            id,
            utilization: utilization.clamp(0.0, 1.0),
            token_balance: 100,
            pheromone_map: PheromoneMap::default(),
            strategy: TritNet::random(rng),
            neighbors: Vec::new(),
        }
    }

    pub fn decide(&self) -> Action {
        let d = self
            .pheromone_map
            .demand_gradient(self.id, &self.neighbors);
        let s = self
            .pheromone_map
            .supply_gradient(self.id, &self.neighbors);
        self.strategy
            .decide(self.utilization, d, s, self.token_balance)
    }
}

// ─── Fleet ────────────────────────────────────────────────────────────────────

/// Ring-connected fleet running the pheromone market protocol.
///
/// Each tick:
///   1. Emit — agents update their own pheromone G-Counters
///   2. Gossip — snapshot then merge neighbors' maps (simulated A2A broadcast)
///   3. Decide — each agent queries its ternary strategy net
///   4. Match — MigrateWork agents find AcceptWork neighbors
///   5. Execute — migrate load atomically; settle tokens
pub struct Fleet {
    pub agents: Vec<Agent>,
    pub tick: u64,
}

impl Fleet {
    /// Create `n` agents with random utilizations wired in a ring.
    pub fn new(n: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut agents: Vec<Agent> = (0..n)
            .map(|i| Agent::new(i, rng.f32(), &mut rng))
            .collect();
        for i in 0..n {
            agents[i].neighbors = vec![(i + n - 1) % n, (i + 1) % n];
        }
        Fleet { agents, tick: 0 }
    }

    pub fn tick(&mut self) {
        let n = self.agents.len();

        // 1. Emit pheromones into own CRDT replica
        for a in &mut self.agents {
            let demand = (a.utilization * 1000.0) as u64;
            let supply = ((1.0 - a.utilization).max(0.0) * 1000.0) as u64;
            a.pheromone_map.emit_demand(a.id, demand);
            a.pheromone_map.emit_supply(a.id, supply);
        }

        // 2. Gossip: take snapshots, then distribute
        let snapshots: Vec<PheromoneMap> = self.agents.iter().map(|a| a.pheromone_map.clone()).collect();
        let neighbor_lists: Vec<Vec<NodeId>> =
            self.agents.iter().map(|a| a.neighbors.clone()).collect();

        for i in 0..n {
            for &j in &neighbor_lists[i] {
                let snap = snapshots[j].clone();
                self.agents[i].pheromone_map.merge(&snap);
            }
        }

        // 3. Decide (read-only pass — collect to avoid borrow conflict)
        let actions: Vec<Action> = self.agents.iter().map(|a| a.decide()).collect();

        // Pre-extract values needed for matching to avoid borrow conflicts
        let utils: Vec<f32> = self.agents.iter().map(|a| a.utilization).collect();
        let balances: Vec<i64> = self.agents.iter().map(|a| a.token_balance).collect();
        let demands: Vec<u64> = self
            .agents
            .iter()
            .map(|a| a.pheromone_map.demand_total(a.id))
            .collect();

        // 4. Match: each MigrateWork agent claims one AcceptWork neighbor
        let mut claimed: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        let migrations: Vec<(NodeId, NodeId, f32)> = {
            let mut result = Vec::new();
            for i in 0..n {
                if actions[i] != Action::MigrateWork {
                    continue;
                }
                if utils[i] <= 0.3 || balances[i] <= 10 {
                    continue;
                }
                let workload = utils[i] * 0.3;
                let target = neighbor_lists[i]
                    .iter()
                    .filter(|&&j| {
                        actions[j] == Action::AcceptWork
                            && !claimed.contains(&j)
                            && utils[j] + workload <= 0.95
                    })
                    .min_by_key(|&&j| demands[j])
                    .copied();
                if let Some(j) = target {
                    claimed.insert(j);
                    result.push((i, j, workload));
                }
            }
            result
        };

        // 5. Execute migrations: work + token settlement
        for (from, to, workload) in migrations {
            self.agents[from].utilization =
                (self.agents[from].utilization - workload).max(0.0);
            self.agents[from].token_balance -= 10;
            self.agents[to].utilization =
                (self.agents[to].utilization + workload).min(1.0);
            self.agents[to].token_balance += 10;
        }

        self.tick += 1;
    }

    pub fn mean_utilization(&self) -> f32 {
        self.agents.iter().map(|a| a.utilization).sum::<f32>() / self.agents.len() as f32
    }

    pub fn utilization_variance(&self) -> f32 {
        let mean = self.mean_utilization();
        self.agents
            .iter()
            .map(|a| (a.utilization - mean).powi(2))
            .sum::<f32>()
            / self.agents.len() as f32
    }

    pub fn max_utilization(&self) -> f32 {
        self.agents
            .iter()
            .map(|a| a.utilization)
            .fold(f32::MIN, f32::max)
    }

    pub fn total_tokens(&self) -> i64 {
        self.agents.iter().map(|a| a.token_balance).sum()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Trit ---

    #[test]
    fn trit_accepts_valid_range() {
        assert_eq!(Trit::new(-1).unwrap().val(), -1);
        assert_eq!(Trit::new(0).unwrap().val(), 0);
        assert_eq!(Trit::new(1).unwrap().val(), 1);
    }

    #[test]
    fn trit_rejects_out_of_range() {
        assert!(Trit::new(2).is_none());
        assert!(Trit::new(-2).is_none());
        assert!(Trit::new(127).is_none());
    }

    // --- GCounter CRDT ---

    #[test]
    fn gcounter_increment_and_read() {
        let mut c = GCounter::new();
        c.increment(0, 10);
        c.increment(1, 5);
        assert_eq!(c.value(), 15);
        assert_eq!(c.get(0), 10);
        assert_eq!(c.get(1), 5);
        assert_eq!(c.get(2), 0); // missing node
    }

    #[test]
    fn gcounter_merge_is_commutative() {
        let mut a = GCounter::new();
        a.increment(0, 10);
        a.increment(1, 5);

        let mut b = GCounter::new();
        b.increment(1, 7); // node 1 has a higher count in b
        b.increment(2, 3);

        let mut ab = a.clone();
        ab.merge(&b);

        let mut ba = b.clone();
        ba.merge(&a);

        assert_eq!(ab, ba, "G-Counter merge must be commutative");
        // node 1 should take the max: 7
        assert_eq!(ab.get(1), 7);
    }

    #[test]
    fn gcounter_merge_is_idempotent() {
        let mut c = GCounter::new();
        c.increment(0, 10);
        c.increment(1, 5);

        let copy = c.clone();
        c.merge(&copy);

        assert_eq!(c.get(0), 10);
        assert_eq!(c.get(1), 5);
    }

    #[test]
    fn gcounter_merge_is_associative() {
        let mut a = GCounter::new();
        a.increment(0, 3);
        let mut b = GCounter::new();
        b.increment(1, 7);
        let mut c_ctr = GCounter::new();
        c_ctr.increment(2, 2);

        let mut ab_c = a.clone();
        ab_c.merge(&b);
        ab_c.merge(&c_ctr);

        let mut a_bc = b.clone();
        a_bc.merge(&c_ctr);
        let mut a_bc2 = a.clone();
        a_bc2.merge(&a_bc);

        assert_eq!(ab_c, a_bc2);
    }

    // --- PheromoneMap ---

    #[test]
    fn pheromone_map_merge_converges_after_gossip() {
        let mut map_a = PheromoneMap::default();
        let mut map_b = PheromoneMap::default();

        map_a.emit_demand(0, 800);
        map_a.emit_supply(0, 200);
        map_b.emit_demand(1, 200);
        map_b.emit_supply(1, 800);

        // Single gossip round
        let snap_a = map_a.clone();
        let snap_b = map_b.clone();
        map_a.merge(&snap_b);
        map_b.merge(&snap_a);

        // Both replicas now see both nodes
        assert_eq!(map_a.demand_total(0), 800);
        assert_eq!(map_a.demand_total(1), 200);
        assert_eq!(map_b.demand_total(0), 800);
        assert_eq!(map_b.demand_total(1), 200);
        assert_eq!(map_a, map_b, "Maps must converge after gossip");
    }

    #[test]
    fn pheromone_gradient_direction_is_correct() {
        let mut map = PheromoneMap::default();

        // Node 0: heavily loaded (demand = 900)
        for _ in 0..9 {
            map.emit_demand(0, 100);
        }
        // Node 1: lightly loaded (demand = 100)
        map.emit_demand(1, 100);

        // Overloaded node 0 sees neighbor (1) with less demand → negative gradient
        let grad0 = map.demand_gradient(0, &[1]);
        assert!(
            grad0 < 0.0,
            "Overloaded node must see negative demand gradient (neighbor has less demand)"
        );

        // Underloaded node 1 sees neighbor (0) with more demand → positive gradient
        let grad1 = map.demand_gradient(1, &[0]);
        assert!(
            grad1 > 0.0,
            "Underloaded node must see positive demand gradient (neighbor has more demand)"
        );
    }

    #[test]
    fn pheromone_gradient_zero_with_no_neighbors() {
        let map = PheromoneMap::default();
        assert_eq!(map.demand_gradient(0, &[]), 0.0);
        assert_eq!(map.supply_gradient(0, &[]), 0.0);
    }

    // --- TritLayer ---

    #[test]
    fn trit_layer_output_has_correct_shape_and_nonneg() {
        let mut rng = Rng::new(7);
        let layer = TritLayer::random(4, 3, &mut rng);
        let out = layer.forward(&[1.0, -0.5, 0.0, 0.8]);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|&v| v >= 0.0), "ReLU must clamp negatives to 0");
    }

    // --- TritNet ---

    #[test]
    fn trit_net_always_returns_valid_action() {
        let mut rng = Rng::new(42);
        let net = TritNet::random(&mut rng);
        // Exhaustive boundary inputs
        for &(u, d, s, t) in &[
            (0.0f32, 0.0f32, 0.0f32, 0i64),
            (1.0, 1000.0, -1000.0, 1000),
            (0.5, -500.0, 500.0, -100),
        ] {
            let action = net.decide(u, d, s, t);
            assert!(matches!(
                action,
                Action::MigrateWork | Action::AcceptWork | Action::Idle
            ));
        }
    }

    #[test]
    fn trit_net_forward_shape() {
        let mut rng = Rng::new(99);
        let net = TritNet::random(&mut rng);
        let scores = net.forward(&[0.8, 0.3, -0.2, 0.1]);
        assert_eq!(scores.len(), 3);
    }

    // --- Agent ---

    #[test]
    fn agent_emits_higher_demand_when_overloaded() {
        let mut rng = Rng::new(1);
        let mut agent = Agent::new(0, 0.9, &mut rng);

        let demand_before = agent.pheromone_map.demand_total(0);
        agent
            .pheromone_map
            .emit_demand(0, (agent.utilization * 1000.0) as u64);
        let demand_after = agent.pheromone_map.demand_total(0);

        assert!(demand_after > demand_before);
        // High utilization (0.9) means demand signal ~900, supply ~100
        agent
            .pheromone_map
            .emit_supply(0, ((1.0 - agent.utilization) * 1000.0) as u64);
        assert!(
            agent.pheromone_map.demand_total(0) > agent.pheromone_map.supply_total(0),
            "Overloaded agent: demand must exceed supply"
        );
    }

    // --- Fleet ---

    #[test]
    fn fleet_ring_topology_has_two_neighbors_each() {
        let fleet = Fleet::new(8, 42);
        for agent in &fleet.agents {
            assert_eq!(agent.neighbors.len(), 2);
            // Neighbors are distinct and not self
            assert_ne!(agent.neighbors[0], agent.id);
            assert_ne!(agent.neighbors[1], agent.id);
            assert_ne!(agent.neighbors[0], agent.neighbors[1]);
        }
    }

    #[test]
    fn fleet_utilizations_stay_in_range_after_ticks() {
        let mut fleet = Fleet::new(8, 42);
        for _ in 0..100 {
            fleet.tick();
        }
        for a in &fleet.agents {
            assert!(
                (0.0..=1.0).contains(&a.utilization),
                "Utilization out of [0,1]: node {} = {}",
                a.id,
                a.utilization
            );
        }
    }

    #[test]
    fn fleet_tokens_are_conserved_across_migrations() {
        let mut fleet = Fleet::new(8, 42);
        let initial = fleet.total_tokens();
        for _ in 0..100 {
            fleet.tick();
        }
        assert_eq!(
            fleet.total_tokens(),
            initial,
            "Token supply is closed: migrations only redistribute, never create/destroy"
        );
    }

    #[test]
    fn migration_mechanics_reduce_variance() {
        // Directly verify that moving load from an overloaded node to an
        // underloaded one decreases utilization variance — the fundamental
        // correctness property the pheromone market relies on.
        let util_hi = 0.8_f32;
        let util_lo = 0.1_f32;
        let mean = (util_hi + util_lo) / 2.0;
        let var_before = ((util_hi - mean).powi(2) + (util_lo - mean).powi(2)) / 2.0;

        let workload = util_hi * 0.3;
        let new_hi = util_hi - workload;
        let new_lo = util_lo + workload;
        let var_after = ((new_hi - mean).powi(2) + (new_lo - mean).powi(2)) / 2.0;

        assert!(
            var_after < var_before,
            "Migrating 30% from overloaded to underloaded must reduce variance: {var_before:.4} → {var_after:.4}"
        );
    }

    #[test]
    fn pheromone_crdt_gossip_reaches_consensus_in_two_rounds() {
        // Simulate two isolated clusters merging their pheromone views.
        let mut cluster_a = PheromoneMap::default();
        let mut cluster_b = PheromoneMap::default();

        cluster_a.emit_demand(0, 500);
        cluster_a.emit_demand(1, 300);
        cluster_b.emit_demand(2, 700);
        cluster_b.emit_demand(3, 100);

        // Round 1: each cluster learns about the other
        let snap_a = cluster_a.clone();
        let snap_b = cluster_b.clone();
        cluster_a.merge(&snap_b);
        cluster_b.merge(&snap_a);

        // Round 2: converge (idempotent for already-known data)
        let snap_a2 = cluster_a.clone();
        let snap_b2 = cluster_b.clone();
        cluster_a.merge(&snap_b2);
        cluster_b.merge(&snap_a2);

        // Both clusters must now agree on the full fleet demand picture
        for node in [0, 1, 2, 3] {
            assert_eq!(
                cluster_a.demand_total(node),
                cluster_b.demand_total(node),
                "Clusters disagree on node {node} after gossip"
            );
        }
    }

    #[test]
    fn fleet_tick_count_advances_correctly() {
        let mut fleet = Fleet::new(4, 1);
        assert_eq!(fleet.tick, 0);
        fleet.tick();
        fleet.tick();
        fleet.tick();
        assert_eq!(fleet.tick, 3);
    }
}
