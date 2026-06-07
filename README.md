# persistent-agent

> **Your agent has a topological fingerprint. Find it.**

[![crates.io](https://img.shields.io/crates/v/persistent-agent.svg)](https://crates.io/crates/persistent-agent)
[![docs.rs](https://docs.rs/persistent-agent/badge.svg)](https://docs.rs/persistent-agent)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust library applying persistent homology to agent behavior analysis. Embeds agent actions as point clouds, constructs Vietoris-Rips complexes, extracts persistence barcodes, and classifies agent archetypes by their topological signatures. Gives every agent a **topological fingerprint** — a mathematically rigorous summary of its behavioral structure.

---

## Table of Contents

- [What is Persistent Homology?](#what-is-persistent-homology)
- [Why Does This Matter?](#why-does-this-matter)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Related Crates](#related-crates)
- [License](#license)

---

## What is Persistent Homology?

**Persistent homology** is a method from computational topology that quantifies the shape of data at multiple scales. Given a point cloud, it builds a sequence of simplicial complexes by connecting points within increasing distance thresholds, then tracks which topological features (connected components, loops, voids) **persist** across scales.

The output is a **barcode** — a collection of intervals [birth, death), one for each topological feature:

```
ε scale:    0    0.5    1.0    1.5    2.0    2.5    3.0
            |     |      |      |      |      |      |
H₀:     █═══════════════════════════════════════     (component 1)
            █████████████████                      (component 2 merges at ε=1.0)
            █████████                              (component 3 merges at ε=0.8)
H₁:                              █══════════       (loop appears at ε=1.2, fills at ε=2.0)
```

Long bars = robust features. Short bars = noise. For agent behavior:

- **H₀** (connected components) = distinct behavioral modes
- **H₁** (loops) = cyclic behavioral patterns (routines)
- **H₂** (voids) = complex recurring structures

Different agent types have different topological fingerprints — this library detects them.

## Why Does This Matter?

**For behavioral analysis**: Persistent homology provides a multiscale, coordinate-free summary of agent behavior. It captures structure that clustering and PCA miss — loops, voids, and their persistence.

**For anomaly detection**: An agent whose topological fingerprint suddenly changes has changed its behavior. This is a robust, parameter-free anomaly signal.

**For agent classification**: The bottleneck distance between persistence diagrams provides a principled metric for comparing agent behaviors — the basis for archetype classification.

**For cognitive science**: The Dream Cycle uses persistent homology to analyze behavioral patterns during sleep — the cortex "dreams" about the topological structure of the agent's actions.

## Architecture

```
persistent-agent
│
├── PointCloud                 ← Agent action embeddings
│   ├── new()                      Empty point cloud
│   ├── add_point(coords, label)   Add labeled observation
│   ├── distance_matrix()          Full pairwise distance matrix
│   └── sorted_distances()         All unique distances, sorted
│
├── VietorisRips               ← Complex construction
│   ├── build(cloud, max_dim)      Full filtered complex
│   └── build_at_threshold()       Complex at fixed ε
│
├── BoundaryMatrix             ← Homology computation
│   ├── from_complex(complex, dim) Boundary for dimension k
│   └── rank()                     Rank of boundary matrix
│
├── Barcode                    ← Persistence output
│   ├── add_bar(birth, death)      Add persistence interval
│   ├── betti(threshold)           Betti number at scale ε
│   ├── num_bars() / num_essential() Feature counts
│   └── visualize(width)           ASCII barcode plot
│
├── PersistenceDiagram         ← Point-cloud representation
│   ├── from_barcode(barcode)      Convert barcode to diagram
│   ├── bottleneck_distance(other) Wasserstein-∞ distance
│   └── total_persistence()        Σ |death − birth|
│
└── AgentArchetype             ← Classification
    ├── Steady / Explorer / Volatile / Cyclic / Complex
    └── classify_agent(barcodes)   Topological fingerprint → type
```

## Quick Start

```rust
use persistent_agent::{
    PointCloud, VietorisRips, compute_barcodes,
    classify_agent,
};

// Embed agent actions as a point cloud
let mut cloud = PointCloud::new();
cloud.add_point(vec![1.0, 0.0, 0.0], "action_a".into());
cloud.add_point(vec![1.1, 0.1, 0.0], "action_a'".into());
cloud.add_point(vec![0.0, 1.0, 0.0], "action_b".into());
cloud.add_point(vec![0.0, 0.0, 1.0], "action_c".into());
cloud.add_point(vec![0.9, 0.0, 0.1], "action_a''".into());

// Build Vietoris-Rips complex (up to dimension 2)
let complex = VietorisRips::build(&cloud, 2);

// Compute persistence barcodes for H₀ and H₁
let barcodes = compute_barcodes(&complex, 2);

// Visualize the barcodes
for (dim, barcode) in barcodes.iter().enumerate() {
    println!("H{} barcode:", dim);
    println!("{}", barcode.visualize(60));
    println!("Betti number at ε=1.0: {}", barcode.betti(1.0));
}

// Classify the agent by its topological fingerprint
let archetype = classify_agent(&barcodes);
println!("Agent archetype: {:?}", archetype);
// → Steady, Explorer, Volatile, Cyclic, or Complex

// Compare two agents via bottleneck distance
let diagram_1 = persistent_agent::PersistenceDiagram::from_barcode(&barcodes[0]);
// ... create diagram_2 from another agent ...
// let dist = diagram_1.bottleneck_distance(&diagram_2);
```

## API Reference

### PointCloud

| Method | Returns | Description |
|--------|---------|-------------|
| `new()` | `Self` | Empty point cloud |
| `add_point(coords, label)` | `()` | Add labeled point |
| `len()` | `usize` | Number of points |
| `distance_matrix()` | `Vec<Vec<f64>>` | Pairwise distances |
| `sorted_distances()` | `Vec<f64>` | Unique sorted distances |

### VietorisRips

| Method | Returns | Description |
|--------|---------|-------------|
| `build(cloud, max_dim)` | `FilteredComplex` | Full filtered complex |
| `build_at_threshold(cloud, max_dim, ε)` | `FilteredComplex` | Complex at fixed scale |

### Barcode

| Method | Returns | Description |
|--------|---------|-------------|
| `new(dim)` | `Self` | Empty barcode for dimension dim |
| `add_bar(birth, death)` | `()` | Add persistence interval |
| `betti(threshold)` | `usize` | Betti number at scale ε |
| `num_bars()` | `usize` | Number of intervals |
| `num_essential()` | `usize` | Number of infinite bars |
| `visualize(width)` | `String` | ASCII barcode plot |

### PersistenceDiagram

| Method | Returns | Description |
|--------|---------|-------------|
| `from_barcode(barcode)` | `Self` | Convert barcode to diagram |
| `bottleneck_distance(&other)` | `f64` | Wasserstein-∞ distance |
| `total_persistence()` | `f64` | Σ \|death − birth\| |

### AgentArchetype

| Variant | Description |
|---------|-------------|
| `Steady` | Few components, no loops — reliable, boring |
| `Explorer` | Many components, loops — visits many states |
| `Volatile` | High H₀ turnover — unstable behavior |
| `Cyclic` | Persistent H₁ — strong routine patterns |
| `Complex` | H₂ features — multi-dimensional recurring structures |

## Mathematical Background

### Vietoris-Rips Complex

Given a point cloud X = {x₁, ..., xₙ} and threshold ε, the Vietoris-Rips complex VR(X, ε) has:
- A k-simplex [x_{i₀}, ..., x_{iₖ}] whenever all pairwise distances d(x_{iₐ}, x_{i_b}) ≤ ε

The **filtered complex** tracks all simplices with their birth thresholds:

```
ε = 0:     {all vertices}                          (no edges)
ε = d_min: {all vertices} ∪ {shortest edge}        (first edge appears)
ε = d_max: {all vertices} ∪ {all edges} ∪ {all triangles} ∪ ...
```

### Persistent Homology

As ε increases from 0 to ∞, we track how homology groups change:

```
H_k(ε₁) → H_k(ε₂) → ... → H_k(∞) = {0}
```

A feature born at ε₁ and dying at ε₂ creates a bar [ε₁, ε₂). The **persistence** ε₂ − ε₁ measures robustness.

### Barcode and Persistence Diagram

- **Barcode**: multiset of intervals [birth, death)
- **Persistence diagram**: multiset of points (birth, death) in ℝ²

The **bottleneck distance** between two diagrams D₁, D₂:

```
d_B(D₁, D₂) = inf_{bijections φ} sup_{p ∈ D₁} ||p − φ(p)||_∞
```

This is the ∞-Wasserstein distance on persistence diagrams, stable under perturbations of the input data (Stability Theorem, Cohen-Steiner et al. 2007).

### Boundary Matrix Reduction

Homology is computed by **column reduction** on the boundary matrix ∂ₖ:

```
∂ₖ : Cₖ → C_{k−1}
```

Each column is reduced to its pivot form. The pivot positions determine which features are born and which die — this is the standard persistent homology algorithm.

## Installation

```bash
cargo add persistent-agent
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
persistent-agent = "0.1"
```

## Related Crates

Part of the **SuperInstance Exocortex** math fleet:

- **[graph-homology](https://github.com/SuperInstance/graph-homology)** — Clique complexes and Betti numbers of graphs
- **[cohomology-ring](https://github.com/SuperInstance/cohomology-ring)** — Cup products and cohomology operations
- **[sheaf-laplacian](https://github.com/SuperInstance/sheaf-laplacian)** — Sheaf Laplacian and Hodge decomposition
- **[tropical-graph](https://github.com/SuperInstance/tropical-graph)** — Max-plus algebra on graphs
- **[dream-cycle](https://github.com/SuperInstance/dream-cycle)** — Sleep consolidation for agent memory

## License

MIT © [SuperInstance](https://github.com/SuperInstance)

Part of the [Exocortex](https://github.com/SuperInstance/exocortex) project.
