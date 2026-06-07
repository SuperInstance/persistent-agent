# persistent-agent

> **Your agent has a topological fingerprint. Find it.**

[![crates.io](https://img.shields.io/crates/v/persistent-agent.svg)](https://crates.io/crates/persistent-agent)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Persistent homology for agent behavior analysis. Embeds agent actions as point clouds, builds Vietoris-Rips complexes, extracts barcodes, and classifies agent archetypes by their persistence signatures.

## The Idea

Every agent leaves a **topological trace** — a pattern in the space of its actions. Persistent homology captures this pattern:
- **H₀** (connected components): how many distinct behavior modes?
- **H₁** (loops): does the agent cycle through behaviors?
- **H₂** (voids): are there complex recurring structures?

Different agent types have different topological fingerprints:
- **Steady**: few components, no loops (boring but reliable)
- **Explorer**: many components, loops (visits many states)
- **Volatile**: high H₀ turnover (unstable behavior)

## Part of [Exocortex](https://github.com/SuperInstance/exocortex)

Used in the Dream Cycle for behavioral analysis — the cortex dreams about agent patterns and extracts archetypes.

## License

MIT © [SuperInstance](https://github.com/SuperInstance)
