//! # persistent-agent
//!
//! Persistent homology for agent behavior analysis. Constructs Vietoris-Rips
//! complexes from agent embedding point clouds, computes barcodes and
//! persistence diagrams, and classifies agent archetypes from their
//! topological signatures.

use std::collections::HashMap;

/// Euclidean distance between two points.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

/// A point cloud representing agent embeddings.
#[derive(Clone, Debug)]
pub struct PointCloud {
    /// Points as vectors of coordinates.
    pub points: Vec<Vec<f64>>,
    /// Labels for each point (e.g., agent IDs).
    pub labels: Vec<String>,
}

impl PointCloud {
    /// Create a new empty point cloud.
    pub fn new() -> Self {
        Self { points: Vec::new(), labels: Vec::new() }
    }

    /// Add a labeled point.
    pub fn add_point(&mut self, coords: Vec<f64>, label: String) {
        self.points.push(coords);
        self.labels.push(label);
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Is the cloud empty?
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute the distance matrix.
    pub fn distance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.points.len();
        let mut dm = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = euclidean_distance(&self.points[i], &self.points[j]);
                dm[i][j] = d;
                dm[j][i] = d;
            }
        }
        dm
    }

    /// Get all unique pairwise distances, sorted.
    pub fn sorted_distances(&self) -> Vec<f64> {
        let dm = self.distance_matrix();
        let mut dists: Vec<f64> = Vec::new();
        for i in 0..dm.len() {
            for j in (i + 1)..dm.len() {
                dists.push(dm[i][j]);
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists
    }
}

/// A simplex (subset of vertices).
pub type Simplex = Vec<usize>;

/// A filtered simplicial complex.
#[derive(Clone, Debug)]
pub struct FilteredComplex {
    /// Simplices with their birth distances.
    pub simplices: Vec<(Simplex, f64)>,
    /// Number of vertices.
    pub num_vertices: usize,
}

/// Vietoris-Rips complex builder.
pub struct VietorisRips;

impl VietorisRips {
    /// Build a Vietoris-Rips complex up to dimension `max_dim` for the given point cloud.
    pub fn build(cloud: &PointCloud, max_dim: usize) -> FilteredComplex {
        let dm = cloud.distance_matrix();
        let n = cloud.points.len();
        let mut simplices: Vec<(Simplex, f64)> = Vec::new();

        // Add vertices (0-simplices) at distance 0
        for i in 0..n {
            simplices.push((vec![i], 0.0));
        }

        // Add edges (1-simplices)
        for i in 0..n {
            for j in (i + 1)..n {
                simplices.push((vec![i, j], dm[i][j]));
            }
        }

        // Build higher simplices up to max_dim
        if max_dim >= 2 {
            for dim in 2..=max_dim {
                let vertices: Vec<usize> = (0..n).collect();
                for combo in combinations(&vertices, dim + 1) {
                    let mut md: f64 = 0.0;
                    for ci in 0..combo.len() {
                        for cj in (ci + 1)..combo.len() {
                            md = f64::max(md, dm[combo[ci]][combo[cj]]);
                        }
                    }
                    let mut s = combo.clone();
                    s.sort();
                    simplices.push((s, md));
                }
            }
        }

        // Sort by (birth distance, dimension, simplex)
        simplices.sort_by(|a, b| {
            match a.1.partial_cmp(&b.1) {
                Some(std::cmp::Ordering::Equal) | None => {
                    match a.0.len().cmp(&b.0.len()) {
                        std::cmp::Ordering::Equal => a.0.cmp(&b.0),
                        o => o,
                    }
                }
                Some(o) => o,
            }
        });

        FilteredComplex { simplices, num_vertices: n }
    }

    /// Build complex at a specific threshold radius.
    pub fn build_at_threshold(cloud: &PointCloud, max_dim: usize, threshold: f64) -> FilteredComplex {
        let full = Self::build(cloud, max_dim);
        let simplices: Vec<(Simplex, f64)> = full.simplices
            .into_iter()
            .filter(|(_, birth)| *birth <= threshold)
            .collect();
        FilteredComplex { simplices, num_vertices: full.num_vertices }
    }
}

/// Generate all k-combinations of a slice.
fn combinations(data: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 { return vec![vec![]]; }
    if k > data.len() { return vec![]; }
    let mut result = Vec::new();
    for i in 0..=data.len() - k {
        for mut tail in combinations(&data[i + 1..], k - 1) {
            let mut combo = vec![data[i]];
            combo.append(&mut tail);
            result.push(combo);
        }
    }
    result
}

/// Boundary matrix (mod 2).
#[derive(Clone, Debug)]
pub struct BoundaryMatrix {
    /// Number of rows (lower-dim simplices).
    pub rows: usize,
    /// Number of columns (higher-dim simplices).
    pub cols: usize,
    /// Non-zero entries stored as (row, col) pairs.
    pub entries: Vec<Vec<bool>>,
}

impl BoundaryMatrix {
    /// Create a zero boundary matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, entries: vec![vec![false; cols]; rows] }
    }

    /// Set entry.
    pub fn set(&mut self, i: usize, j: usize, val: bool) {
        self.entries[i][j] = val;
    }

    /// Get entry.
    pub fn get(&self, i: usize, j: usize) -> bool {
        self.entries[i][j]
    }

    /// Build boundary matrix from a filtered complex for a given dimension.
    /// Columns = dim-simplices, rows = (dim-1)-simplices.
    pub fn from_complex(complex: &FilteredComplex, dim: usize) -> Self {
        let lower: Vec<&Simplex> = complex.simplices.iter()
            .filter(|(s, _)| s.len() == dim)
            .map(|(s, _)| s)
            .collect();
        let upper: Vec<&Simplex> = complex.simplices.iter()
            .filter(|(s, _)| s.len() == dim + 1)
            .map(|(s, _)| s)
            .collect();

        let mut bm = Self::zeros(lower.len(), upper.len());

        // Build index map for lower simplices
        let mut lower_idx: HashMap<Vec<usize>, usize> = HashMap::new();
        for (i, s) in lower.iter().enumerate() {
            lower_idx.insert((*s).clone(), i);
        }

        for (col, simplex) in upper.iter().enumerate() {
            // Boundary: omit each vertex one at a time
            for skip in 0..simplex.len() {
                let mut face: Vec<usize> = (*simplex).clone();
                face.remove(skip);
                face.sort();
                if let Some(&row) = lower_idx.get(&face) {
                    bm.entries[row][col] = !bm.entries[row][col]; // mod 2
                }
            }
        }
        bm
    }

    /// Compute rank using row reduction (mod 2).
    pub fn rank(&self) -> usize {
        let mut matrix = self.entries.clone();
        let mut rank = 0;
        for col in 0..self.cols {
            // Find pivot row
            let pivot = (rank..self.rows).find(|&r| matrix[r][col]);
            if let Some(p) = pivot {
                // Swap
                matrix.swap(rank, p);
                // Eliminate
                for row in 0..self.rows {
                    if row != rank && matrix[row][col] {
                        for c in 0..self.cols {
                            matrix[row][c] ^= matrix[rank][c];
                        }
                    }
                }
                rank += 1;
            }
        }
        rank
    }
}

/// A barcode: collection of (birth, death) pairs for homology features.
#[derive(Clone, Debug)]
pub struct Barcode {
    /// Dimension of the homology.
    pub dim: usize,
    /// (birth, death) pairs. death = f64::INFINITY for essential features.
    pub bars: Vec<(f64, f64)>,
}

impl Barcode {
    /// Create a new barcode.
    pub fn new(dim: usize) -> Self {
        Self { dim, bars: Vec::new() }
    }

    /// Add a bar.
    pub fn add_bar(&mut self, birth: f64, death: f64) {
        self.bars.push((birth, death));
    }

    /// Number of bars.
    pub fn num_bars(&self) -> usize {
        self.bars.len()
    }

    /// Number of essential (infinite) bars.
    pub fn num_essential(&self) -> usize {
        self.bars.iter().filter(|(_, d)| d.is_infinite()).count()
    }

    /// Betti number: number of bars alive at a given threshold.
    pub fn betti(&self, threshold: f64) -> usize {
        self.bars.iter()
            .filter(|(b, d)| *b <= threshold && (d.is_infinite() || *d > threshold))
            .count()
    }

    /// Format as ASCII barcode visualization.
    pub fn visualize(&self, max_width: usize) -> String {
        if self.bars.is_empty() { return String::new(); }
        let max_birth = self.bars.iter().map(|(b, _)| *b).fold(0.0_f64, f64::max);
        let max_death = self.bars.iter()
            .filter(|(_, d)| d.is_finite())
            .map(|(_, d)| *d)
            .fold(0.0_f64, f64::max);
        let max_val = max_birth.max(max_death);
        if max_val == 0.0 { return String::new(); }

        let mut lines = Vec::new();
        for (b, d) in &self.bars {
            let start = (*b / max_val * max_width as f64) as usize;
            let end = if d.is_infinite() {
                max_width
            } else {
                (*d / max_val * max_width as f64) as usize
            };
            let mut line = " ".repeat(start);
            line += &"|".repeat((end - start).max(1));
            if d.is_infinite() { line += "→∞"; }
            lines.push(line);
        }
        lines.join("\n")
    }
}

/// A persistence diagram: points in (birth, death) space.
#[derive(Clone, Debug)]
pub struct PersistenceDiagram {
    pub dim: usize,
    pub points: Vec<(f64, f64)>,
}

impl PersistenceDiagram {
    /// Create from a barcode.
    pub fn from_barcode(barcode: &Barcode) -> Self {
        Self {
            dim: barcode.dim,
            points: barcode.bars.clone(),
        }
    }

    /// Bottleneck distance to another persistence diagram of the same dimension.
    pub fn bottleneck_distance(&self, other: &PersistenceDiagram) -> f64 {
        // Simplified: use the matching that pairs points greedily by distance.
        // Full implementation would use the Hungarian algorithm.
        let n = self.points.len();
        let m = other.points.len();

        if n == 0 && m == 0 { return 0.0; }
        if n == 0 || m == 0 {
            // All points must be matched to the diagonal
            let pts = if n > 0 { &self.points } else { &other.points };
            return pts.iter()
                .map(|(b, d)| (d - b) / 2.0)
                .fold(0.0_f64, f64::max);
        }

        // Compute cost matrix: L∞ distance between points
        let mut costs = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                costs[i][j] = (self.points[i].0 - other.points[j].0).abs()
                    .max((self.points[i].1 - other.points[j].1).abs());
            }
        }

        // Greedy matching (approximate bottleneck distance)
        let mut matched_a = vec![false; n];
        let mut matched_b = vec![false; m];
        let mut max_cost = 0.0;

        // Sort all pairs by cost
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in 0..m {
                pairs.push((costs[i][j], i, j));
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (cost, i, j) in &pairs {
            if !matched_a[*i] && !matched_b[*j] {
                matched_a[*i] = true;
                matched_b[*j] = true;
                max_cost = f64::max(max_cost, *cost);
            }
        }

        // Account for unmatched points (matched to diagonal)
        for i in 0..n {
            if !matched_a[i] {
                max_cost = f64::max(max_cost, (self.points[i].1 - self.points[i].0) / 2.0);
            }
        }
        for j in 0..m {
            if !matched_b[j] {
                max_cost = f64::max(max_cost, (other.points[j].1 - other.points[j].0) / 2.0);
            }
        }

        max_cost
    }

    /// Total persistence: sum of |death - birth| for all features.
    pub fn total_persistence(&self) -> f64 {
        self.points.iter()
            .filter(|(_b, d)| d.is_finite())
            .map(|(b, d)| (d - b).abs())
            .sum()
    }
}

/// Agent archetype classification based on persistence signatures.
#[derive(Clone, Debug, PartialEq)]
pub enum AgentArchetype {
    /// Clustered: many persistent H0 features (distinct clusters).
    Clustered,
    /// Connected: single dominant H0, no higher features (well-connected).
    Connected,
    /// Looped: persistent H1 features (cyclic behavior patterns).
    Looped,
    /// Complex: significant H1 and H2 features (complex topology).
    Complex,
    /// Sparse: few features, low persistence.
    Sparse,
}

/// Classify agent behavior from persistence barcodes.
pub fn classify_agent(barcodes: &[Barcode]) -> AgentArchetype {
    let h0 = barcodes.iter().find(|b| b.dim == 0);
    let h1 = barcodes.iter().find(|b| b.dim == 1);
    let h2 = barcodes.iter().find(|b| b.dim == 2);

    let h1_bars = h1.map(|b| b.num_bars()).unwrap_or(0);
    let h2_bars = h2.map(|b| b.num_bars()).unwrap_or(0);
    let h0_essential = h0.map(|b| b.num_essential()).unwrap_or(0);

    if h2_bars > 0 { return AgentArchetype::Complex; }
    if h1_bars > 0 { return AgentArchetype::Looped; }
    if h0_essential > 2 { return AgentArchetype::Clustered; }
    let total_bars: usize = barcodes.iter().map(|b| b.num_bars()).sum();
    if h2_bars > 0 { return AgentArchetype::Complex; }
    if h1_bars > 0 { return AgentArchetype::Looped; }
    if h0_essential > 2 { return AgentArchetype::Clustered; }
    // Single essential H0 bar with no other features = connected but simple
    if h0_essential == 1 && total_bars == 1 { return AgentArchetype::Connected; }
    if total_bars == 0 { return AgentArchetype::Sparse; }
    AgentArchetype::Connected
}

/// Compute barcodes for a filtered complex using a simplified algorithm.
pub fn compute_barcodes(complex: &FilteredComplex, max_dim: usize) -> Vec<Barcode> {
    let mut barcodes = Vec::new();

    for dim in 0..=max_dim {
        if dim == 0 {
            // H0: each vertex born at 0, dies when merged
            let mut bc = Barcode::new(0);
            let n = complex.num_vertices;
            // Use union-find for H0
            let mut parent: Vec<usize> = (0..n).collect();
            let birth: Vec<f64> = vec![0.0; n];

            fn find_root(parent: &mut Vec<usize>, i: usize) -> usize {
                if parent[i] != i {
                    parent[i] = find_root(parent, parent[i]);
                }
                parent[i]
            }

            let mut num_components = n;

            // Process edges in order
            let mut edges: Vec<(usize, usize, f64)> = complex.simplices.iter()
                .filter(|(s, _)| s.len() == 2)
                .map(|(s, d)| (s[0], s[1], *d))
                .collect();
            edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            for (u, v, dist) in edges {
                let ru = find_root(&mut parent, u);
                let rv = find_root(&mut parent, v);
                if ru != rv {
                    // Merge: younger component dies
                    let (older, younger) = if birth[ru] <= birth[rv] { (ru, rv) } else { (rv, ru) };
                    parent[younger] = older;
                    bc.add_bar(birth[younger], dist);
                    num_components -= 1;
                }
            }

            // Remaining components are essential
            let mut root_set: Vec<usize> = Vec::new();
            for i in 0..n {
                let r = find_root(&mut parent, i);
                if !root_set.contains(&r) {
                    root_set.push(r);
                }
            }
            for r in &root_set {
                bc.add_bar(birth[*r], f64::INFINITY);
            }
            barcodes.push(bc);
        } else {
            // Higher dimensions: use boundary matrix rank
            let bm = BoundaryMatrix::from_complex(complex, dim);
            let rank = bm.rank();
            let num_upper: usize = complex.simplices.iter()
                .filter(|(s, _)| s.len() == dim + 1)
                .count();

            let betti = num_upper.saturating_sub(rank);

            let mut bc = Barcode::new(dim);
            let upper_simplices: Vec<(Simplex, f64)> = complex.simplices.iter()
                .filter(|(s, _)| s.len() == dim + 1)
                .map(|(s, d)| (s.clone(), *d))
                .collect();

            if betti > 0 && !upper_simplices.is_empty() {
                for i in 0..betti.min(upper_simplices.len()) {
                    let birth = upper_simplices[i].1;
                    bc.add_bar(birth, f64::INFINITY);
                }
            }
            barcodes.push(bc);
        }
    }

    barcodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_creation() {
        let mut cloud = PointCloud::new();
        cloud.add_point(vec![0.0, 0.0], "a".into());
        cloud.add_point(vec![1.0, 0.0], "b".into());
        cloud.add_point(vec![0.0, 1.0], "c".into());
        assert_eq!(cloud.len(), 3);
    }

    #[test]
    fn test_distance_matrix() {
        let mut cloud = PointCloud::new();
        cloud.add_point(vec![0.0], "a".into());
        cloud.add_point(vec![3.0], "b".into());
        cloud.add_point(vec![6.0], "c".into());
        let dm = cloud.distance_matrix();
        assert!((dm[0][1] - 3.0).abs() < 1e-10);
        assert!((dm[0][2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sorted_distances() {
        let mut cloud = PointCloud::new();
        cloud.add_point(vec![0.0], "a".into());
        cloud.add_point(vec![1.0], "b".into());
        cloud.add_point(vec![3.0], "c".into());
        let dists = cloud.sorted_distances();
        assert_eq!(dists.len(), 3);
        assert!((dists[0] - 1.0).abs() < 1e-10);
        assert!((dists[1] - 2.0).abs() < 1e-10);
        assert!((dists[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vietoris_rips_triangle() {
        let mut cloud = PointCloud::new();
        cloud.add_point(vec![0.0, 0.0], "a".into());
        cloud.add_point(vec![1.0, 0.0], "b".into());
        cloud.add_point(vec![0.5, 0.866], "c".into());
        let complex = VietorisRips::build(&cloud, 2);
        // Should have 3 vertices + 3 edges + 1 triangle = 7 simplices
        assert_eq!(complex.simplices.len(), 7);
    }

    #[test]
    fn test_vietoris_rips_threshold() {
        let mut cloud = PointCloud::new();
        cloud.add_point(vec![0.0], "a".into());
        cloud.add_point(vec![1.0], "b".into());
        cloud.add_point(vec![10.0], "c".into());
        let complex = VietorisRips::build_at_threshold(&cloud, 2, 5.0);
        // c should be isolated, only a-b edge
        let edges: Vec<_> = complex.simplices.iter().filter(|(s, _)| s.len() == 2).collect();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_boundary_matrix_triangle() {
        let mut cloud = PointCloud::new();
        cloud.add_point(vec![0.0, 0.0], "a".into());
        cloud.add_point(vec![1.0, 0.0], "b".into());
        cloud.add_point(vec![0.0, 1.0], "c".into());
        let complex = VietorisRips::build(&cloud, 2);
        assert_eq!(complex.simplices.len(), 7);
        
        // dim=1: lower = edges (len 2), upper = triangles (len 3)
        // Actually: lower = simplices of len dim=2 (edges), upper = simplices of len dim+1=3 (triangles)
        // Wait no: from_complex(complex, dim=1) means lower = len(dim)=1 (vertices), upper = len(dim+1)=2 (edges)
        let bm = BoundaryMatrix::from_complex(&complex, 1);
        // dim=1: rows = 1-simplices (vertices, len=1 -> wait no...)
        // Actually: lower = simplices of size dim=1, upper = simplices of size dim+1=2
        // lower (vertices, size 1) = 3, upper (edges, size 2) = 3
        assert_eq!(bm.rows, 3);
        assert_eq!(bm.cols, 3);
        // Boundary of each edge: edge (i,j) has boundary vertices {i} and {j}
        // So each column should have exactly 2 non-zero entries (the two endpoints)
        for col in 0..3 {
            let nnz: usize = (0..3).filter(|&row| bm.get(row, col)).count();
            assert_eq!(nnz, 2);
        }
    }

    #[test]
    fn test_boundary_matrix_rank() {
        let mut bm = BoundaryMatrix::zeros(3, 2);
        bm.set(0, 0, true);
        bm.set(1, 0, true);
        bm.set(1, 1, true);
        bm.set(2, 1, true);
        assert_eq!(bm.rank(), 2);
    }

    #[test]
    fn test_barcode_betti() {
        let mut bc = Barcode::new(0);
        bc.add_bar(0.0, 1.0);
        bc.add_bar(0.0, f64::INFINITY);
        assert_eq!(bc.betti(0.5), 2);
        assert_eq!(bc.betti(2.0), 1);
        assert_eq!(bc.num_essential(), 1);
    }

    #[test]
    fn test_barcode_visualize() {
        let mut bc = Barcode::new(0);
        bc.add_bar(0.0, 1.0);
        bc.add_bar(0.0, f64::INFINITY);
        let vis = bc.visualize(40);
        assert!(!vis.is_empty());
        assert!(vis.contains("→∞"));
    }

    #[test]
    fn test_persistence_diagram_bottleneck() {
        let mut bc1 = Barcode::new(0);
        bc1.add_bar(0.0, 2.0);
        bc1.add_bar(0.0, f64::INFINITY);
        let pd1 = PersistenceDiagram::from_barcode(&bc1);

        let mut bc2 = Barcode::new(0);
        bc2.add_bar(0.0, 2.1);
        bc2.add_bar(0.0, f64::INFINITY);
        let pd2 = PersistenceDiagram::from_barcode(&bc2);

        let dist = pd1.bottleneck_distance(&pd2);
        assert!(dist >= 0.0);
        assert!(dist < 1.0); // Should be small since diagrams are similar
    }

    #[test]
    fn test_agent_archetype_classification() {
        // Connected: 1 essential H0, no H1
        let mut bc0 = Barcode::new(0);
        bc0.add_bar(0.0, f64::INFINITY);
        assert_eq!(classify_agent(&[bc0]), AgentArchetype::Connected);

        // Clustered: 3+ essential H0
        let mut bc_cluster = Barcode::new(0);
        bc_cluster.add_bar(0.0, f64::INFINITY);
        bc_cluster.add_bar(0.0, f64::INFINITY);
        bc_cluster.add_bar(0.0, f64::INFINITY);
        assert_eq!(classify_agent(&[bc_cluster]), AgentArchetype::Clustered);

        // Looped: H1 features
        let mut bc0_l = Barcode::new(0);
        bc0_l.add_bar(0.0, f64::INFINITY);
        let mut bc1_l = Barcode::new(1);
        bc1_l.add_bar(1.0, f64::INFINITY);
        assert_eq!(classify_agent(&[bc0_l, bc1_l]), AgentArchetype::Looped);

        // Complex: H2 features
        let mut bc0_c = Barcode::new(0);
        bc0_c.add_bar(0.0, f64::INFINITY);
        let mut bc2_c = Barcode::new(2);
        bc2_c.add_bar(2.0, f64::INFINITY);
        assert_eq!(classify_agent(&[bc0_c, bc2_c]), AgentArchetype::Complex);
    }

    #[test]
    fn test_compute_barcodes_line() {
        let mut cloud = PointCloud::new();
        cloud.add_point(vec![0.0], "a".into());
        cloud.add_point(vec![1.0], "b".into());
        cloud.add_point(vec![2.0], "c".into());
        let complex = VietorisRips::build(&cloud, 1);
        let barcodes = compute_barcodes(&complex, 1);
        // H0 should have 1 essential bar
        let h0 = barcodes.iter().find(|b| b.dim == 0).unwrap();
        assert_eq!(h0.num_essential(), 1);
    }
}
