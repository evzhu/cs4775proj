#!/usr/bin/env python3

import random
import numpy as np
import time
from math import log
from collections import defaultdict
from scipy.special import erf, erfc

################################################
# (1) SIMULATE A (SIMPLIFIED) BIRTH-DEATH TREE
################################################
def simulate_birth_death_tree(num_taxa, birth_rate=1.0, death_rate=0.5, seed=None):
    """
    Generates a random bifurcating tree topology with random branch lengths.
    A fully rigorous birth-death simulator is complex, so we simplify:
      - Start with 1 lineage and repeatedly 'birth' new lineages until we have num_taxa.
      - Randomly merge them into a binary tree.
      - Assign random branch lengths from an exponential distribution.
    Returns:
      - parent_map: dict: {child: (parent, branch_length)}
      - leaves: list of leaf IDs (0..num_taxa-1)
      - root_id: the ID of the root
    """
    if seed is not None:
        random.seed(seed)

    # Start with a single lineage labeled 0
    nodes = [0]
    next_id = 1

    # Generate new lineages until we reach num_taxa
    while len(nodes) < num_taxa:
        nodes.append(next_id)
        next_id += 1

    # We'll now pair up nodes in a simple random way to ensure a binary tree structure
    # internal nodes will be labeled from 'num_taxa' upward
    internal_id = num_taxa
    parent_map = {}
    while len(nodes) > 1:
        # pick any two randomly
        c1 = random.choice(nodes)
        nodes.remove(c1)
        c2 = random.choice(nodes)
        nodes.remove(c2)

        # combine them into an internal node
        p = internal_id
        internal_id += 1

        # Exponential random branch lengths
        bl1 = random.expovariate(birth_rate)  # branch length to child1
        bl2 = random.expovariate(birth_rate)  # branch length to child2

        parent_map[c1] = (p, bl1)
        parent_map[c2] = (p, bl2)

        nodes.append(p)

    # the last node in 'nodes' is the root
    root_id = nodes[0]

    # leaves are simply 0..(num_taxa-1)
    leaves = list(range(num_taxa))

    return parent_map, leaves, root_id


################################################
# (2) SIMULATE SEQUENCES UNDER JUKES-CANTOR
################################################
def simulate_sequences(parent_map, leaves, root_id,
                       seq_length=500, alphabet="ACGT", mu=1e-3, seed=None):
    """
    Simulates nucleotide sequences along a given tree using a Jukes-Cantor-like model.
    For simplicity, we:
      - generate a random sequence at the root
      - propagate down the tree with a small per-branch mutation rate * branch_length
    Returns:
      - sequences: dict of form {leaf_id: "ACTG..."}
    """
    if seed is not None:
        random.seed(seed)

    # build adjacency list from parent_map for downward simulation
    children_map = defaultdict(list)
    for child, (parent, _) in parent_map.items():
        children_map[parent].append(child)

    # topological order traversal from root to leaves
    # we store sequences for each node, not just leaves
    node_sequence = {}

    def generate_root_sequence(length, alphabet):
        return "".join(random.choice(alphabet) for _ in range(length))

    # simple recursion to simulate
    def simulate_node(node, seq):
        # assign this node's sequence
        node_sequence[node] = seq
        # for each child, mutate sequence along the branch
        for child in children_map[node]:
            parent, bl = parent_map[child]
            # expected number of substitutions = mu * bl * seq_length
            child_seq = mutate_sequence(seq, mu * bl, alphabet)
            simulate_node(child, child_seq)

    # create a random root sequence
    root_seq = generate_root_sequence(seq_length, alphabet)
    simulate_node(root_id, root_seq)

    # extract sequences only for leaves
    leaf_seqs = {leaf: node_sequence[leaf] for leaf in leaves}
    return leaf_seqs


def mutate_sequence(seq, expected_substitutions, alphabet):
    """
    Under Jukes-Cantor, each site has probability p of substituting to a different base,
    where p = expected_substitutions / len(seq). This is a *very* rough approximation.
    """
    p = expected_substitutions / len(seq) if len(seq) > 0 else 0
    new_seq = []
    for base in seq:
        if random.random() < p:
            # mutate to a different random base
            new_base = random.choice(alphabet.replace(base, ""))  # pick a different base
            new_seq.append(new_base)
        else:
            new_seq.append(base)
    return "".join(new_seq)


################################################
# (3) COMPUTE PAIRWISE DISTANCES
################################################
def compute_distance_matrix(leaf_seqs, leaves):
    """
    For each pair of leaves, compute the proportion of mismatches,
    then convert to Jukes-Cantor distance: d = -3/4 * ln(1 - 4/3 * p).
    If p >= 0.75, we set distance to some large number to avoid log domain error.
    """
    n = len(leaves)
    dist_mat = np.zeros((n, n))
    leaf_idx_map = {leaf: i for i, leaf in enumerate(leaves)}

    for i in range(n):
        for j in range(i + 1, n):
            seq_i = leaf_seqs[leaves[i]]
            seq_j = leaf_seqs[leaves[j]]
            p_mismatch = sum(a != b for a, b in zip(seq_i, seq_j)) / len(seq_i)
            if p_mismatch < 0.75:
                dist = -0.75 * log(1 - (4.0/3.0)*p_mismatch)
            else:
                dist = 2.0  # large distance if saturating
            dist_mat[i][j] = dist
            dist_mat[j][i] = dist
    return dist_mat


################################################
# (4) WEIGHBOR ALGORITHM IMPLEMENTATION
################################################
class Weighbor:
    def __init__(self, sequence_length=500, base_types=4):
        self.L = sequence_length
        self.B = base_types
        self.EPSILON = 1e-9 / self.L
        self.MINB = 0.0 / self.L
        
        self.sigBi = 1.0 / self.B
        self.sigBB = (self.B - 1.0) / self.B
        self.sigBBi = self.B / (self.B - 1.0)
        self.sigBiBBLinv = (self.B - 1.0) / (self.B * self.B * self.L)

    def sigma2t(self, d):
        """Total variance calculation (equation 0.3 from paper)"""
        if d <= 0:
            return 0.0
        sub1 = np.expm1(d * self.sigBBi)
        return self.sigBiBBLinv * sub1 * (sub1 + self.B)

    def sigma2_3(self, D, i, j, k, C):
        """Non-additive noise calculation"""
        if i == j or i == k or k == j:
            return 0.0

        dij = D[i][j]
        dik = D[i][k]
        djk = D[j][k]

        if djk > (dij + dik):
            return self.EPSILON
        elif dij > (dik + djk):
            d_il = min(dik, dij - djk)
            d_lj = dij - d_il
            sigma = self.sigma2t(dij + C[i] + C[j]) - self.sigma2t(d_il + C[i]) - self.sigma2t(d_lj + C[j])
        elif dik > (dij + djk):
            return self.EPSILON
        else:
            d_il = (dij + dik - djk) / 2
            d_lj = dij - d_il
            sigma = self.sigma2t(dij + C[i] + C[j]) - self.sigma2t(d_il + C[i]) - self.sigma2t(d_lj + C[j])

        return max(self.EPSILON, sigma)

    def build_tree(self, D_input, mapping):
        """Build tree using refined Weighbor algorithm."""
        N = len(D_input)
        D = defaultdict(lambda: defaultdict(float))

        # Initialize distance matrix
        for i in range(N):
            for j in range(N):
                D[i][j] = D_input[i][j]

        nodes = list(range(N))
        next_node = N
        edges = []
        branch_lengths = {}
        C = defaultdict(float)  # Renormalization terms

        while len(nodes) > 2:
            # Find the best pair to join
            i, j = self.find_best_pair(D, nodes, C)
            new_node = next_node
            node_i = nodes[i]
            node_j = nodes[j]

            # Calculate branch lengths
            dij = D[node_i][node_j]
            sum_diffs = 0
            weights = 0
            for k in range(len(nodes)):
                if k != i and k != j:
                    node_k = nodes[k]
                    sigma_i = self.sigma2_3(D, node_i, node_k, node_j, C)
                    sigma_j = self.sigma2_3(D, node_j, node_k, node_i, C)
                    weight = 1.0 / (sigma_i + sigma_j + self.EPSILON)
                    sum_diffs += weight * (D[node_i][node_k] - D[node_j][node_k])
                    weights += weight

            if weights > 0:
                branch_i = (dij + sum_diffs / weights) / 2.0
                branch_j = dij - branch_i
            else:
                branch_i = branch_j = dij / 2.0

            # Ensure non-negative branch lengths
            branch_i = max(0.0, branch_i)
            branch_j = max(0.0, branch_j)

            # Store branch lengths
            branch_lengths[(new_node, node_i)] = branch_i
            branch_lengths[(node_i, new_node)] = branch_i
            branch_lengths[(new_node, node_j)] = branch_j
            branch_lengths[(node_j, new_node)] = branch_j

            # Add edges
            edges.append((new_node, node_i))
            edges.append((new_node, node_j))

            # Update distances to new node
            for k in nodes:
                if k != node_i and k != node_j:
                    sigma_i = self.sigma2_3(D, node_i, k, node_j, C)
                    sigma_j = self.sigma2_3(D, node_j, k, node_i, C)
                    total_weight = 1.0 / (sigma_i + self.EPSILON) + 1.0 / (sigma_j + self.EPSILON)
                    if total_weight > 0:
                        new_dist = (D[node_i][k] / (sigma_i + self.EPSILON) +
                                    D[node_j][k] / (sigma_j + self.EPSILON)) / total_weight
                    else:
                        new_dist = (D[node_i][k] + D[node_j][k]) / 2.0
                    D[new_node][k] = D[k][new_node] = max(0.0, new_dist)

            C[new_node] = 0.5 * (C[node_i] + C[node_j] + D[node_i][node_j] / len(nodes))

            # Remove merged nodes and add new node
            nodes.pop(max(i, j))
            nodes.pop(min(i, j))
            nodes.append(new_node)
            next_node += 1

        # Handle final two nodes
        final_dist = D[nodes[0]][nodes[1]]
        edges.append((nodes[0], nodes[1]))
        branch_lengths[(nodes[0], nodes[1])] = final_dist / 2.0
        branch_lengths[(nodes[1], nodes[0])] = final_dist / 2.0

        # Convert branch lengths to matrix format
        uD = defaultdict(lambda: defaultdict(float))
        for (i, j), length in branch_lengths.items():
            uD[i][j] = length

        return edges, dict(uD), next_node - 1

    def find_best_pair(self, D, nodes, C):
        """Find the best pair of nodes to join."""
        min_cost = float('inf')
        best_pair = None
        N = len(nodes)
        gamma = 1.0 / (N - 3) if N > 4 else 1.0  # Correction factor

        for i in range(N - 1):
            for j in range(i + 1, N):
                node_i = nodes[i]
                node_j = nodes[j]

                # Additivity term
                add_terms = []
                for k in range(N):
                    if k != i and k != j:
                        node_k = nodes[k]
                        sigma = self.sigma2_3(D, node_i, node_j, node_k, C)
                        if sigma > self.EPSILON:
                            diff = D[node_i][node_k] - D[node_j][node_k]
                            add_terms.append(diff ** 2 / (2 * sigma))

                add_cost = gamma * np.mean(add_terms) if add_terms else 0.0

                # Positivity term
                branch_lengths = self.estimate_branch_lengths(D, nodes, i, j, C)
                var = self.sigma2t(D[node_i][node_j])

                if var > 0:
                    z = -min(branch_lengths) / np.sqrt(2 * var)
                    # smaller z => bigger cost
                    pos_cost = -np.log(0.5 * erfc(z)) if z < 0 else 0
                else:
                    pos_cost = 0 if min(branch_lengths) >= 0 else float('inf')

                total_cost = add_cost + pos_cost
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_pair = (i, j)

        return best_pair

    def estimate_branch_lengths(self, D, nodes, i, j, C):
        node_i = nodes[i]
        node_j = nodes[j]
        dij = D[node_i][node_j]
        
        sum_diffs = 0
        weights = 0
        for k in range(len(nodes)):
            if k != i and k != j:
                node_k = nodes[k]
                sigma_i = self.sigma2_3(D, node_i, node_k, node_j, C)
                sigma_j = self.sigma2_3(D, node_j, node_k, node_i, C)
                if sigma_i + sigma_j > self.EPSILON:
                    w = 2.0 / (sigma_i + sigma_j)
                    sum_diffs += w * (D[node_i][node_k] - D[node_j][node_k])
                    weights += w

        if weights > 0:
            branch_i = (dij + sum_diffs / weights) / 2.0
            branch_j = dij - branch_i
        else:
            branch_i = branch_j = dij / 2.0

        return [max(0.0, branch_i), max(0.0, branch_j)]


################################################
# (5) SIMPLE ROBINSON-FOULDS (RF) DISTANCE
################################################
def compute_rf_distance(true_parent_map, inferred_edges, leaves, root_id):
    """
    Minimalistic approach to computing the RF distance:
      - Convert both the true tree and the inferred tree into sets of bipartitions.
      - The RF distance is the size of the symmetric difference of those sets.
    """
    # 1. Get bipartitions from the true tree
    true_bipartitions = get_bipartitions(true_parent_map, leaves, root_id)

    # 2. Convert edges in the inferred tree to parent_map
    inferred_parent_map = build_parent_map_from_edges(inferred_edges)
    # We don't exactly know the root in the inferred tree, so pick the highest node ID as root.
    # (The Weighbor code returns 'fake_root' as next_node - 1)
    inf_root_id = max(e[0] for e in inferred_edges)
    inferred_bipartitions = get_bipartitions(inferred_parent_map, leaves, inf_root_id)

    # 3. The RF distance is the size of the symmetric difference
    #    ignoring bipartitions that include all leaves or single leaves
    sd = (true_bipartitions ^ inferred_bipartitions)
    return len(sd)


def build_parent_map_from_edges(edges):
    """
    For edges (a,b), assume a is parent of b if a > b.
    This is a naive assumption: Weighbor returns edges (parent, child).
    """
    parent_map = {}
    for a, b in edges:
        if a > b:
            parent_map[b] = (a, 0.1)  # we ignore the branch length for bipart partition
        else:
            parent_map[a] = (b, 0.1)
    return parent_map


def get_bipartitions(parent_map, leaves, root_id):
    """
    Return all bipartitions (as frozensets) from the tree.
    We'll do a post-order recursion and collect the set of leaves
    under each child. Each edge from parent->child defines a bipartition
    of child subtree vs. the rest.
    """
    children_map = defaultdict(list)
    for c, (p, _) in parent_map.items():
        children_map[p].append(c)

    def get_leafset(node):
        if node in leaves:
            return set([node])
        else:
            # union of child leafsets
            union_set = set()
            for ch in children_map[node]:
                union_set |= get_leafset(ch)
            return union_set

    all_bipartitions = set()

    def traverse(node):
        child_sets = []
        for ch in children_map[node]:
            child_leaf_set = get_leafset(ch)
            child_sets.append(child_leaf_set)
            # define bipartition: child_leaf_set vs (all_leaves - child_leaf_set)
            if len(child_leaf_set) < len(leaves):  # ignore trivial splits
                all_bipartitions.add(frozenset(child_leaf_set))

            traverse(ch)

    traverse(root_id)
    return all_bipartitions


################################################
# (6) RUN EXPERIMENTS FOR MULTIPLE DATASETS
################################################
def run_experiment_weighbor():
    dataset_sizes = [10, 20, 50, 100, 200]
    replicates = 3

    # We will store results in a dictionary: {n: [(runtime, rf_dist), ...], ...}
    results = defaultdict(list)

    for n in dataset_sizes:
        for rep in range(replicates):
            # Simulate a random tree
            parent_map, leaves, root_id = simulate_birth_death_tree(
                num_taxa=n, birth_rate=1.0, death_rate=0.5, seed=None
            )

            # Simulate sequences
            leaf_seqs = simulate_sequences(parent_map, leaves, root_id,
                                           seq_length=500, mu=1e-3, seed=None)

            # Compute distance matrix
            dist_mat = compute_distance_matrix(leaf_seqs, leaves)

            # Construct a mapping from index -> leaf_id (Weighbor expects some naming)
            # For simplicity, map i -> i, just keep track
            index_to_leaf = {i: leaves[i] for i in range(n)}

            # Run Weighbor
            weighbor = Weighbor(sequence_length=500, base_types=4)
            start_time = time.time()
            E, uD, fake_root = weighbor.build_tree(dist_mat, index_to_leaf)
            end_time = time.time()

            runtime = end_time - start_time

            # Compute RF distance between true tree and the inferred tree
            rf_dist = compute_rf_distance(parent_map, E, leaves, root_id)

            # Store result
            results[n].append((runtime, rf_dist))

    # Summarize results
    for n in dataset_sizes:
        runtimes = [r[0] for r in results[n]]
        rf_vals = [r[1] for r in results[n]]
        avg_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)
        avg_rf = np.mean(rf_vals)
        std_rf = np.std(rf_vals)

        print(f"{n} Taxa => Avg. Runtime: {avg_runtime:.2f}s (SD {std_runtime:.2f}) | "
              f"Avg. RF Distance: {avg_rf:.2f} (SD {std_rf:.2f})")


if __name__ == "__main__":
    run_experiment_weighbor()
