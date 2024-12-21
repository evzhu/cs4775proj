#!/usr/bin/env python3

import random
import time
import math
import argparse
import numpy as np
from collections import defaultdict

################################################
# (1) SIMULATE A (SIMPLIFIED) BIRTH-DEATH TREE
################################################
def simulate_birth_death_tree(num_taxa, birth_rate=1.0, death_rate=0.5, seed=None):
    """
    Generates a random bifurcating tree topology with random branch lengths.
    A fully rigorous birth-death simulator is complex, so we simplify:
      - Start with 1 lineage, keep adding lineages until we have num_taxa.
      - Randomly merge them into a binary tree.
      - Assign random branch lengths from an exponential distribution.
    Returns:
      - parent_map: dict: {child: (parent, branch_length)}
      - leaves: list of leaf IDs (0..num_taxa-1)
      - root_id: the ID of the final root
    """
    if seed is not None:
        random.seed(seed)

    # Start with a single lineage labeled 0
    nodes = [0]
    next_id = 1

    # Keep adding lineages
    while len(nodes) < num_taxa:
        nodes.append(next_id)
        next_id += 1

    # Now pair them up to form internal nodes
    internal_id = num_taxa
    parent_map = {}
    while len(nodes) > 1:
        c1 = random.choice(nodes)
        nodes.remove(c1)
        c2 = random.choice(nodes)
        nodes.remove(c2)

        p = internal_id
        internal_id += 1

        bl1 = random.expovariate(birth_rate)
        bl2 = random.expovariate(birth_rate)

        parent_map[c1] = (p, bl1)
        parent_map[c2] = (p, bl2)

        nodes.append(p)

    root_id = nodes[0]
    leaves = list(range(num_taxa))
    return parent_map, leaves, root_id


################################################
# (2) SIMULATE SEQUENCES UNDER JUKES-CANTOR
################################################
def simulate_sequences(parent_map, leaves, root_id,
                       seq_length=500, alphabet="ACGT", mu=1e-3, seed=None):
    """
    Simulates nucleotide sequences along the tree using a simplified Jukes-Cantor model.
    For each branch of length L, expected_substitutions ~ mu * L * seq_length.
    Returns:
      - leaf_seqs: dict of leaf_id -> sequence
    """
    if seed is not None:
        random.seed(seed)

    children_map = defaultdict(list)
    for child, (parent, _) in parent_map.items():
        children_map[parent].append(child)

    node_sequence = {}

    def generate_root_sequence(length, alphabet):
        return "".join(random.choice(alphabet) for _ in range(length))

    def simulate_node(node, seq):
        node_sequence[node] = seq
        for child in children_map[node]:
            parent, bl = parent_map[child]
            child_seq = mutate_sequence(seq, mu * bl, alphabet)
            simulate_node(child, child_seq)

    root_seq = generate_root_sequence(seq_length, alphabet)
    simulate_node(root_id, root_seq)
    leaf_seqs = {leaf: node_sequence[leaf] for leaf in leaves}
    return leaf_seqs

def mutate_sequence(seq, expected_substitutions, alphabet):
    """
    Jukes-Cantor approximation for site-by-site mutation with probability p.
    p = expected_substitutions / len(seq).
    """
    if not seq:
        return seq
    p = expected_substitutions / len(seq)
    out = []
    for base in seq:
        if random.random() < p:
            out.append(random.choice(alphabet.replace(base, "")))
        else:
            out.append(base)
    return "".join(out)


################################################
# (3) COMPUTE PAIRWISE DISTANCES (Jukes-Cantor)
################################################
def compute_distance_matrix(leaf_seqs, leaves):
    """
    For each pair, compute proportion mismatch p, then Jukes-Cantor:
      d = -3/4 * ln(1 - 4p/3).
    If p >= 0.75, set a large distance (2.0) to avoid log domain error.
    """
    n = len(leaves)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            seq_i = leaf_seqs[leaves[i]]
            seq_j = leaf_seqs[leaves[j]]
            p = sum(a != b for a, b in zip(seq_i, seq_j)) / len(seq_i)
            if p < 0.75:
                dist = -0.75 * math.log(1 - (4.0/3.0)*p)
            else:
                dist = 2.0  # Saturation
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat


################################################
# (4) FASTNJ ALGORITHM
################################################
def fast_nj(D, og):
    """
    Performs the Fast Neighbor Joining (FNJ) algorithm on the distance matrix D,
    with the outgroup index og to root the tree.
    D is a dict-of-dicts, {i: {j: distance}}.
    """
    n = list(D.keys())
    E = []
    lengths = {}
    counter = max(D.keys()) + 1
    fake_root = None

    # Initialize row sums
    R = {i: sum(D[i].values()) for i in D}

    # Visible set
    V = set()
    for i in D:
        for j in D:
            if i != j:
                V.add((i, j))

    while len(n) > 2:
        # (a) Find minimal in the visible set
        (i, j) = min(V, key=lambda x: (len(n) - 2) * D[x[0]][x[1]] - R[x[0]] - R[x[1]])
        
        # (b) Calculate branch lengths
        r_i = R[i]
        r_j = R[j]
        iz_dist = 0.5 * D[i][j] + 0.5 * (r_i - r_j) / (len(n) - 2)
        jz_dist = D[i][j] - iz_dist

        # (c) Create new internal node
        z = counter
        counter += 1

        # (d) Update D
        D[z] = {}
        for k in n:
            if k != i and k != j:
                new_dist = 0.5 * (D[i][k] + D[j][k] - D[i][j])
                D[z][k] = new_dist
                D[k][z] = new_dist
        D[z][z] = 0.0

        # (e) Update row sums
        R[z] = sum(D[z].values())
        R.pop(i, None)
        R.pop(j, None)

        # (f) Update visible set
        V = {(x, y) for (x, y) in V if x not in (i, j) and y not in (i, j)}
        for k in n:
            if k not in (i, j):
                V.add((z, k))

        # (g) Update edges and lengths
        E.append((z, i))
        E.append((z, j))
        lengths[(z, i)] = iz_dist
        lengths[(z, j)] = jz_dist

        # (h) Remove merged nodes
        del D[i]
        del D[j]
        n.remove(i)
        n.remove(j)
        n.append(z)

    # (i) Add the last edge
    if len(n) == 2:
        i, j = n
        E.append((i, j))
        lengths[(i, j)] = D[i][j]
    else:
        # fallback
        i = n[0]

    # (j) Handle the outgroup
    if og in n:
        # find edge that includes og
        out_edges = [edge for edge in E if og in edge]
        if out_edges:
            edge = out_edges[0]
            start, end = (edge[1], edge[0]) if edge[1] == og else (edge[0], edge[1])
            length = lengths.get(edge, lengths.get((start, end), 0.0))

            # Remove that edge
            E.remove(edge)
            if (start, end) in lengths:
                lengths.pop((start, end), None)
            if (end, start) in lengths:
                lengths.pop((end, start), None)

            # Create a fake root
            fake_root = counter
            counter += 1

            E.append((fake_root, og))
            E.append((fake_root, end))
            lengths[(fake_root, og)] = length / 2
            lengths[(fake_root, end)] = length / 2
    else:
        # If the outgroup is not present, set any new internal node as root
        fake_root = counter
        counter += 1
        # For the last edge i-j, we can attach them to the root with half-length
        if len(n) == 2:
            E.remove((i, j))
            half_len = lengths.get((i, j), 0) / 2
            E.append((fake_root, i))
            E.append((fake_root, j))
            lengths[(fake_root, i)] = half_len
            lengths[(fake_root, j)] = half_len
        else:
            # no edge found, i alone
            E.append((fake_root, i))
            lengths[(fake_root, i)] = 0.0

    # (k) Build final distance dictionary
    nodes = set()
    for (a, b) in E:
        nodes.add(a)
        nodes.add(b)
    uD = {x: {y: 0.0 for y in nodes} for x in nodes}
    for (a, b), dist in lengths.items():
        uD[a][b] = dist
        uD[b][a] = dist

    return E, uD, fake_root


################################################
# (5) BUILD TREE & NEWICK
################################################
def assemble_tree(fake_root, E):
    tree_map = defaultdict(list)
    adj = defaultdict(list)
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    stack = [(fake_root, None)]
    while stack:
        node, parent = stack.pop()
        visited.add(node)
        for nbr in adj[node]:
            if nbr != parent and nbr not in visited:
                tree_map[node].append(nbr)
                stack.append((nbr, node))

    return tree_map

def generate_newick(fake_root, tree_map, uD, mapping=None):
    def traverse(node, parent):
        if node not in tree_map or len(tree_map[node]) == 0:
            label = mapping[node] if mapping and node in mapping else str(node)
        else:
            label = "(" + ",".join(traverse(ch, node) for ch in tree_map[node]) + ")"

        if parent is None:
            return label + ";"
        else:
            return f"{label}:{uD[node][parent]:.6f}"

    return traverse(fake_root, None)


################################################
# (6) RF DISTANCE (SIMPLE BIPARTITION)
################################################
def compute_rf_distance(true_parent_map, inferred_edges, leaves, root_id):
    true_bip = get_bipartitions(true_parent_map, leaves, root_id)
    # Convert edges to parent map for the inferred tree
    # guess the largest node as root
    inf_root = max(e[0] for e in inferred_edges)
    inferred_pm = {}
    for a, b in inferred_edges:
        if a > b:
            inferred_pm[b] = (a, 0.1)
        else:
            inferred_pm[a] = (b, 0.1)
    inferred_bip = get_bipartitions(inferred_pm, leaves, inf_root)

    return len(true_bip ^ inferred_bip)

def get_bipartitions(parent_map, leaves, root):
    children_map = defaultdict(list)
    for c, (p, _) in parent_map.items():
        children_map[p].append(c)

    all_bip = set()

    def collect_leafset(node):
        if node in leaves:
            return {node}
        s = set()
        for ch in children_map[node]:
            s |= collect_leafset(ch)
        return s

    def traverse(node):
        for ch in children_map[node]:
            cset = collect_leafset(ch)
            if 0 < len(cset) < len(leaves):
                all_bip.add(frozenset(cset))
            traverse(ch)

    traverse(root)
    return all_bip


################################################
# (7) RUN EXPERIMENTS
################################################
def run_experiment_fastnj():
    dataset_sizes = [10, 20, 50, 100, 200]
    replicates = 3
    results = {}

    for n in dataset_sizes:
        runtimes = []
        rfs = []
        for rep in range(replicates):
            # 1. Simulate tree
            parent_map, leaves, root_id = simulate_birth_death_tree(num_taxa=n)

            # 2. Simulate sequences
            leaf_seqs = simulate_sequences(parent_map, leaves, root_id, seq_length=500, mu=1e-3)

            # 3. Compute distance matrix
            dist_mat = compute_distance_matrix(leaf_seqs, leaves)

            # 4. Convert to dict-of-dicts for the fastNJ function
            D = {}
            for i in range(n):
                D[i] = {}
                for j in range(n):
                    D[i][j] = dist_mat[i][j]

            # Arbitrarily pick 0 as outgroup
            og_index = 0

            # 5. Run FastNJ
            start_time = time.time()
            E, uD, fake_root = fast_nj(D, og_index)
            elapsed = time.time() - start_time

            # 6. Compute RF distance
            rf_dist = compute_rf_distance(parent_map, E, leaves, root_id)

            runtimes.append(elapsed)
            rfs.append(rf_dist)

        results[n] = {
            "avg_runtime": np.mean(runtimes),
            "std_runtime": np.std(runtimes),
            "avg_rf": np.mean(rfs),
            "std_rf": np.std(rfs),
        }

    # Print results
    for n in dataset_sizes:
        print(f"{n} Taxa => Avg. Runtime: {results[n]['avg_runtime']:.2f}s "
              f"(SD {results[n]['std_runtime']:.2f}) | "
              f"Avg. RF Distance: {results[n]['avg_rf']:.2f} "
              f"(SD {results[n]['std_rf']:.2f})")


################################################
# (8) MAIN
################################################
def main():
    """
    Example usage:
        python fastNJ_experiment.py
    You can redirect the console output or adapt as needed.
    """
    parser = argparse.ArgumentParser(description="FastNJ Experiment Script")
    args = parser.parse_args()

    run_experiment_fastnj()


if __name__ == "__main__":
    main()
