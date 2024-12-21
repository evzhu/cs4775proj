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
    A fully rigorous birth-death simulator is complex; we simplify:
      - Start with 1 lineage and repeatedly add lineages until we have num_taxa.
      - Merge them into a binary tree in a random manner.
      - Assign branch lengths from an exponential distribution.
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

    # Keep adding lineages until we reach num_taxa
    while len(nodes) < num_taxa:
        nodes.append(next_id)
        next_id += 1

    # Randomly pair up nodes to create internal nodes
    internal_id = num_taxa
    parent_map = {}
    while len(nodes) > 1:
        c1 = random.choice(nodes)
        nodes.remove(c1)
        c2 = random.choice(nodes)
        nodes.remove(c2)

        p = internal_id
        internal_id += 1

        # Assign random branch lengths
        bl1 = random.expovariate(birth_rate)
        bl2 = random.expovariate(birth_rate)

        parent_map[c1] = (p, bl1)
        parent_map[c2] = (p, bl2)

        nodes.append(p)

    # The last node is the root
    root_id = nodes[0]

    # Leaves are the first num_taxa
    leaves = list(range(num_taxa))
    return parent_map, leaves, root_id

################################################
# (2) SIMULATE SEQUENCES UNDER JUKES-CANTOR
################################################
def simulate_sequences(parent_map, leaves, root_id,
                       seq_length=500, alphabet="ACGT", mu=1e-3, seed=None):
    """
    Simulates nucleotide sequences down the tree using a Jukes-Cantor-like model.
    For each branch of length L, we assume expected_substitutions ~ mu * L * seq_length.
    Returns:
      - leaf_seqs: dict {leaf_id: sequence_string}
    """
    if seed is not None:
        random.seed(seed)

    # Build child map for top-down simulation
    children_map = defaultdict(list)
    for child, (parent, _) in parent_map.items():
        children_map[parent].append(child)

    node_sequence = {}

    def generate_root_sequence(length, alphabet):
        return "".join(random.choice(alphabet) for _ in range(length))

    def simulate_node(node, seq):
        node_sequence[node] = seq
        for child in children_map[node]:
            parent, branch_len = parent_map[child]
            child_seq = mutate_sequence(seq, mu * branch_len, alphabet)
            simulate_node(child, child_seq)

    # Random root sequence
    root_seq = generate_root_sequence(seq_length, alphabet)
    simulate_node(root_id, root_seq)

    # Extract leaf sequences only
    leaf_seqs = {leaf: node_sequence[leaf] for leaf in leaves}
    return leaf_seqs

def mutate_sequence(seq, expected_substitutions, alphabet):
    """
    With Jukes-Cantor, each site has probability p of mutating, p = expected_substitutions / len(seq).
    Any mutation is to a different base in 'alphabet'.
    """
    if len(seq) == 0:
        return seq
    p = expected_substitutions / len(seq)

    out = []
    for base in seq:
        if random.random() < p:
            # mutate to a different base
            possible_bases = alphabet.replace(base, "")
            out.append(random.choice(possible_bases))
        else:
            out.append(base)
    return "".join(out)

################################################
# (3) COMPUTE PAIRWISE DISTANCES (Jukes-Cantor)
################################################
def compute_distance_matrix(leaf_seqs, leaves):
    """
    For each pair (i, j), compute the proportion of mismatches p,
    then apply d = -3/4 * ln(1 - 4p/3). If p >= 0.75, we set a large distance to avoid log error.
    Returns a numpy NxN distance matrix.
    """
    n = len(leaves)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            seq_i = leaf_seqs[leaves[i]]
            seq_j = leaf_seqs[leaves[j]]
            p = sum(a != b for a, b in zip(seq_i, seq_j)) / len(seq_i)
            if p < 0.75:
                dist = -0.75 * math.log(1 - (4.0/3.0)*p)
            else:
                dist = 2.0
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat

################################################
# (4) NEIGHBOR JOIN FUNCTION (BIO-NJ VARIANT)
################################################
def bionj(D, og):
    """
    Performs the BioNJ algorithm (as provided in the user code docstring) to produce a phylogenetic tree.
    D is a dict-of-dicts distance matrix. og is the outgroup index.
    Returns (E, uD, fake_root).
    """
    n = list(D.keys())
    E = []
    lengths = {}
    counter = max(D.keys()) + 1

    while len(n) > 2:
        # Step 1: Compute Q-matrix
        Q = {}
        for i in D.keys():
            for j in D.keys():
                if i < j:
                    # BioNJ can incorporate variance, but here we do a simpler approach, as in code snippet
                    Q[(i, j)] = (len(n) - 2)*D[i][j] - sum(D[i].values()) - sum(D[j].values())

        # Step 2: Find best pair
        i, j = min(Q, key=Q.get)

        # Step 3: Compute branch lengths
        li = 0.5 * D[i][j] + (sum(D[i].values()) - sum(D[j].values()))/(2*(len(n)-2))
        lj = D[i][j] - li

        z = counter
        counter += 1

        E.append((z, i))
        E.append((z, j))
        lengths[(z, i)] = li
        lengths[(z, j)] = lj

        # Step 4: Update distances
        new_dists = {}
        for k in list(D.keys()):
            if k != i and k != j:
                new_dists[k] = (D[i][k] + D[j][k] - D[i][j]) / 2

        D[z] = {}
        for k, dist in new_dists.items():
            D[z][k] = dist
            D[k][z] = dist

        # Remove old nodes
        del D[i], D[j]
        for k in D.keys():
            D[k].pop(i, None)
            D[k].pop(j, None)

        n.remove(i)
        n.remove(j)
        n.append(z)

    i, j = n
    z = counter
    counter += 1
    li = D[i][j]/2
    lj = D[i][j]/2

    E.append((z, i))
    E.append((z, j))
    lengths[(z, i)] = li
    lengths[(z, j)] = lj

    # Outgroup handling
    if og in n:
        z2 = counter
        counter += 1
        other = i if i != og else j
        E.append((z2, og))
        E.append((z2, z))
        # For demonstration, assume half of li goes to 'z2->og' and half to 'z2->z'
        lengths[(z2, og)] = li/2
        lengths[(z2, z)] = li/2

    # Build final distance matrix uD
    nodes = {x for e in E for x in e}
    uD = {a: {b: 0.0 for b in nodes} for a in nodes}
    for (a, b), dist in lengths.items():
        uD[a][b] = dist
        uD[b][a] = dist

    return E, uD, counter - 1

################################################
# (5) ASSEMBLE TREE & NEWICK FORMATTING
################################################
def assemble_tree(fake_root, E):
    """
    As in user code, build adjacency from edges, then do a DFS from fake_root to
    define a tree_map: node->list_of_children.
    """
    tree_map = defaultdict(list)
    adjacency_list = defaultdict(list)
    for a, b in E:
        adjacency_list[a].append(b)
        adjacency_list[b].append(a)

    visited = set()
    stack = [(fake_root, None)]
    while stack:
        node, parent = stack.pop()
        visited.add(node)
        for nbr in adjacency_list[node]:
            if nbr != parent and nbr not in visited:
                tree_map[node].append(nbr)
                stack.append((nbr, node))
    return tree_map

def generate_newick(fake_root, tree_map, uD, mapping=None):
    """
    Produce a Newick string. If node has children, list them in parentheses;
    otherwise, it's a leaf. Branch length = uD[node][parent].
    """
    def newick_recursive(node, parent):
        if node not in tree_map or not tree_map[node]:
            # A leaf
            label = mapping.get(node, str(node)) if mapping else str(node)
        else:
            label = "(" + ",".join(newick_recursive(ch, node) for ch in tree_map[node]) + ")"

        if parent is None:
            return label + ";"  # end of recursion
        else:
            return f"{label}:{uD[node][parent]:.6f}"

    return newick_recursive(fake_root, None)

################################################
# (6) SIMPLE RF DISTANCE CALCULATOR
################################################
def compute_rf_distance(true_parent_map, inferred_edges, leaves, root_id):
    """
    A minimal set-based bipartition approach.
    For large trees, a more robust library or method is recommended.
    """
    # 1. bipartitions from true
    true_bip = get_bipartitions_from_parent_map(true_parent_map, leaves, root_id)

    # 2. reconstruct parent_map from inferred edges
    #    We'll guess a direction: parent > child
    inferred_parent_map = {}
    for a, b in inferred_edges:
        if a > b:
            inferred_parent_map[b] = (a, 0.1)
        else:
            inferred_parent_map[a] = (b, 0.1)

    # We guess the largest node ID might be root (as done in bionj, but might differ).
    inferred_root = max(e[0] for e in inferred_edges)
    inferred_bip = get_bipartitions_from_parent_map(inferred_parent_map, leaves, inferred_root)

    sd = (true_bip ^ inferred_bip)
    return len(sd)

def get_bipartitions_from_parent_map(parent_map, leaves, root):
    children_map = defaultdict(list)
    for c, (p, _) in parent_map.items():
        children_map[p].append(c)

    all_bip = set()

    def get_leafset(node):
        if node in leaves:
            return {node}
        s = set()
        for ch in children_map[node]:
            s |= get_leafset(ch)
        return s

    def traverse(node):
        child_sets = []
        for ch in children_map[node]:
            cset = get_leafset(ch)
            child_sets.append(cset)
            # bipart: cset vs. the rest
            if 0 < len(cset) < len(leaves):
                all_bip.add(frozenset(cset))
            traverse(ch)

    traverse(root)
    return all_bip

################################################
# (7) RUN EXPERIMENTS FOR MULTIPLE DATASET SIZES
################################################
def run_experiment_bionj():
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

            # 4. Convert to dict-of-dicts (BioNJ input format)
            D = {}
            for i in range(n):
                D[i] = {}
                for j in range(n):
                    D[i][j] = dist_mat[i, j]

            # pick an outgroup arbitrarily (e.g., the 0th leaf)
            og_index = 0

            # 5. Run BioNJ
            start = time.time()
            E, uD, fake_root = bionj(D, og_index)
            elapsed = time.time() - start

            # 6. Compute RF distance
            rf_dist = compute_rf_distance(parent_map, E, leaves, root_id)

            # Store results
            runtimes.append(elapsed)
            rfs.append(rf_dist)

        # Summarize for this dataset size
        results[n] = {
            "avg_runtime": np.mean(runtimes),
            "std_runtime": np.std(runtimes),
            "avg_rf": np.mean(rfs),
            "std_rf": np.std(rfs),
        }

    # Print final results
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
       python bionj_experiment.py
    You can redirect outputs or adapt as needed.
    """
    parser = argparse.ArgumentParser(description="BioNJ Experiment Script")
    args = parser.parse_args()

    run_experiment_bionj()

if __name__ == "__main__":
    main()
