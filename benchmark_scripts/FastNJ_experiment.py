#!/usr/bin/env python3

import random
import time
import math
import argparse
import numpy as np
from collections import defaultdict

def simulate_birth_death_tree(num_taxa, birth_rate=1.0, death_rate=0.5, seed=None):
    if seed is not None:
        random.seed(seed)

    nodes = [0]
    next_id = 1

    while len(nodes) < num_taxa:
        nodes.append(next_id)
        next_id += 1

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

def simulate_sequences(parent_map, leaves, root_id, seq_length=500, alphabet="ACGT", mu=1e-3, seed=None):
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

def compute_distance_matrix(leaf_seqs, leaves):
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
                dist = 2.0
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat

def fast_nj(D, og):
    n = list(D.keys())
    E = []
    lengths = {}
    counter = max(D.keys()) + 1
    fake_root = None

    R = {i: sum(D[i].values()) for i in D}

    V = set()
    for i in D:
        for j in D:
            if i != j:
                V.add((i, j))

    while len(n) > 2:
        (i, j) = min(V, key=lambda x: (len(n) - 2) * D[x[0]][x[1]] - R[x[0]] - R[x[1]])
        
        r_i = R[i]
        r_j = R[j]
        iz_dist = 0.5 * D[i][j] + 0.5 * (r_i - r_j) / (len(n) - 2)
        jz_dist = D[i][j] - iz_dist

        z = counter
        counter += 1

        D[z] = {}
        for k in n:
            if k != i and k != j:
                new_dist = 0.5 * (D[i][k] + D[j][k] - D[i][j])
                D[z][k] = new_dist
                D[k][z] = new_dist
        D[z][z] = 0.0

        R[z] = sum(D[z].values())
        R.pop(i, None)
        R.pop(j, None)

        V = {(x, y) for (x, y) in V if x not in (i, j) and y not in (i, j)}
        for k in n:
            if k not in (i, j):
                V.add((z, k))

        E.append((z, i))
        E.append((z, j))
        lengths[(z, i)] = iz_dist
        lengths[(z, j)] = jz_dist

        del D[i]
        del D[j]
        n.remove(i)
        n.remove(j)
        n.append(z)

    if len(n) == 2:
        i, j = n
        E.append((i, j))
        lengths[(i, j)] = D[i][j]
    else:
        i = n[0]

    if og in n:
        out_edges = [edge for edge in E if og in edge]
        if out_edges:
            edge = out_edges[0]
            start, end = (edge[1], edge[0]) if edge[1] == og else (edge[0], edge[1])
            length = lengths.get(edge, lengths.get((start, end), 0.0))

            E.remove(edge)
            if (start, end) in lengths:
                lengths.pop((start, end), None)
            if (end, start) in lengths:
                lengths.pop((end, start), None)

            fake_root = counter
            counter += 1

            E.append((fake_root, og))
            E.append((fake_root, end))
            lengths[(fake_root, og)] = length / 2
            lengths[(fake_root, end)] = length / 2
    else:
        fake_root = counter
        counter += 1
        if len(n) == 2:
            E.remove((i, j))
            half_len = lengths.get((i, j), 0) / 2
            E.append((fake_root, i))
            E.append((fake_root, j))
            lengths[(fake_root, i)] = half_len
            lengths[(fake_root, j)] = half_len
        else:
            E.append((fake_root, i))
            lengths[(fake_root, i)] = 0.0

    nodes = set()
    for (a, b) in E:
        nodes.add(a)
        nodes.add(b)
    uD = {x: {y: 0.0 for y in nodes} for x in nodes}
    for (a, b), dist in lengths.items():
        uD[a][b] = dist
        uD[b][a] = dist

    return E, uD, fake_root

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

def compute_rf_distance(true_parent_map, inferred_edges, leaves, root_id):
    true_bip = get_bipartitions(true_parent_map, leaves, root_id)
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

def compute_branch_length_error(true_parent_map, inferred_edges, inferred_lengths):
    inferred_map = {}
    for (parent, child), length in inferred_lengths.items():
        inferred_map[(parent, child)] = length
        inferred_map[(child, parent)] = length

    squared_errors = []
    for child, (parent, true_length) in true_parent_map.items():
        if (parent, child) in inferred_map:
            inferred_length = inferred_map[(parent, child)]
        elif (child, parent) in inferred_map:
            inferred_length = inferred_map[(child, parent)]
        else:
            continue
        squared_errors.append((inferred_length - true_length) ** 2)

    return math.sqrt(sum(squared_errors) / len(squared_errors))

def run_experiment_fastnj():
    def modify_branch_lengths_for_rate_variation(parent_map, rate_factor=20):
        for child, (parent, branch_length) in parent_map.items():
            if random.random() < 0.3:
                parent_map[child] = (parent, branch_length * rate_factor)
        return parent_map

    def add_random_noise(dist_mat, noise_level=0.1):
        avg_dist = np.mean(dist_mat)
        noise = np.random.normal(scale=noise_level * avg_dist, size=dist_mat.shape)
        noisy_dist_mat = dist_mat + noise
        noisy_dist_mat = (noisy_dist_mat + noisy_dist_mat.T) / 2
        np.fill_diagonal(noisy_dist_mat, 0)
        return np.maximum(noisy_dist_mat, 0)

    dataset_sizes = [10, 20, 50, 100, 200]
    replicates = 3
    results = {}

    for n in dataset_sizes:
        for challenge in ["baseline", "rate_variation", "noise"]:
            runtimes = []
            rfs = []
            bles = []
            for rep in range(replicates):
                parent_map, leaves, root_id = simulate_birth_death_tree(num_taxa=n)

                if challenge == "rate_variation":
                    parent_map = modify_branch_lengths_for_rate_variation(parent_map, rate_factor=20)

                leaf_seqs = simulate_sequences(parent_map, leaves, root_id, seq_length=500, mu=1e-3)

                dist_mat = compute_distance_matrix(leaf_seqs, leaves)

                if challenge == "noise":
                    dist_mat = add_random_noise(dist_mat, noise_level=0.1)

                D = {i: {j: dist_mat[i, j] for j in range(n)} for i in range(n)}

                og_index = 0

                start = time.time()
                E, uD, fake_root = fast_nj(D, og_index)
                elapsed = time.time() - start

                rf_dist = compute_rf_distance(parent_map, E, leaves, root_id)

                inferred_lengths = {(parent, child): uD[parent][child] for parent, child in E}
                ble = compute_branch_length_error(parent_map, E, inferred_lengths)

                runtimes.append(elapsed)
                rfs.append(rf_dist)
                bles.append(ble)

            results[(n, challenge)] = {
                "avg_runtime": np.mean(runtimes),
                "std_runtime": np.std(runtimes),
                "avg_rf": np.mean(rfs),
                "std_rf": np.std(rfs),
                "avg_ble": np.mean(bles),
                "std_ble": np.std(bles),
            }

    for (n, challenge), res in results.items():
        print(f"{n} Taxa | {challenge} => Avg. Runtime: {res['avg_runtime']:.2f}s "
              f"(SD {res['std_runtime']:.2f}) | Avg. RF Distance: {res['avg_rf']:.2f} "
              f"(SD {res['std_rf']:.2f}) | Avg. BLE: {res['avg_ble']:.4f} "
              f"(SD {res['std_ble']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="FastNJ Experiment Script")
    args = parser.parse_args()

    run_experiment_fastnj()

if __name__ == "__main__":
    main()