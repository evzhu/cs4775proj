#!/usr/bin/env python3

import random
import time
import math
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
            parent, branch_len = parent_map[child]
            child_seq = mutate_sequence(seq, mu * branch_len, alphabet)
            simulate_node(child, child_seq)

    root_seq = generate_root_sequence(seq_length, alphabet)
    simulate_node(root_id, root_seq)

    leaf_seqs = {leaf: node_sequence[leaf] for leaf in leaves}
    return leaf_seqs

def mutate_sequence(seq, expected_substitutions, alphabet):
    if len(seq) == 0:
        return seq
    p = expected_substitutions / len(seq)

    out = []
    for base in seq:
        if random.random() < p:
            possible_bases = alphabet.replace(base, "")
            out.append(random.choice(possible_bases))
        else:
            out.append(base)
    return "".join(out)

def compute_distance_matrix(leaf_seqs, leaves):
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

def calculate_q_matrix(D, active_nodes):
    n = len(active_nodes)
    r = {i: sum(D[i][j] for j in active_nodes) for i in active_nodes}
    Q = {i: {j: 0.0 for j in active_nodes} for i in active_nodes}
    
    for i in active_nodes:
        for j in active_nodes:
            if i != j:
                Q[i][j] = (n - 2) * D[i][j] - r[i] - r[j]
    return Q

def find_minimum_q(Q, active_nodes):
    min_q = float('inf')
    min_i = min_j = -1
    
    for i in active_nodes:
        for j in active_nodes:
            if i < j and Q[i][j] < min_q:
                min_q = Q[i][j]
                min_i, min_j = i, j
    return min_i, min_j

def neighbor_join(D, og):
    n = len(D)
    next_node = n
    active_nodes = set(range(n))
    E = []
    uD = {i: {j: D[i][j] for j in range(n)} for i in range(n)}
    
    while len(active_nodes) > 2:
        Q = calculate_q_matrix(uD, active_nodes)
        
        i, j = find_minimum_q(Q, active_nodes)
        
        r = {k: sum(uD[k][m] for m in active_nodes) for k in active_nodes}
        delta = (r[i] - r[j]) / (len(active_nodes) - 2)
        li = (uD[i][j] + delta) / 2
        lj = (uD[i][j] - delta) / 2
        
        new_node = next_node
        next_node += 1
        uD[new_node] = {k: 0.0 for k in range(next_node)}
        for k in range(next_node):
            uD[k][new_node] = uD[new_node][k]
        
        for k in active_nodes:
            if k != i and k != j:
                d = (uD[i][k] + uD[j][k] - uD[i][j]) / 2
                uD[new_node][k] = uD[k][new_node] = d
        
        E.append((new_node, i))
        E.append((new_node, j))
        uD[new_node][i] = uD[i][new_node] = li
        uD[new_node][j] = uD[j][new_node] = lj
        
        active_nodes.remove(i)
        active_nodes.remove(j)
        active_nodes.add(new_node)
    
    final_nodes = list(active_nodes)
    final_length = uD[final_nodes[0]][final_nodes[1]]
    E.append((final_nodes[0], final_nodes[1]))
    
    for idx, (u, v) in enumerate(E):
        if u == og or v == og:
            other_node = v if u == og else u
            E.pop(idx)
            fake_root = next_node
            next_node += 1
            edge_length = uD[u][v]
            half_length = edge_length / 2
            E.append((fake_root, og))
            E.append((fake_root, other_node))
            uD[fake_root] = {k: 0.0 for k in range(next_node)}
            for k in range(next_node):
                uD[k][fake_root] = uD[fake_root][k]
            uD[fake_root][og] = uD[og][fake_root] = half_length
            uD[fake_root][other_node] = uD[other_node][fake_root] = half_length
            break
    
    return E, uD, fake_root

def compute_rf_distance(true_parent_map, inferred_edges, leaves, root_id):
    true_bip = get_bipartitions_from_parent_map(true_parent_map, leaves, root_id)

    inferred_parent_map = {}
    for a, b in inferred_edges:
        if a > b:
            inferred_parent_map[b] = (a, 0.1)
        else:
            inferred_parent_map[a] = (b, 0.1)

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

def run_experiment_nj():
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
                E, uD, fake_root = neighbor_join(D, og_index)
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
    run_experiment_nj()

if __name__ == "__main__":
    main()