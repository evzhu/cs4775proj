#!/usr/bin/env python3

import random
import time
import math
import numpy as np
from collections import defaultdict

def neighbor_joining_root_fix(distance_matrix):
    ids = list(distance_matrix.keys())
    n = len(ids)
    active_nodes = {i: ids[i] for i in range(n)}

    tree = []
    branch_lengths = {}
    counter = n

    D = {i: {j: distance_matrix[ids[i]][ids[j]] for j in range(n)} for i in range(n)}

    while len(D) > 2:
        Q = {}
        total_distances = {i: sum(D[i].values()) for i in D}
        for i in D:
            for j in D:
                if i != j:
                    Q[(i, j)] = (len(D) - 2) * D[i][j] - total_distances[i] - total_distances[j]

        i, j = min(Q, key=Q.get)

        total_i = total_distances[i]
        total_j = total_distances[j]
        new_node_distance_i = 0.5 * D[i][j] + (total_i - total_j) / (2 * (len(D) - 2))
        new_node_distance_j = D[i][j] - new_node_distance_i

        new_node_label = str(counter)
        tree.append((active_nodes[i], new_node_label))
        tree.append((active_nodes[j], new_node_label))
        branch_lengths[(active_nodes[i], new_node_label)] = new_node_distance_i
        branch_lengths[(active_nodes[j], new_node_label)] = new_node_distance_j

        new_distances = {}
        for k in D:
            if k != i and k != j:
                new_distances[k] = 0.5 * (D[i][k] + D[j][k] - D[i][j])

        D[counter] = new_distances
        for k in new_distances:
            D[k][counter] = new_distances[k]

        del D[i], D[j]
        for k in list(D.keys()):
            if i in D[k]:
                del D[k][i]
            if j in D[k]:
                del D[k][j]

        active_nodes[counter] = new_node_label
        del active_nodes[i]
        del active_nodes[j]

        counter += 1

    keys = list(D.keys())
    i, j = keys[0], keys[1]
    tree.append((active_nodes[i], active_nodes[j]))

    return tree, branch_lengths, counter-1

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
    dist_mat = {}
    for i in range(n):
        dist_mat[str(i)] = {}
        for j in range(n):
            if i == j:
                dist_mat[str(i)][str(j)] = 0.0
                continue
            seq_i = leaf_seqs[leaves[i]]
            seq_j = leaf_seqs[leaves[j]]
            p = sum(a != b for a, b in zip(seq_i, seq_j)) / len(seq_i)
            if p < 0.75:
                dist = -0.75 * math.log(1 - (4.0/3.0)*p)
            else:
                dist = 2.0
            dist_mat[str(i)][str(j)] = dist
    return dist_mat

def compute_rf_distance(true_parent_map, inferred_edges, leaves, root_id):
    true_bip = get_bipartitions_from_parent_map(true_parent_map, leaves, root_id)

    inferred_parent_map = {}
    for a, b in inferred_edges:
        if not isinstance(a, str):
            a = str(a)
        if not isinstance(b, str):
            b = str(b)
        if int(a if a.isdigit() else b) > int(b if b.isdigit() else a):
            inferred_parent_map[int(b if b.isdigit() else a)] = (int(a if a.isdigit() else b), 0.1)
        else:
            inferred_parent_map[int(a if a.isdigit() else b)] = (int(b if b.isdigit() else a), 0.1)

    inferred_root = max(sum([[int(a if a.isdigit() else b), int(b if b.isdigit() else a)] 
                            for a, b in inferred_edges], []))
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
        parent_int = int(parent) if isinstance(parent, str) and parent.isdigit() else None
        child_int = int(child) if isinstance(child, str) and child.isdigit() else None
        if parent_int is not None and child_int is not None:
            inferred_map[(parent_int, child_int)] = length
            inferred_map[(child_int, parent_int)] = length

    squared_errors = []
    for child, (parent, true_length) in true_parent_map.items():
        if (parent, child) in inferred_map:
            inferred_length = inferred_map[(parent, child)]
            squared_errors.append((inferred_length - true_length) ** 2)
        elif (child, parent) in inferred_map:
            inferred_length = inferred_map[(child, parent)]
            squared_errors.append((inferred_length - true_length) ** 2)

    if not squared_errors:
        return float('inf')
    return math.sqrt(sum(squared_errors) / len(squared_errors))

def run_experiment_njml():
    def modify_branch_lengths_for_rate_variation(parent_map, rate_factor=20):
        for child, (parent, branch_length) in parent_map.items():
            if random.random() < 0.3:
                parent_map[child] = (parent, branch_length * rate_factor)
        return parent_map

    def add_random_noise(dist_mat, noise_level=0.1):
        avg_dist = np.mean([dist_mat[i][j] for i in dist_mat for j in dist_mat[i]])
        for i in dist_mat:
            for j in dist_mat[i]:
                if i != j:
                    noise = random.gauss(0, noise_level * avg_dist)
                    dist_mat[i][j] += noise
                    dist_mat[j][i] += noise
        return dist_mat

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

                start = time.time()
                E, lengths, fake_root = neighbor_joining_root_fix(dist_mat)
                elapsed = time.time() - start

                rf_dist = compute_rf_distance(parent_map, E, leaves, root_id)

                ble = compute_branch_length_error(parent_map, E, lengths)

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
    run_experiment_njml()

if __name__ == "__main__":
    main()