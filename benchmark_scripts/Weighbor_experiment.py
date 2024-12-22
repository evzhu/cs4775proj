#!/usr/bin/env python3

import random
import time
import math
import numpy as np
from collections import defaultdict
from scipy.special import erf, erfc

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
        if d <= 0:
            return 0.0
        sub1 = np.expm1(d * self.sigBBi)
        return self.sigBiBBLinv * sub1 * (sub1 + self.B)

    def sigma2_3(self, D, i, j, k, C):
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

    def build_tree(self, D_input, mapping=None):
        N = len(D_input)
        D = defaultdict(lambda: defaultdict(float))

        for i in range(N):
            for j in range(N):
                D[i][j] = D_input[i][j]

        nodes = list(range(N))
        next_node = N
        edges = []
        branch_lengths = {}
        C = defaultdict(float)

        while len(nodes) > 2:
            i, j = self.find_best_pair(D, nodes, C)
            new_node = next_node
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
                    weight = 1.0 / (sigma_i + sigma_j + self.EPSILON)
                    sum_diffs += weight * (D[node_i][node_k] - D[node_j][node_k])
                    weights += weight

            if weights > 0:
                branch_i = (dij + sum_diffs / weights) / 2.0
                branch_j = dij - branch_i
            else:
                branch_i = branch_j = dij / 2.0

            branch_i = max(0.0, branch_i)
            branch_j = max(0.0, branch_j)

            branch_lengths[(new_node, node_i)] = branch_i
            branch_lengths[(node_i, new_node)] = branch_i
            branch_lengths[(new_node, node_j)] = branch_j
            branch_lengths[(node_j, new_node)] = branch_j

            edges.append((new_node, node_i))
            edges.append((new_node, node_j))

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

            nodes.pop(max(i, j))
            nodes.pop(min(i, j))
            nodes.append(new_node)
            next_node += 1

        final_dist = D[nodes[0]][nodes[1]]
        edges.append((nodes[0], nodes[1]))
        branch_lengths[(nodes[0], nodes[1])] = final_dist / 2.0
        branch_lengths[(nodes[1], nodes[0])] = final_dist / 2.0

        uD = defaultdict(lambda: defaultdict(float))
        for (i, j), length in branch_lengths.items():
            uD[i][j] = length

        return edges, dict(uD), next_node - 1

    def find_best_pair(self, D, nodes, C):
        min_cost = float('inf')
        best_pair = None
        N = len(nodes)
        gamma = 1.0 / (N - 3) if N > 4 else 1.0

        for i in range(N - 1):
            for j in range(i + 1, N):
                node_i = nodes[i]
                node_j = nodes[j]

                add_terms = []
                for k in range(N):
                    if k != i and k != j:
                        node_k = nodes[k]
                        sigma = self.sigma2_3(D, node_i, node_j, node_k, C)
                        if sigma > self.EPSILON:
                            diff = D[node_i][node_k] - D[node_j][node_k]
                            add_terms.append(diff ** 2 / (2 * sigma))

                add_cost = gamma * np.mean(add_terms) if add_terms else 0.0

                branch_lengths = self.estimate_branch_lengths(D, nodes, i, j, C)
                var = self.sigma2t(D[node_i][node_j])

                if var > 0:
                    z = -min(branch_lengths) / np.sqrt(2 * var)
                    pos_cost = -np.log(0.5 * erfc(z))
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
            branch_i = (dij + sum_diffs/weights) / 2.0
            branch_j = dij - branch_i
        else:
            branch_i = branch_j = dij / 2.0

        return [max(0.0, branch_i), max(0.0, branch_j)]

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

def run_experiment_weighbor():
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

    dataset_sizes = [10, 20, 50, 100]
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

                weighbor = Weighbor(sequence_length=500)

                start = time.time()
                E, uD, fake_root = weighbor.build_tree(dist_mat)
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
    run_experiment_weighbor()

if __name__ == "__main__":
    main()