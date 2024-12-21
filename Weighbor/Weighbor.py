#!/usr/bin/env python3
import argparse
from collections import defaultdict
import numpy as np
from scipy.special import erf, erfc

class Weighbor:
    def __init__(self, sequence_length=500, base_types=4):
        self.L = sequence_length
        self.B = bease_types
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

        # Initialize distance matrix and renormalization vector
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

            # Update renormalization vector for the new node
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
        gamma = 1.0 / (N - 3) if N > 4 else 1.0  # Correction factor for additivity

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

def read_data(distances_file):
    """Read distance matrix file"""
    with open(distances_file, "r") as f:
        lines = [l.strip().split() for l in f.readlines()]
        mapping = {i: s for i, s in enumerate(lines[0])}
        lines = [l[1:] for l in lines[1:]]
        D = [[float(sval) for sval in l] for l in lines]
    return D, mapping

def assemble_tree(fake_root, E):
    """Convert edges to tree structure"""
    tree_map = defaultdict(list)
    adjacency_list = defaultdict(list)

    for a, b in E:
        adjacency_list[b].append(a)
        adjacency_list[a].append(b)

    visited_nodes = set()
    stack = [(fake_root, None)]

    while stack:
        current_node, parent_node = stack.pop()
        visited_nodes.add(current_node)
        
        for neighbor in adjacency_list[current_node]:
            if neighbor != parent_node and neighbor not in visited_nodes:
                tree_map[current_node].append(neighbor)
                stack.append((neighbor, current_node))

    return tree_map

def generate_newick(fake_root, tree_map, uD, mapping = None):
    """Generate Newick format string"""
    def newick(node, parent):
        if node not in tree_map or not tree_map[node]:
            node_string = mapping.get(node, str(node))
        else:
            children = [newick(child, node) for child in tree_map[node]]
            node_string = '(%s)' % ','.join(children)
        if parent is None:
            return '%s;' % node_string
        else:
            return '%s:%.6f' % (node_string, uD[node][parent])
    return newick(fake_root, None)

def main():
    parser = argparse.ArgumentParser(
        description='Weighted neighbor-joining algorithm for phylogenetic tree reconstruction')
    parser.add_argument('-f', type=str, default='dist10.txt')
    parser.add_argument('-nwk', type=str, default='tree.nwk')
    parser.add_argument('-o', type=str, default='Green_monkey')
    args = parser.parse_args()
    
    D, mapping = read_data(args.f)
    og = dict(map(reversed, mapping.items()))[args.o]
    
    weighbor = Weighbor()
    E, uD, fake_root = weighbor.build_tree(D, mapping)
    
    mapping[fake_root] = "root"
    tree_map = assemble_tree(fake_root, E)
    nwk_str = generate_newick(fake_root, tree_map, uD, mapping)
    
    print(nwk_str)
    with open(args.nwk, "w") as nwk_file:
        print(nwk_str, file=nwk_file)

if __name__ == "__main__":
    main()