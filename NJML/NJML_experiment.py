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
    A fully rigorous birth-death simulator is more complex, but we simplify:
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

    # Keep adding lineages until we get num_taxa
    while len(nodes) < num_taxa:
        nodes.append(next_id)
        next_id += 1

    # Now pair them up to create internal nodes
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
        for j in range(i + 1, n):
            seq_i = leaf_seqs[leaves[i]]
            seq_j = leaf_seqs[leaves[j]]
            p = sum(a != b for a, b in zip(seq_i, seq_j)) / len(seq_i)
            if p < 0.75:
                dist = -0.75 * math.log(1 - (4.0/3.0)*p)
            else:
                dist = 2.0  # Saturated
            dist_mat[i][j] = dist
            dist_mat[j][i] = dist
    return dist_mat


################################################
# (4) NJML ALGORITHM (SIMPLIFIED)
################################################
def njml_inference(distance_matrix):
    """
    A *simplified demonstration* of an NJML-like method:
      1) Build an initial NJ tree (via neighbor_joining_root_fix).
      2) (Placeholder) "ML refinement" would go here.
         For demonstration, we skip the refine step and treat it as done.
      3) Return final edges and branch lengths in a structure that we can use.
    """
    # Convert the user-provided data structure to the code snippet's expected format
    # that is parseable by neighbor_joining_root_fix.
    # Keys must be strings. Already the user code handles dict-of-dict with strings.
    # So we can pass distance_matrix directly if it is labeled by strings.
    from collections import defaultdict

    # Step 1: Build the NJ Tree
    # We'll reuse the user’s neighbor_joining_root_fix function inlined here
    tree, lengths = neighbor_joining_root_fix(distance_matrix)

    return tree, lengths


def neighbor_joining_root_fix(distance_dict):
    """
    Provided user code for building a neighbor-joining tree, ignoring root branch length.
    Returns:
      tree: list of edges (parent, child) or (child, parent)
      lengths: dict of edge -> branch length
    """
    ids = list(distance_dict.keys())
    n = len(ids)
    active_nodes = {i: ids[i] for i in range(n)}  # Map indices to node labels

    tree = []
    branch_lengths = {}
    counter = n

    # Convert distance matrix to numerical form with integer keys
    D = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(n):
            D[i][j] = distance_dict[ids[i]][ids[j]]

    while len(D) > 2:
        # Compute Q-matrix
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

        new_node_label = f"Node{counter}"
        tree.append((active_nodes[i], new_node_label))
        tree.append((active_nodes[j], new_node_label))
        branch_lengths[(active_nodes[i], new_node_label)] = new_node_distance_i
        branch_lengths[(active_nodes[j], new_node_label)] = new_node_distance_j

        # Update distances for new node
        new_distances = {}
        for k in D:
            if k != i and k != j:
                new_distances[k] = 0.5 * (D[i][k] + D[j][k] - D[i][j])

        D[counter] = {}
        for k in new_distances:
            D[counter][k] = new_distances[k]
            D[k][counter] = new_distances[k]
        D[counter][counter] = 0.0

        # Remove old nodes
        del D[i]
        del D[j]
        for k in list(D.keys()):
            if i in D[k]:
                del D[k][i]
            if j in D[k]:
                del D[k][j]

        active_nodes[counter] = new_node_label
        del active_nodes[i]
        del active_nodes[j]

        counter += 1

    # Connect the last two nodes without assigning branch length to the root
    keys = list(D.keys())
    if len(keys) == 2:
        i, j = keys
        tree.append((active_nodes[i], active_nodes[j]))
        # No branch length for the final connection (root edge)
    elif len(keys) == 1:
        # If only one key remains, treat it as a leaf / single node
        pass

    return tree, branch_lengths


################################################
# (5) ASSEMBLE TREE & GENERATE NEWICK
################################################
def assemble_tree(inferred_tree, lengths):
    """
    Build adjacency from (parent, child) edges. We do not have a single 'root' label, 
    but we can pick the first item in the tree or an internal node with no parent.
    """
    adjacency_list = defaultdict(list)
    for (p, c) in inferred_tree:
        adjacency_list[p].append(c)
        adjacency_list[c].append(p)

    # Attempt to guess a root: pick the first 'internal node' we see
    internal_nodes = set(edge[1] for edge in inferred_tree if edge[1].startswith("Node"))
    if internal_nodes:
        root = next(iter(internal_nodes))
    else:
        # fallback to the first node in the adjacency list
        root = next(iter(adjacency_list))

    return adjacency_list, root

def generate_newick(adjacency_list, lengths, root):
    """
    Minimal DFS-based Newick generator. 
    Root has no parent in this representation, so we skip its branch length.
    """
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        children = [c for c in adjacency_list[node] if c != parent]
        if not children:
            # Leaf node
            brlen = lengths.get((node, parent), lengths.get((parent, node), 0.0)) if parent else 0.0
            return f"{node}:{brlen:.6f}"
        else:
            # Internal node
            subtrees = [dfs(child, node) for child in children]
            node_string = f"({','.join(subtrees)})"
            if parent is None:
                # Root, no branch length
                return node_string
            else:
                brlen = lengths.get((node, parent), lengths.get((parent, node), 0.0))
                return f"{node_string}:{brlen:.6f}"

    newick_str = dfs(root, None) + ";"
    return newick_str


################################################
# (6) SIMPLE ROBINSON-FOULDS (RF) DISTANCE
################################################
def compute_rf_distance(true_parent_map, inferred_tree, leaves, root_id):
    """
    A minimal bipartition-based approach. The 'NJML' result is in the form of edges (p->c).
    We will guess the largest 'NodeX' or an internal node as root for the inferred tree.
    """
    true_bip = get_bipartitions(true_parent_map, leaves, root_id)

    # Convert inferred edges to a synthetic parent_map. 
    # We'll pick the largest 'NodeX' as root or pick the first Node we see if not labeled.
    # Then gather bipartitions in the same manner.
    # We do a naive approach: if we see an edge (a, NodeK), treat NodeK as parent if NodeK has "Node" prefix.
    # This is just for demonstration.

    # Build adjacency
    adjacency = defaultdict(list)
    for (p, c) in inferred_tree:
        adjacency[p].append(c)
        adjacency[c].append(p)

    # Find a root candidate
    all_nodes = set(adjacency.keys())
    node_labels = [n for n in all_nodes if n.startswith("Node")]
    if node_labels:
        # pick the node with the highest number in "NodeXXX"
        node_ids = [int(n[4:]) for n in node_labels if n[4:].isdigit()]
        if node_ids:
            candidate_root_id = max(node_ids)
            root_label = f"Node{candidate_root_id}"
        else:
            root_label = node_labels[0]
    else:
        # fallback to an arbitrary node
        root_label = next(iter(adjacency))

    # Convert edges -> parent_map with root as root_label
    inferred_pm = build_inferred_parent_map(adjacency, root_label)

    # Compute bipartitions
    inf_bip = get_bipartitions(inferred_pm, leaves, root_label)

    return len(true_bip ^ inf_bip)

def get_bipartitions(parent_map, leaves, root):
    """
    Collect bipartitions by doing a DFS from root, forming sets of leaves under each branch.
    """
    children_map = defaultdict(list)
    for child, (p, _) in parent_map.items():
        children_map[p].append(child)

    bip_set = set()

    def gather_leaves(node):
        if node in leaves:
            return {node}
        out = set()
        for ch in children_map[node]:
            out |= gather_leaves(ch)
        return out

    def traverse(node):
        for ch in children_map[node]:
            cset = gather_leaves(ch)
            if 0 < len(cset) < len(leaves):
                bip_set.add(frozenset(cset))
            traverse(ch)

    traverse(root)
    return bip_set

def build_inferred_parent_map(adjacency, root):
    """
    Convert adjacency to parent_map {child: (parent, branch_length)} using a DFS from root.
    We'll store arbitrary branch lengths = 0.1. 
    """
    parent_map = {}
    visited = set()
    stack = [(root, None)]
    while stack:
        node, parent = stack.pop()
        visited.add(node)
        if parent is not None:
            parent_map[node] = (parent, 0.1)
        for nbr in adjacency[node]:
            if nbr not in visited:
                stack.append((nbr, node))
    return parent_map


################################################
# (7) RUN EXPERIMENT (SIMULATED DATASETS)
################################################
def run_experiment_njml():
    dataset_sizes = [10, 20, 50, 100, 200]
    replicates = 3
    results = {}

    for n in dataset_sizes:
        runtimes = []
        rfs = []
        for rep in range(replicates):
            # 1. Simulate a random tree
            parent_map, leaves, root_id = simulate_birth_death_tree(num_taxa=n)

            # 2. Simulate sequences
            leaf_seqs = simulate_sequences(parent_map, leaves, root_id, seq_length=500, mu=1e-3)

            # 3. Compute distance matrix
            dist_mat = compute_distance_matrix(leaf_seqs, leaves)

            # 4. Convert to dict-of-dicts with string labels
            #    Example label: str(leaf_i) or 'Taxon_i'
            label_map = {i: f"Taxon_{i}" for i in range(n)}
            distance_dict = {}
            for i in range(n):
                distance_dict[label_map[i]] = {}
                for j in range(n):
                    distance_dict[label_map[i]][label_map[j]] = dist_mat[i][j]

            # 5. Run the "NJML" inference
            start_time = time.time()
            inferred_tree, inferred_lengths = njml_inference(distance_dict)
            elapsed = time.time() - start_time

            # 6. Compute RF distance
            rf_dist = compute_rf_distance(parent_map, inferred_tree, leaves, root_id)

            runtimes.append(elapsed)
            rfs.append(rf_dist)

        # Summarize
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
        python njml_experiment.py
    """
    parser = argparse.ArgumentParser(description="NJML Experiment Script")
    args = parser.parse_args()

    run_experiment_njml()

if __name__ == "__main__":
    main()
