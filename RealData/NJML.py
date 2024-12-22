import argparse
from collections import defaultdict
import random
import time


def parse_distance_matrix(file_path):
    """
    Parses a distance matrix file into a nested dictionary.
    """
    with open(file_path, "r") as f:
        lines = [line.strip().split() for line in f.readlines()]

    labels = lines[0]
    matrix = lines[1:]

    distance_matrix = {}
    for i, row in enumerate(matrix):
        row_label = row[0]
        distance_matrix[row_label] = {}
        for j, value in enumerate(row[1:]):
            distance_matrix[row_label][labels[j]] = float(value)
    return distance_matrix, labels


def neighbor_joining_root_fix(distance_matrix):
    """
    Constructs a Neighbor-Joining (NJ) tree while avoiding root branch length.
    """
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

        new_node_label = f"Node{counter}"
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

    return tree, branch_lengths


def final_generate_newick_ignore_root_length(tree, lengths, root=None):
    """
    Final Newick generation that explicitly ignores any branch lengths for the root node.
    """
    adjacency_list = defaultdict(list)
    for parent, child in tree:
        adjacency_list[parent].append(child)
        adjacency_list[child].append(parent)

    if root is None:
        root = tree[0][0]

    def dfs(node, parent):
        children = [child for child in adjacency_list[node] if child != parent]
        if not children:
            branch_length = lengths.get((node, parent), lengths.get((parent, node), 0.0))
            return f"{node}:{branch_length:.6f}"
        else:
            branches = [dfs(child, node) for child in children]
            if parent:  
                branch_length = lengths.get((node, parent), lengths.get((parent, node), 0.0))
                return f"({','.join(branches)}):{branch_length:.6f}"
            else:  
                return f"({','.join(branches)})"  

    return dfs(root, None) + ";"


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="NJML Phylogenetic Tree Construction")
    parser.add_argument("-f", "--file", required=True, help="Input distance matrix file")
    parser.add_argument("-o", "--output", default="tree.nwk", help="Output Newick file")
    args = parser.parse_args()

    distance_matrix, labels = parse_distance_matrix(args.file)
    distance_dict = {
        labels[i]: {labels[j]: distance_matrix[labels[i]][labels[j]] for j in range(len(labels))}
        for i in range(len(labels))
    }

    nj_tree, lengths = neighbor_joining_root_fix(distance_dict)

    newick_tree = final_generate_newick_ignore_root_length(nj_tree, lengths)

    with open(args.output, "w") as f:
        f.write(newick_tree)

    print(f"Newick tree saved to {args.output}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime: {total_time:.6f} seconds")


if __name__ == "__main__":
    main()
