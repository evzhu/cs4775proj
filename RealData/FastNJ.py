#!/usr/bin/env python3

''' Generates a Newick formatted tree using the Neighbor-Joining algorithm, with an outgroup species specified.

Arguments:
    -f: The path to the distance matrix file (a symmetric matrix with 0 on the diagonal).
        (Default: dist10.txt)
    -nwk: The path to the output file to store the nwk format 
        (Default: tree.nwk)
    -o: Name of the outgroup species to root the tree.

Outputs:
    A Newick formatted tree.

Example usage:
    python 1a.py -f dist10.txt -o Green_monkey -nwk tree.nwk


***************** IMPORTANT:   ***************************************************************************************
********    If any, please remove any print statements except the one provided in the main function before    ********
********    submission, as the autograder evaluation based solely on the print output of this script.         ********
**********************************************************************************************************************

'''

import argparse
import time
from collections import defaultdict


''' Reads the input distance file between species, encode the distance matrix into a dictionary
    with the species names encoded as integer indices.

Arguments:
    distances_file: Path to the file containing the distance matrix between species.
Returns:
    D: A dictionary of dictionaries, defining distances between all species, every key is a species index,
        and the corresponding value is a dictionary containing all species indexes as keys. The values of
        these keys are the distance between species. For example {1: {1: 0.0, 2: 1.0}, 2: {1: 1.0, 2: 0.0}}
        defines the distance between two species, 1 and 2.
    mapping: A dictionary mapping indices to species names. For example, {1: 'Chimp', 2: 'Bonobo'}.
'''
def read_data(distances_file):
    with open(distances_file, "r") as f:
        lines = [l.strip().split() for l in f.readlines()]
        mapping = {i: s for i, s in enumerate(lines[0])}
        lines = [l[1:] for l in lines[1:]]
        D = {i: {} for i in range(len(lines))}
        for i, l in enumerate(lines):
            for j, sval in enumerate(l):
                D[i][j] = float(sval)
    return D, mapping


''' Performs the neighbor joining algorithm on the distance matrix and the index of the outgroup species.

Arguments:
    D: A dictionary of dictionaries, defining distances between all species, every key is a species index,
        and the corresponding value is a dictionary containing all species indexes as keys. (See the description
        in `read_data` for details).
    og: outgroup index, defining which species serves as an outgroup.
        A fake root should be inserted in the middle of the pendant edge
        leading to this outgroup node.

Returns:
    E : A list storing the edges chosen from the NJ algorithm in the form of tuples: (index, index). 
        For example [(3,1),(3,2)] represents an unrooted NJ tree of two edges, 
        3<-->1 and 3<-->2, where 1 & 2 are indexes of leaf nodes in the tree,
        and 3 is the index of the internal node you added.
    uD: A dictionary of dictionary, defining distances between all nodes (leaves and internal nodes),
        it's of the same format as D, storing all edge lengths of the NJ tree whose topology is specified by E.
        For example, {1: {1: 0.0, 2: 1.0, 3: 1.5}, 2: {1: 1.0, 2: 0.0, 3: 2.0}, 3: {1: 1.5, 2: 2.0, 3: 0.0}}
        will fully specify the edge lengths for the tree represented by the E example ([(3,1),(3,2)]):
        Length(3<-->1) = 1.5, Length(3<-->2) = 2.0.
    fake_root: which node index represent the root

        *************************************************************************************
        ***** Please note that this return value format is just for guiding purpose,    *****
        ***** and we are not grading based on the output of this function (neighbor_join).***
        ***** Please feel free to use return structure / indexing system you like.      *****
        *************************************************************************************
'''
def neighbor_join(D, og):
    """Performs the Fast Neighbor Joining (FNJ) algorithm on the distance matrix.

    Args:
        D: A dictionary of dictionaries, defining distances between all species.
        og: Outgroup index, defining which species serves as an outgroup.

    Returns:
        E: A list storing the edges chosen from the NJ algorithm in the form of tuples: (index, index).
        uD: Updated distance dictionary defining distances between all nodes.
        fake_root: The node index representing the root.
    """
    n = list(D.keys())
    E = []
    lengths = {}
    uD = {}
    counter = max(D.keys()) + 1

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
                D[z][k] = 0.5 * (D[i][k] + D[j][k] - D[i][j])
                D[k][z] = D[z][k]
        D[z][z] = 0.0

        R[z] = sum(D[z].values())
        R.pop(i, None)
        R.pop(j, None)

        V = {(x, y) for x, y in V if x != i and y != i and x != j and y != j}
        for k in n:
            if k != i and k != j:
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

    i, j = n
    E.append((i, j))
    lengths[(i, j)] = D[i][j]

    out = [edge for edge in E if og in edge]
    if out:
        for edge in out:
            if og in edge:
                start, end = (edge[1], edge[0]) if edge[1] == og else (edge[0], edge[1])
                length = lengths.get(edge, lengths.get((start, end)))
                break

        fake_root = counter
        counter += 1

        E.remove(edge)
        E.append((fake_root, og))
        E.append((fake_root, end))
        lengths[(fake_root, og)] = length / 2
        lengths[(fake_root, end)] = length / 2

    nodes = set(node for edge in E for node in edge)
    uD = {i: {j: 0.0 for j in nodes} for i in nodes}
    for (i, j), dist in lengths.items():
        uD[i][j] = dist
        uD[j][i] = dist

    return E, uD, fake_root






''' Helper function for defining a tree data structure.
    First finds the root node and find its children, and then generates 
    the whole binary tree based on .

Arguments:
    E：A list storing the edges chosen from the NJ algorithm in the form of tuples: (index, index). 
         (See the description in `neighbor_join` for details).
    fake_root: which node index represent the root
Returns:
    tree_map：A dictionary storing the topology of the tree, where each key is a node index and each value is a list of
              node indexs of the direct children nodes of the key, of at most length 2. For example, {3: [1, 2], 2: [], 1: []}
              represents the tree with 1 internal node (3) and two leaves (1, 2). the [] value shows the leaf node status.

    *************************************************************************************
    ***** Please note that this return value format is just for guiding purpose,    *****
    ***** and we are not grading based on the output of this function (assemble_tree).***
    ***** Please feel free to use return structure / indexing system you like.      *****
    *************************************************************************************
'''
def assemble_tree(fake_root, E):
    ''' Complete this function. '''
    tree_map = defaultdict(list)
    adjacency_list = defaultdict(list)

    for a, b in E:
        adjacency_list[b].append(a)
        adjacency_list[a].append(b)
        

    visited_nodes = set()
    stack = [(fake_root, None)]  # (current_node, parent_node)

    while stack:
        current_node, parent_node = stack.pop()
        visited_nodes.add(current_node)
        
        for neighbor in adjacency_list[current_node]:
            if neighbor != parent_node and neighbor not in visited_nodes:
                tree_map[current_node].append(neighbor)
                stack.append((neighbor, current_node))

    return tree_map


''' Returns a string of the Newick tree format for the tree, rooted at a pre-defined node (fake_root).

Arguments:
    fake_root: which node index represent the root
    tree_map：A dictionary storing the topology of the tree (See the description in `assemble_tree` for details).
    uD: A dictionary of dictionary, defining distances between all nodes (leaves and internal nodes)
        (See the description in `neighbor_join` for details)
    mapping: A dictionary mapping indices to species names. (See the description in `read_data` for details)
Returns:
    output: rooted tree in Newick tree format (string). The branch lengths should be in 6 decimal digits.
            For example, you could do "string = '%s:%.6f' % (name, length)"

    *********************************************************************************************
    ***** We will grade on the newick string output by this function (generate_newick).     *****
    *********************************************************************************************
'''
def generate_newick(fake_root, tree_map, uD, mapping = None):
    ''' Complete this function. '''
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
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Neighbor-joining algorithm on a set of n sequences')
    parser.add_argument('-f', action="store", dest="f", type=str, default='dist10.txt')
    parser.add_argument('-nwk', action="store", dest="nwk", type=str, default='tree.nwk')
    parser.add_argument('-o', action="store", dest="o", type=str, default='Green_monkey')
    args = parser.parse_args()
    distances_file = args.f
    og_ = args.o
    nwk_ = args.nwk

    D, mapping = read_data(distances_file)
    og = dict(map(reversed, mapping.items()))[og_]

    E, uD, fake_root = neighbor_join(D, og) # TIP: Please change the arguments to pass and return here if you don't want to follow the provided structure
    mapping[fake_root] = "root"
    tree_map = assemble_tree(fake_root, E) # TIP: Please change the arguments to pass and return here if you don't want to follow the provided structure
    nwk_str = generate_newick(fake_root, tree_map, uD, mapping) # TIP: Please change the arguments to pass and return here if you don't want to follow the provided structure
    
    # Print and save the Newick string.
    ''' 
    ****************************************************************************************
    ***** Please note that we will grade on this print statement, so please make sure  *****
    ***** to delete any print statement except for this given one before submission!!! *****
    ****************************************************************************************
    '''
    print(nwk_str)
    with open(nwk_, "w") as nwk_file:
        print(nwk_str, file=nwk_file)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime: {total_time:.6f} seconds")


if __name__ == "__main__":
    main()
