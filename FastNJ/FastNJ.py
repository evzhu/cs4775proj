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

    # Initialize row sums for FNJ
    R = {i: sum(D[i].values()) for i in D}

    # Initialize the visible set
    V = set()
    for i in D:
        for j in D:
            if i != j:
                V.add((i, j))

    while len(n) > 2:
        # Find the pair with the minimum NJ function in the visible set
        (i, j) = min(V, key=lambda x: (len(n) - 2) * D[x[0]][x[1]] - R[x[0]] - R[x[1]])

        # Calculate branch lengths
        r_i = R[i]
        r_j = R[j]
        iz_dist = 0.5 * D[i][j] + 0.5 * (r_i - r_j) / (len(n) - 2)
        jz_dist = D[i][j] - iz_dist

        # Create new internal node
        z = counter
        counter += 1

        # Update the distance matrix
        D[z] = {}
        for k in n:
            if k != i and k != j:
                D[z][k] = 0.5 * (D[i][k] + D[j][k] - D[i][j])
                D[k][z] = D[z][k]
        D[z][z] = 0.0

        # Update row sums
        R[z] = sum(D[z].values())
        R.pop(i, None)
        R.pop(j, None)

        # Update visible set
        V = {(x, y) for x, y in V if x != i and y != i and x != j and y != j}
        for k in n:
            if k != i and k != j:
                V.add((z, k))

        # Update edges and lengths
        E.append((z, i))
        E.append((z, j))
        lengths[(z, i)] = iz_dist
        lengths[(z, j)] = jz_dist

        # Remove merged nodes
        del D[i]
        del D[j]
        n.remove(i)
        n.remove(j)
        n.append(z)

    # Add the last edge
    i, j = n
    E.append((i, j))
    lengths[(i, j)] = D[i][j]

    # Handle the outgroup
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

    # Prepare the final distance dictionary
    nodes = set(node for edge in E for node in edge)
    uD = {i: {j: 0.0 for j in nodes} for i in nodes}
    for (i, j), dist in lengths.items():
        uD[i][j] = dist
        uD[j][i] = dist

    return E, uD, fake_root