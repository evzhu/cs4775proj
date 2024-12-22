from Bio import Phylo
import matplotlib.pyplot as plt

# Read the Newick tree from the file
tree = Phylo.read('tree.nwk', 'newick')

# Create a figure and add a subplot with specified size and resolution
fig = plt.figure(figsize=(12, 6), dpi=100)
ax = fig.add_subplot(1, 1, 1)

# Define a function to format branch labels with 6 decimal places
def format_branch_length(clade):
    if clade.branch_length is not None:
        return f'{clade.branch_length:.6f}'  # Ensure 6 decimal places
    else:
        return ''

# Draw the tree with branch lengths as labels
Phylo.draw(tree, axes=ax, do_show=False, branch_labels=format_branch_length)

# Set title and axis labels
ax.set_title("Phylogenetic Tree of Primate Species", fontsize=16)
ax.set_xlabel("Branch Length", fontsize=12)
ax.set_ylabel("Taxa", fontsize=12)

# Adjust tick label size and increase the font size for branch lengths for clarity
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# Display the plot
plt.show()
