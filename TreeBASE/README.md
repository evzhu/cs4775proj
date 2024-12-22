# Extracting Distance Matrix Data from TreeBASE

## Overview
This guide explains how to repoduce the real biological data computations in our project. It walks through extracting and processing phylogenetic distance matrices from TreeBASE using a provided Python script (`nexus_to_dist.py`). The goal is to compute Jukes-Cantor distance matrices from phylogenetic datasets stored in NEXUS format.

## Setup
### 1. Install Dependencies
To install the necessary Python libraries, run:
```bash
pip install numpy pandas
```

### 2. Prepare Input Data
Obtain a dataset in NEXUS format from TreeBASE. You can download a NEXUS file from TreeBASE directly using links such as:
- [Study ID 11474](https://treebase.org/treebase-web/search/study/summary.html?id=11474)
- [Study ID 16027](https://treebase.org/treebase-web/search/study/summary.html?id=16027)

Go to the *Matrices* section and download the sequence NEXUS data. For our project, we used M8736 for Study ID 11474 and M23070 for Study ID 16027.

Save the NEXUS file to your working directory.

---

## Instructions

### 1. Run the Script
To compute the Jukes-Cantor distance matrix from a NEXUS file:
```bash
python nexus_to_dist.py <input_file.nexus> <output_file.txt>
```
#### Arguments:
- `<input_file.nexus>`: Path to the input NEXUS file
- `<output_file.txt>`: Path to save the resulting distance matrix

Example:
```bash
python nexus_to_dist.py example_data.nexus distances_output.txt
```

### 2. Script Description
The `nexus_to_dist.py` script performs the following tasks:
1. Extracts DNA sequences from the `MATRIX` section of the NEXUS file.
2. Cleans sequences to remove invalid characters.
3. Aligns sequences to a uniform length (truncated or padded).
4. Computes pairwise Jukes-Cantor distances.
5. Saves the resulting distance matrix in a tab-delimited file.

#### Handling Missing Data
The script replaces missing or invalid values in the distance matrix with a neutral value (mean of valid distances). This ensures compatibility for downstream analyses.

#### NEXUS File Format
Ensure the NEXUS file adheres to the expected format, with a valid `MATRIX` section. If errors arise, check the file structure and clean metadata.


### 3. Verify Output
The output file is a tab-delimited distance matrix with rows and columns labeled by taxon names. Example from `51taxa_distances.txt`:
```
Taxon1  Taxon2  Taxon3
0.0     0.035   0.036
0.035   0.0     0.034
0.036   0.034   0.0
```

This format can be directly used by our NJ-variant implementations.

---

# Extracting Newick Tree Data from TreeBASE

The tree data of a given study can be found on TreeBASE at the *Trees* tab. 

For our project, we used Tr44921 for Study ID 11474 and Tr23070 for Study ID 16027.

## Extracting the Tree

To locate and extract the phylogenetic tree from the NEXUS file:

1. Open the NEXUS file in a text editor.

2. Scroll to the section beginning with the keyword TREE (e.g., TREE tree1 = [&U] ((A,B),(C,D));).

3. Copy the tree data, which is typically represented in Newick format.

4. Save the tree data to a separate file.

The tree section may look like this:

```
BEGIN TREES;
	TREE tree1 = [&U] ((A,B),(C,D));
END;
```

The content after TREE is the Newick representation of the tree.

---

# Performing Analysis

See folder `RealData` on how we analyzed the data we extracted from TreeBase.
