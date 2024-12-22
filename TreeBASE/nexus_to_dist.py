import argparse
import numpy as np
import re
import pandas as pd

def extract_nexus_sequences_skip_metadata(file_path, skip_rows=5):
    sequences = {}
    with open(file_path, 'r') as file:
        data = file.readlines()[skip_rows:]
    matrix_started = False
    for line in data:
        line = line.strip()
        if line.upper() == "MATRIX":
            matrix_started = True
            continue
        if line.upper() == "END;" and matrix_started:
            break
        if matrix_started and line and not line.startswith('['):
            parts = line.split(None, 1)
            if len(parts) == 2:
                taxon, sequence = parts
                sequences[taxon] = sequence.replace(" ", "")
    if not sequences:
        raise ValueError("No sequences found in the MATRIX section.")
    return sequences

def clean_sequences(sequences):
    cleaned_sequences = {}
    for taxon, sequence in sequences.items():
        cleaned_sequence = re.sub(r"[^ACGT?-]", "", sequence)
        cleaned_sequences[taxon] = cleaned_sequence
    return cleaned_sequences

def align_sequences(sequences, target_length=None):
    if target_length is None:
        target_length = min(len(seq) for seq in sequences.values())
    aligned_sequences = {}
    for taxon, sequence in sequences.items():
        if len(sequence) > target_length:
            aligned_sequence = sequence[:target_length]
        else:
            aligned_sequence = sequence.ljust(target_length, "-")
        aligned_sequences[taxon] = aligned_sequence
    return aligned_sequences

def jukes_cantor_distance(seq1, seq2):
    valid_positions = [(a, b) for a, b in zip(seq1, seq2) if a in "ACGT" and b in "ACGT"]
    if not valid_positions:
        return np.nan
    mismatches = sum(1 for a, b in valid_positions if a != b)
    total_valid_positions = len(valid_positions)
    p = mismatches / total_valid_positions if total_valid_positions > 0 else 0
    if p >= 0.75:
        return np.inf
    return -3/4 * np.log(1 - 4/3 * p)

def compute_jukes_cantor_matrix(sequences):
    taxa = list(sequences.keys())
    num_taxa = len(taxa)
    distance_matrix = np.zeros((num_taxa, num_taxa))
    for i in range(num_taxa):
        for j in range(i, num_taxa):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                dist = jukes_cantor_distance(sequences[taxa[i]], sequences[taxa[j]])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    return taxa, distance_matrix

def replace_nan_with_neutral(matrix, neutral_value=None):
    if neutral_value is None:
        valid_values = matrix[~np.isnan(matrix)]
        neutral_value = np.mean(valid_values) if valid_values.size > 0 else 0
    filled_matrix = np.where(np.isnan(matrix), neutral_value, matrix)
    return filled_matrix

def main():
    parser = argparse.ArgumentParser(description="Compute Jukes-Cantor distance matrix from a NEXUS file.")
    parser.add_argument("input_file", help="Path to the input NEXUS file")
    parser.add_argument("output_file", help="Path to the output file for the distance matrix")
    args = parser.parse_args()

    print("Processing the NEXUS file...")
    sequences = extract_nexus_sequences_skip_metadata(args.input_file)
    cleaned_sequences = clean_sequences(sequences)
    aligned_sequences = align_sequences(cleaned_sequences)
    taxa, distance_matrix = compute_jukes_cantor_matrix(aligned_sequences)
    filled_matrix = replace_nan_with_neutral(distance_matrix)

    distance_df = pd.DataFrame(filled_matrix, index=taxa, columns=taxa)
    distance_df.to_csv(args.output_file, sep='\t', index=True, header=True)
    print(f"Distance matrix saved to {args.output_file}")

if __name__ == "__main__":
    main()