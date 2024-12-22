import matplotlib.pyplot as plt
import numpy as np

# Data for the algorithms
data = {
    "Algorithm": ["BioNJ", "FastNJ", "NJ", "NJML", "Weighbor"],
    "Taxa": [10, 20, 50, 100, 200],
    "BioNJ": {
        "runtime": [0.00, 0.00, 0.05, 0.79, 11.21],
        "rf": [14.67, 34.67, 95.33, 194.67, 394.00],
        "ble": [1.8572, 0.8562, 1.1932, 0.8521, 0.9895]
    },
    "FastNJ": {
        "runtime": [0.00, 0.00, 0.01, 0.11, 0.87],
        "rf": [15.33, 34.00, 96.00, 196.00, 395.33],
        "ble": [0.6236, 0.7448, 1.1364, 1.2177, 1.5285]
    },
    "NJ": {
        "runtime": [0.00, 0.01, 0.02, 0.16, 1.35],
        "rf": [16.00, 36.00, 96.00, 196.00, 396.00],
        "ble": [1.2172, 1.0484, 1.2105, 1.8524, 1.2113]
    },
    "NJML": {
        "runtime": [0.00, 0.00, 0.01, 0.10, 0.75],
        "rf": [14.33, 34.33, 94.33, 194.33, 394.33],
        "ble": [1.2935, 1.0133, 1.1401, 1.7287, 1.3742]
    },
    "Weighbor": {
        "runtime": [0.00, 0.05, 2.07, 37.92, np.nan],
        "rf": [14.33, 34.33, 94.33, 193.67, np.nan],
        "ble": [0.5928, 0.7014, 0.9266, 1.1841, np.nan]
    }
}

def plot_graphs():
    taxa = data["Taxa"]

    # Runtime Plot
    plt.figure(figsize=(10, 5))
    for algo in data["Algorithm"]:
        # Replace runtime values of 0 with 0.01 for plotting
        runtime_adjusted = [0.01 if val == 0 else val for val in data[algo]["runtime"]]
        plt.plot(
            taxa, runtime_adjusted, marker='o', label=algo
        )
    plt.title("Average Runtime vs. Taxa (Log-Log Scale, 0 Adjusted to 0.01)")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Taxa (Log Scale)")
    plt.ylabel("Runtime (seconds, Log Scale)")
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("runtime_vs_taxa.png")
    plt.close()

    # RF Distance Plot
    plt.figure(figsize=(10, 5))
    for algo in data["Algorithm"]:
        plt.plot(
            taxa, data[algo]["rf"], marker='o', label=algo
        )
    plt.title("RF Distance vs. Taxa (Log-Log Scale)")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Taxa (Log Scale)")
    plt.ylabel("Average RF Distance (Log Scale)")
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("rf_distance_vs_taxa.png")
    plt.close()

    # BLE Plot
    plt.figure(figsize=(10, 5))
    for algo in data["Algorithm"]:
        plt.plot(
            taxa, data[algo]["ble"], marker='o', label=algo
        )
    plt.title("Branch Length Error (BLE) vs. Taxa (Log-Log Scale)")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Taxa (Log Scale)")
    plt.ylabel("Average BLE (Log Scale)")
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("ble_vs_taxa.png")
    plt.close()

if __name__ == "__main__":
    plot_graphs()
