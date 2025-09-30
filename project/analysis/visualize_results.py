
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
RESULTS_FILE = '/home/dex/Open CV project/project/analysis/results.json'
OUTPUT_DIR = '/home/dex/Open CV project/project/analysis/plots'

# --- Main Visualization ---
def main():
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    for dataset in ['lightbox', 'sunlamp']:
        print(f'Visualizing results for {dataset} dataset...')
        translation_errors = [r['translation_error'] for r in results[dataset]]
        attitude_errors = [r['attitude_error'] for r in results[dataset]]

        # --- Summary Statistics ---
        print(f'  Translation Error (m):')
        print(f'    Mean: {np.mean(translation_errors):.4f}')
        print(f'    Median: {np.median(translation_errors):.4f}')
        print(f'    Std Dev: {np.std(translation_errors):.4f}')

        print(f'  Attitude Error (deg):')
        print(f'    Mean: {np.mean(attitude_errors):.4f}')
        print(f'    Median: {np.median(attitude_errors):.4f}')
        print(f'    Std Dev: {np.std(attitude_errors):.4f}')

        # --- Histograms ---
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(translation_errors, bins=50)
        plt.title(f'{dataset.capitalize()} Translation Error')
        plt.xlabel('Error (m)')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.hist(attitude_errors, bins=50)
        plt.title(f'{dataset.capitalize()} Attitude Error')
        plt.xlabel('Error (deg)')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f'{dataset}_error_histograms.png')
        plt.savefig(plot_path)
        print(f'  Saved histogram to {plot_path}')

if __name__ == '__main__':
    main()
