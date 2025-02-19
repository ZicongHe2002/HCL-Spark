import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict


def min_max_scaling(data):

    values = list(data.values())
    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return {key: 0.5 for key in data}

    scaled_data = {key: (value - min_val) / (max_val - min_val) for key, value in data.items()}

    return scaled_data


def count_lines_in_all_unique_files(root_folder):
    """count the diversity"""
    layer_line_counts = defaultdict(int)
    layer_file_counts = defaultdict(int)

    if not os.path.exists(root_folder):
        raise FileNotFoundError(f"the file {root_folder} not exist。")

    for root, _, files in os.walk(root_folder):
        for file_name in files:
            if file_name.startswith("unique") and file_name.endswith(".txt"):
                # 13
                # 7
                match = re.search(r"layer(\d+)", file_name)

                if match:
                    layer_num = int(match.group(1))

                    layer_name = f"layer{match.group(1)}"
                    file_path = os.path.join(root, file_name)
                    try:
                        with open(file_path, "r") as f:
                            line_count = sum(1 for _ in f)
                        layer_line_counts[layer_name] += line_count
                        layer_file_counts[layer_name] += 1
                    except Exception as e:
                        print(f"read file failed: {file_path}, error: {e}")

    return layer_line_counts, layer_file_counts


def count_correct_lines(root_folder):
    """count hallucination"""
    correct_line_counts = defaultdict(int)
    correct_file_counts = defaultdict(int)

    for root, _, files in os.walk(root_folder):
        for file_name in files:
            if file_name.startswith("correct") and file_name.endswith(".txt"):
                # 13
                # 7
                match = re.search(r"layer(\d+)", file_name)

                if match:
                    layer_num = int(match.group(1))
                    layer_name = f"layer{match.group(1)}"
                    file_path = os.path.join(root, file_name)
                    try:
                        with open(file_path, "r") as f:
                            line_count = sum(1 for _ in f)
                        correct_line_counts[layer_name] += line_count
                        correct_file_counts[layer_name] += 1
                    except Exception as e:
                        print(f"cannot read the file: {file_path}, error: {e}")

    correct_averages = {
        layer: correct_line_counts[layer] / correct_file_counts[layer]
        for layer in correct_line_counts
    }
    adjusted_correct_values = {layer: ((50 - avg) / 50) for layer, avg in correct_averages.items()}

    for layer in sorted(adjusted_correct_values.keys(), key=lambda x: int(x.replace("layer", ""))):
        print(f"{layer}: {adjusted_correct_values[layer]:.4f}")

    return adjusted_correct_values


def plot_average_lines(layer_line_counts, layer_file_counts, output_image):
    """plot the diversity"""
    sorted_layers = sorted(layer_line_counts.keys(), key=lambda x: int(x.replace("layer", "")))
    average_lines = [layer_line_counts[layer] / layer_file_counts[layer] for layer in sorted_layers]

    scaled_lines = min_max_scaling(dict(zip(sorted_layers, average_lines)))

    plt.figure(figsize=(15, 6))
    plt.plot(sorted_layers, [scaled_lines[layer] for layer in sorted_layers], marker='o', markersize=8, linestyle='-',
             color='b')
    for i, layer in enumerate(sorted_layers):
        plt.text(layer, scaled_lines[layer], f"{scaled_lines[layer]:.2f}", ha='center', va='bottom', fontsize=9)
    plt.title("diversity")
    plt.xlabel("Layer")
    plt.ylabel("standard_value")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_image)
    plt.show()


def plot_correct_lines(adjusted_correct_values, output_image):
    """the hallucination"""
    sorted_layers = sorted(adjusted_correct_values.keys(), key=lambda x: int(x.replace("layer", "")))
    adjusted_values = [adjusted_correct_values[layer] for layer in sorted_layers]

    plt.figure(figsize=(15, 6))
    plt.plot(sorted_layers, adjusted_values, marker='o', markersize=10, linestyle='-', color='red', linewidth=2)
    for i, val in enumerate(adjusted_values):
        plt.text(sorted_layers[i], val, f"{val:.2f}", ha='center', va='bottom', fontsize=10, color='blue')
    plt.title("hallucination", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("hallucination rate", fontsize=12)
    plt.ylim(0.445, 0.485)  # 设置 y 轴范围缩小
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_image)
    plt.show()


def plot_combined_score(unique_values, correct_values, output_image):
    """
    plot the HCB_score
    """
    sorted_layers = sorted(unique_values.keys(), key=lambda x: int(x.replace("layer", "")))

    combined_scores = {}
    for layer in sorted_layers:
        diversity_score = unique_values[layer] * 0.5
        hallucination_score = (1 - correct_values[layer]) * 0.5
        combined_scores[layer] = diversity_score + hallucination_score

    plt.figure(figsize=(15, 6))
    plt.plot(sorted_layers,
             [combined_scores[layer] for layer in sorted_layers],
             marker='o',
             markersize=10,
             linestyle='-',
             color='green',
             linewidth=2)

    # 添加数值标签
    for i, layer in enumerate(sorted_layers):
        plt.text(layer,
                 combined_scores[layer],
                 f"{combined_scores[layer]:.2f}",
                 ha='center',
                 va='bottom',
                 fontsize=10,
                 color='black')
    for layer in sorted(combined_scores.keys(), key=lambda x: int(x.replace("layer", ""))):
        print(f"{layer}: {combined_scores[layer]:.4f}")

    plt.title("final_score (diversity*0.8 + (1-hallucination)*0.2)", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("score", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_image)
    plt.show()


if __name__ == "__main__":
    unique_folder = "ur_correctAnswers_path/unique_answers"
    correct_folder = "ur_correctAnswers_path"

    try:
        print(f"Starting to count unique files...")
        layer_counts, layer_files = count_lines_in_all_unique_files(unique_folder)

        print("\nResults (unique files):")
        if not layer_counts:
            print("No unique files found, or their contents are empty.")
        for layer, count in layer_counts.items():
            print(f"{layer}: total lines = {count}, number of files = {layer_files[layer]}")

        if layer_counts:
            plot_average_lines(layer_counts, layer_files, "paper_graph/nq/7b/1.0/t_diversity_1b.png")
        else:
            print("Not enough data to generate a chart for unique files.")

        # Count information for correct files
        print(f"\nStarting to count correct files...")
        adjusted_correct_counts = count_correct_lines(correct_folder)

        print("\nResults (correct files):")
        if not adjusted_correct_counts:
            print("No correct files found, or their contents are empty.")
        else:
            for layer, value in adjusted_correct_counts.items():
                print(f"{layer}: adjusted value = {value:.2f}")

        if adjusted_correct_counts:
            plot_correct_lines(adjusted_correct_counts, "paper_graph/nq/7b/1.0/t_hallucination_1b.png")
        else:
            print("Not enough data to generate a chart for correct files.")

        print(f"\nStarting to plot the combined score chart...")
        if layer_counts and adjusted_correct_counts:
            # First, standardize both sets of data individually
            unique_scaled = min_max_scaling(
                {layer: layer_counts[layer] / layer_files[layer] for layer in layer_counts}
            )
            correct_scaled = adjusted_correct_counts

            # Calculate the combined score using the standardized values
            plot_combined_score(unique_scaled, correct_scaled, "paper_graph/nq/7b/1.0/t_final_score_1b.png")
        else:
            print("Not enough data to generate the combined score chart.")

    except Exception as e:
        print(f"An error occurred: {e}")
