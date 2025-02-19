from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import os
import glob

# Load the SentenceTransformer model for similarity calculation
st_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_answers_from_txt(file_path):
    """
    Load answers from a text file, one answer per line.
    Empty lines will be removed.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        answers = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines
    return answers

def find_txt_files(base_folder, pattern="correct_*.txt"):
    """
    Recursively search for the specified .txt files in all subfolders.
    """
    return glob.glob(os.path.join(base_folder, "**", pattern), recursive=True)

# Set the path to the folder containing all the correct answer txt files
# (Please replace this path with your own folder path)
base_folder_path = "../np_correct_answer_8b_1.0"

# Get all txt files (including subfolders) under the main folder
txt_files = find_txt_files(base_folder_path)

# Traverse each txt file
for txt_file in txt_files:
    # Load all answers from the current file
    correct_answers = load_answers_from_txt(txt_file)

    # Used to store unique correct answers and their frequencies
    unique_correct_answers = []
    answer_counts = defaultdict(int)

    # Perform similarity analysis for each correct answer
    correct_answers_embeddings = {
        answer: st_model.encode(answer, convert_to_tensor=True)
        for answer in correct_answers
    }

    for answer, answer_embedding in correct_answers_embeddings.items():
        is_unique = True

        # Compare the similarity of the current answer with known unique answers
        for unique_answer in unique_correct_answers:
            similarity = util.cos_sim(answer_embedding, correct_answers_embeddings[unique_answer]).item()

            # If the answer is similar to an existing unique answer, update count and skip
            # Similarity threshold is 0.8; adjust if needed
            if similarity >= 0.8:
                answer_counts[unique_answer] += 1
                is_unique = False
                break

        # If the answer is new, add it to the unique answers list and initialize the count
        if is_unique:
            unique_correct_answers.append(answer)
            answer_counts[answer] = 1

    # Get the relative path of the original file
    relative_path = os.path.relpath(txt_file, base_folder_path)
    relative_folder = os.path.dirname(relative_path)

    # Build the output folder and filename
    output_folder = os.path.join(base_folder_path, "unique_answers", relative_folder)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"unique_{os.path.basename(txt_file)}")

    # Write the unique correct answers to the output file
    with open(output_file, "w", encoding="utf-8") as out_f:
        for answer in unique_correct_answers:
            out_f.write(f"{answer}\n")

    # Output the processing result
    print(f"Processed file: {txt_file}")
    print(f"Unique correct answers saved to: {output_file}")
    print("-" * 40)