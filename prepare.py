import os
import json

# Maximum number of rows to include in the prepared dataset for each file.
MAX_ROWS = 10000

# Message that is included with every prompt.
SYSTEM_CONTENT = (
    "Understand and respond to the user's intent based on the keywords."
)

def prepare_data(input_file, output_file):
    """Convert a single text file into a JSON Lines file.
    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output JSON Lines file.
    """
    current_dir = os.getcwd()  # Get the current working directory
    dataset = []  # Initialize dataset list

    # Read the data from the input text file with specified encoding
    with open(input_file, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines()]

    for text in data[:MAX_ROWS]:
        parts = text.split(':', 2)  # Split the line into exactly three parts
        if len(parts) < 3:
            continue  # Skip lines that do not have at least three parts

        label_category, keywords, response = parts

        messages = []
        messages.append({"role": "system", "content": SYSTEM_CONTENT})
        messages.append({"role": "user", "content": f"Intent: {label_category.strip()}, Keywords: {keywords.strip()}"})
        messages.append({"role": "assistant", "content": response.strip()})
        dataset.append({"messages": messages})

    # Prepare output directory inside the current directory
    prepared_dir = os.path.join(current_dir, "prepared")
    os.makedirs(prepared_dir, exist_ok=True)  # Ensure the directory exists
    output_path = os.path.join(prepared_dir, output_file)  # Define the full output path

    print(f"Saving files to: {output_path}")  # Log the path to check where files are being saved

    # Write the data to the output JSON Lines file
    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row))
            f.write("\n")

def main():
    # Define input and output files
    file_mapping = {
        "train_data.txt": "train.jsonl",
        "test_data.txt": "test.jsonl",
        "val_data.txt": "val.jsonl"
    }

    # Process each file pair
    for input_file, output_file in file_mapping.items():
        prepare_data(input_file, output_file)

if __name__ == "__main__":
    main()
