import os
import json

# Maximum number of rows to include in the prepared dataset.
MAX_ROWS = 10000

# Message that is included with every prompt.
SYSTEM_CONTENT = (
    "Determine the content category of the following text."
)

def prepare_data(input_file, output_file):
    """Convert a single text file into a JSON Lines file.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output JSON Lines file.
    """
    dataset = []

    # Read the data from the input text file with specified encoding
    with open(input_file, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines()]

    # Prepare the data for JSON Lines format
    for i, text in enumerate(data[:MAX_ROWS]):
        messages = []
        messages.append({"role": "system", "content": SYSTEM_CONTENT})
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": "Category not assigned"})
        dataset.append({"messages": messages})

    # Write the data to the output JSON Lines file
    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row))
            f.write("\n")

def main():
    input_file = "./scraped_content.txt"
    output_file = "sample.jsonl"
    prepare_data(input_file, output_file)

if __name__ == "__main__":
    main()
