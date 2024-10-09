import os
import json

# Maximum number of rows to include in the prepared dataset.
MAX_ROWS = 10000

# Message that is included with every prompt.
SYSTEM_CONTENT = (
    "Determine the failure mode of the observation provided by the user."
)

def prepare_data():
    """Iterate over each dataset (train, validation, and test), converting
    them into JSONL format where each line contains a message with a label, keyword, and response.
    """
    # Get the current working directory
    current_dir = os.getcwd()
    
    datasets = ["train", "validation", "test"]
    
    for ds in datasets:
        dataset = []
        
        # Set the correct file path using the current working directory
        fn = os.path.join(current_dir, f"{ds}_data.txt")
        
        # Read the file and process the data
        with open(fn, "r") as f:
            data = [line.strip().split(":") for line in f.readlines()]
        
        for i, line in enumerate(data[:MAX_ROWS]):
            if len(line) == 3:
                label, keyword, response = line
                messages = [
                    {"role": "system", "content": SYSTEM_CONTENT},
                    {"role": "user", "content": f"Keyword: {keyword.strip()}"},
                    {"role": "assistant", "content": response.strip()}
                ]
                dataset.append({"label": label.strip(), "messages": messages})

        # Write to JSONL file in a 'prepared' directory within the current directory
        prepared_dir = os.path.join(current_dir, "prepared")

        # Ensure the directory exists
        os.makedirs(prepared_dir, exist_ok=True)  
        output_path = os.path.join(prepared_dir, f"{ds}.jsonl")
        
        with open(output_path, "w") as f:
            for row in dataset:
                f.write(json.dumps(row))
                f.write("\n")

def main():
    prepare_data()

if __name__ == "__main__":
    main()
