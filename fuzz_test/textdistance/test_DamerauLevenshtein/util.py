import os
import json
from tqdm import tqdm
from collections import defaultdict

def analyze_test_results(logs_dir="logs"):
    """
    Analyze test results by comparing `mutations1_<suffix>.json` 
    and `original_<suffix>.json` files in the logs directory.
    Count the number of passing and failing tests.
    """
    # Dictionary to store counts for passing and failing tests
    results_summary = defaultdict(int)

    # Get all files in the logs directory
    all_files = [f for f in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, f))]

    # Separate mutation and original files by suffix
    mutation_files = [f for f in all_files if f.startswith("mutation5_")]
    original_files = [f for f in all_files if f.startswith("original_")]

    # Process each mutation file and find its matching original file
    for mutation_file in tqdm(mutation_files, desc="Analyzing logs"):
        suffix = mutation_file[len("mutation5_"):]
        original_file = f"original_{suffix}"

        if original_file not in original_files:
            print(f"Warning: No matching original file for {original_file}")
            continue

        # Load JSON data from both files
        with open(os.path.join(logs_dir, mutation_file), "r") as m_file:
            mutation_data = json.load(m_file)

        with open(os.path.join(logs_dir, original_file), "r") as o_file:
            original_data = json.load(o_file)

        # Compare the `output` field
        if mutation_data.get("output") == original_data.get("output"):
            results_summary["passing"] += 1
            mutation_data["verdict"] = "pass"
        else:
            results_summary["failing"] += 1
            mutation_data["verdict"] = "fail"

        # Save the updated mutation data
        with open(os.path.join(logs_dir, mutation_file), "w") as m_file:
            json.dump(mutation_data, m_file, indent=4)

    # Print results summary
    print("\nResults Summary:")
    print(f"Passing Tests: {results_summary['passing']}")
    print(f"Failing Tests: {results_summary['failing']}")

    return results_summary


if __name__ == "__main__":
    # Analyze test results in the default "logs" directory
    analyze_test_results("logs")