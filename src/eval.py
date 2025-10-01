"""
eval.py

Evaluation script for processing a batch of multi-agent debate transcripts.
This script iterates through an experiment folder, calculates advanced metrics for each
transcript using metrics.py, and saves the consolidated results to a single JSON file.
"""

import argparse
import json
import os

import torch
from metrics import run_all_metrics

# Define the base directory for experiment transcripts
BASE_TRANSCRIPT_DIR = "../experiment_transcripts"


def append_result_to_json(result_data: dict, output_filepath: str):
    """
    Safely appends a new result dictionary to a JSON file.
    The file is structured as a list of results.
    """
    existing_results = []
    # Load existing data if the file already exists and is not empty
    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            # Ensure the loaded data is a list
            if not isinstance(existing_results, list):
                print(f"Warning: Output file '{output_filepath}' does not contain a JSON list. Overwriting.")
                existing_results = []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode existing JSON from '{output_filepath}'. Overwriting.")
            existing_results = []

    # Append the new result
    existing_results.append(result_data)

    # Write the updated list back to the file
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"FATAL: Could not write to output file '{output_filepath}'. Error: {e}")


def evaluate_experiment_folder(experiment_folder: str, output_file: str):
    """
    Processes all debate transcripts in a given experiment folder.

    Args:
        experiment_folder (str): The relative path to the experiment folder
                                 (e.g., 'Llama_3.2_3B_Instruct/test_transcript').
        output_file (str): The name of the JSON file to save results to.
    """
    # Construct the full path to the directory
    full_experiment_path = os.path.join(BASE_TRANSCRIPT_DIR, experiment_folder)

    if not os.path.isdir(full_experiment_path):
        print(f"Error: Experiment directory not found at '{full_experiment_path}'")
        return

    print(f"🔍 Starting evaluation for experiment: {experiment_folder}")
    print(f"💾 Results will be saved to: {output_file}")

    # Clear the output file at the beginning of a new run to avoid appending to old results
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Cleared existing results file: {output_file}")

    # Find all JSON transcript files in the directory
    transcript_files = [f for f in os.listdir(full_experiment_path) if f.endswith('.json')]
    
    if not transcript_files:
        print("No JSON transcripts found in the specified folder.")
        return

    total_files = len(transcript_files)
    for i, filename in enumerate(transcript_files):
        transcript_path = os.path.join(full_experiment_path, filename)
        print(f"\nProcessing file {i+1}/{total_files}: {filename}...")

        try:
            # 1. Load the transcript data
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)

            # 2. Run all metrics using the new orchestrator function
            metrics_results = run_all_metrics(transcript_data)

            # 3. Prepare the final result object for this transcript
            final_result = {
                "source_transcript": filename,
                "topic": transcript_data.get("metadata", {}).get("topic", "Unknown Topic"),
                "model": transcript_data.get("metadata", {}).get("llm_model", "Unknown Model"),
                "metrics": metrics_results
            }

            # 4. Append the result to the output file immediately
            append_result_to_json(final_result, output_file)
            print(f"✅ Successfully processed and saved results for {filename}")

        except json.JSONDecodeError:
            print(f"❌ Error: Could not parse JSON from file: {filename}. Skipping.")
        except Exception as e:
            print(f"❌ An unexpected error occurred while processing {filename}: {e}. Skipping.")
        
        finally:
            # 5. Clear GPU cache to prevent memory overflow
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared PyTorch GPU cache.")

    print(f"\n🎉 Evaluation complete. All results saved in '{output_file}'.")


def main():
    """Main function to parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Run evaluation on a folder of debate transcripts.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--experiment-folder",
        required=True,
        type=str,
        help="Path to the experiment folder relative to the `experiment_transcripts` directory.\n"
             "Example: Llama_3.2_3B_Instruct/my_test_run"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.json",
        help="Name of the output JSON file to store aggregated metrics. Defaults to 'evaluation_results.json'."
    )

    args = parser.parse_args()
    output_file = os.path.join("../data/results", args.output_file)

    evaluate_experiment_folder(args.experiment_folder, output_file)


if __name__ == "__main__":
    main()