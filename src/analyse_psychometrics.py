# analyze_psychometrics.py
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- CONFIGURATION ---
# Base directory where all individual transcript files are stored.
# Update this if your structure is different.
BASE_TRANSCRIPT_DIR = "../experiment_transcripts/openai_gpt_oss_20b/cmv_100_5_moderator_consensus_builder" # e.g., "experiment_transcripts/Llama_3.2_3B_Instruct/test_transcript"
PLOTS_DIRECTORY = "../data/plots"
model_name = "gpt-oss"
rounds = 5

# --- DATA EXTRACTION ---

def extract_psychometric_data(transcript_data: dict) -> dict:
    """
    Extracts and aggregates psychometric metrics for each agent from a single transcript.
    """
    # Dynamically find agent names and assign personas based on your common setup
    agent_names = transcript_data.get("metadata", {}).get("agent_names", ["A1", "A2"])
    agent_personas = {
        agent_names[0]: "D1-Contrarian", #"Evidence-Driven Analyst",
        agent_names[1]: "D2-Contrarian" #"Values-Focused Ethicist"
    }

    agent_data = {
        agent: {
            "confidence_scores": [], "cognitive_efforts": [], "argument_complexities": [],
            "rebuttal_rates": [], "empathy_scores": [], "perspective_taking_accuracies": [],
            "common_ground_ids": [], "cognitive_dissonances": [], "confidence_changes": []
        } for agent in agent_names
    }

    psych_analysis = transcript_data.get("psychometric_analysis", {})

    for round_data in psych_analysis.get("rounds", []):
        for arg_data in round_data.get("arguments", []):
            agent = arg_data.get("agent")
            if agent in agent_data:
                struct_an = arg_data.get("structured_analysis", {})
                agent_data[agent]["confidence_scores"].append(struct_an.get("confidence_score"))
                agent_data[agent]["cognitive_efforts"].append(struct_an.get("cognitive_effort"))
                
                comp_met = arg_data.get("complexity_metrics", {})
                agent_data[agent]["argument_complexities"].append(comp_met.get("complexity_score"))
                agent_data[agent]["rebuttal_rates"].append(comp_met.get("rebuttal_rate"))

        for tom_data in round_data.get("theory_of_mind", []):
            agent = tom_data.get("agent")
            if agent in agent_data:
                tom_met = tom_data.get("tom_metrics", {})
                agent_data[agent]["empathy_scores"].append(tom_met.get("empathy_score"))
                agent_data[agent]["perspective_taking_accuracies"].append(tom_met.get("perspective_taking_accuracy"))
                agent_data[agent]["common_ground_ids"].append(tom_met.get("common_ground_identification"))

    for update_data in psych_analysis.get("belief_updates", []):
        agent = update_data.get("agent")
        if agent in agent_data:
            update_an = update_data.get("update_analysis", {})
            agent_data[agent]["cognitive_dissonances"].append(update_an.get("cognitive_dissonance"))
            
            conf_before = update_an.get("confidence_before")
            conf_after = update_an.get("confidence_after")
            if conf_before is not None and conf_after is not None:
                agent_data[agent]["confidence_changes"].append(conf_after - conf_before)
            
    final_metrics = {}
    for agent, data_lists in agent_data.items():
        persona = agent_personas.get(agent, agent)
        # Filter out None values before calculating mean
        final_metrics[persona] = {
            key: np.mean([v for v in values if v is not None]) if any(v is not None for v in values) else 0.0
            for key, values in data_lists.items()
        }
        
    return final_metrics


def append_result_to_json(result_data: dict, output_filepath: str):
    """Safely appends a new result dictionary to a JSON file."""
    existing_results = []
    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = []
        except json.JSONDecodeError:
            existing_results = []
    existing_results.append(result_data)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)


def summarize_and_plot_results(output_file: str):
    """Loads the full results file, prints an aggregate summary table, and creates a plot."""
    print("\n--- Generating Final Summary and Plots ---")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    flat_data = []
    for entry in data:
        for persona, metrics in entry["psychometrics"].items():
            row = {"topic": entry["topic"], "persona": persona}
            # Rename keys for better plotting labels
            renamed_metrics = {f"avg_{k}": v for k, v in metrics.items()}
            row.update(renamed_metrics)
            flat_data.append(row)
            
    df = pd.DataFrame(flat_data)

    # 1. Print the summary table
    metric_columns = [
        'avg_confidence_scores', 'avg_cognitive_efforts', 'avg_argument_complexities',
        'avg_rebuttal_rates', 'avg_empathy_scores', 'avg_perspective_taking_accuracies',
        'avg_common_ground_ids', 'avg_cognitive_dissonances', 'avg_confidence_changes'
    ]
    
    # Ensure all expected columns exist, fill missing with 0
    for col in metric_columns:
        if col not in df.columns:
            df[col] = 0.0

    summary_table = df.groupby('persona')[metric_columns].mean().round(3)
    
    # Select a subset of the most interesting metrics for the summary table
    display_table = summary_table[['avg_confidence_scores', 'avg_cognitive_efforts', 'avg_empathy_scores', 'avg_cognitive_dissonances']]
    print("\nTable 3: Aggregate Psychometric Metrics by Agent Persona")
    print(display_table)
    
    # 2. Generate and save the plot
    plot_data = summary_table.reset_index()
    plot_data_melted = plot_data.melt(id_vars="persona", 
                                      value_vars=['avg_confidence_scores', 'avg_cognitive_efforts', 'avg_empathy_scores', 'avg_cognitive_dissonances'],
                                      var_name="metric", value_name="score")
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=plot_data_melted, x="metric", y="score", hue="persona", palette="viridis")
    plt.title(f"Comparison of Psychometric Profiles by Agent Persona (Moderator: Consensus)", fontsize=18)
    plt.ylabel("Average Score", fontsize=14)
    plt.xlabel("Psychometric Metric", fontsize=14)
    plt.xticks(rotation=10, ticks=range(4), labels=[
        "Argument Confidence", "Cognitive Effort", 
        "Empathy Score (ToM)", "Cognitive Dissonance"
    ])
    plt.legend(title="Agent Persona")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(PLOTS_DIRECTORY, f"figure4_psychometric_profiles_{str(rounds)}_{model_name}_neut.png")
    plt.savefig(plot_path)
    print(f"\nSaved Figure 4: Comparison of Psychometric Profiles to '{plot_path}'")


def analyze_experiment_folder(experiment_folder: str, output_file: str):
    """Processes all debate transcripts in a given folder."""
    if not os.path.isdir(experiment_folder):
        print(f"Error: Experiment directory not found at '{experiment_folder}'")
        return

    print(f"🔍 Starting psychometric analysis for experiment: {experiment_folder}")
    if os.path.exists(output_file):
        os.remove(output_file)

    transcript_files = [f for f in os.listdir(experiment_folder) if f.endswith('.json')]
    if not transcript_files:
        print("No JSON transcripts found.")
        return

    for i, filename in enumerate(transcript_files):
        transcript_path = os.path.join(experiment_folder, filename)
        print(f"Processing file {i+1}/{len(transcript_files)}: {filename}...")
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            psychometrics = extract_psychometric_data(transcript_data)
            
            result = {
                "source_transcript": filename,
                "topic": transcript_data.get("metadata", {}).get("topic", "Unknown"),
                "psychometrics": psychometrics
            }
            append_result_to_json(result, output_file)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}. Skipping.")

    print(f"\n✅ Analysis of all transcripts complete. Results saved in '{output_file}'.")
    
    summarize_and_plot_results(output_file)



def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Run psychometric analysis on a folder of debate transcripts.")
    parser.add_argument("--experiment-folder", required=False, type=str, default=BASE_TRANSCRIPT_DIR, help="Path to the folder containing the individual debate JSON files.")
    parser.add_argument("--output-file", type=str, default="cmv_3_llama32_psychometric_results.json", help="Name of the output JSON file.")
    args = parser.parse_args()

    args.output_file = os.path.join("../data/results", args.output_file)

    if not os.path.exists(PLOTS_DIRECTORY):
        os.makedirs(PLOTS_DIRECTORY)
        
    analyze_experiment_folder(args.experiment_folder, args.output_file)


if __name__ == "__main__":
    main()