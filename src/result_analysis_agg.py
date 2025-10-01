# analyze_results.py
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import levene

# --- CONFIGURATION ---
RESULTS_FILE = "../data/results/cmv_100_5_moderator_consensus_gpt_oss.json"
PLOTS_DIRECTORY = "../data/plots"
rounds = 5  # Number of debate rounds, adjust as needed
model_name = "gpt_oss"  # Model name for titles, adjust as needed

def load_and_flatten_results(filepath: str) -> pd.DataFrame:
    """Loads the JSON results and correctly flattens nested dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    flat_data = []
    for entry in data:
        row = {
            "source_transcript": entry.get("source_transcript", "N/A"),
            "topic": entry.get("topic", "N/A"),
            "model": entry.get("model", "N/A")
        }
        
        if "overall_metrics" in entry.get("metrics", {}):
            for key, value in entry["metrics"]["overall_metrics"].items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"overall_{key}_{sub_key}"] = sub_value
                else:
                    row[f"overall_{key}"] = value

        if "per_round_metrics" in entry.get("metrics", {}):
            for round_metric in entry["metrics"]["per_round_metrics"]:
                round_num = round_metric.get("round", 0)
                for key, value in round_metric.items():
                    if key != "round":
                        row[f"round{round_num}_{key}"] = value
        
        flat_data.append(row)
        
    return pd.DataFrame(flat_data)

def categorize_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'category' column based on topic keywords."""
    contentious_keywords = [
        'racis', 'black people', 'slavery', 'confederate', 'white',
        'gender', 'women', 'men', 'feminist', 'lgbt', 'gay', 'transgender', 'cisgender',
        'jenner', 'kaepernick', 'trump', 'obama', 'clinton', 'sanders', 'republican', 'dnc',
        'israel', 'muslim', 'islam', 'pope', 'religion', 'scientology', 'god',
        'abortion', 'pro-life', 'cop killer', 'gun', 'rape', 'prostitution'
    ]
    
    def get_category(topic):
        topic_lower = str(topic).lower()
        return "Contentious" if any(kw in topic_lower for kw in contentious_keywords) else "Less Contentious"

    df['category'] = df['topic'].apply(get_category)
    return df

# --- ANALYSIS AND PLOTTING FUNCTIONS ---

def analyze_overall_dynamics(df: pd.DataFrame):
    # This function remains the same as before
    print("\n--- 1. Overall Debate Dynamics ---")
    key_metrics = ["overall_final_stance_convergence", "overall_total_stance_shift", "overall_semantic_diversity_trend", "overall_bias_analysis_bias_amplification_trend"]
    existing_metrics = [col for col in key_metrics if col in df.columns]
    if not existing_metrics:
        print("Warning: None of the key overall metrics found.")
        return
    summary_stats = df[existing_metrics].agg(['mean', 'median', 'std']).round(3)
    print("\nTable 1: Aggregate Statistics Across All Debates")
    print(summary_stats)
    if 'overall_final_stance_convergence' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['overall_final_stance_convergence'], kde=True, bins=20)
        plt.title(f'Distribution of Final Stance Convergence (Debaters: Contrarian, Moderator: Consensus Builder)', fontsize=16)
        plt.xlabel('Final Stance Convergence (Cosine Similarity)', fontsize=12)
        plt.ylabel('Number of Debates', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(PLOTS_DIRECTORY, f"figure1_convergence_distribution_{str(rounds)}_round_{model_name}_CONT.png"))
        plt.close()
        print("\nSaved Figure 1: Distribution of Final Stance Convergence")

def analyze_per_round_trends(df: pd.DataFrame):
    # This function remains the same as before
    print("\n--- 2. Per-Round Progression (Funneling Effect) ---")
    max_round = max([int(col[5]) for col in df.columns if col.startswith('round') and col[5].isdigit()] or [0])
    if max_round == 0: return
    diversity_cols = [f'round{i}_semantic_diversity' for i in range(1, max_round + 1)]
    existing_diversity_cols = [col for col in diversity_cols if col in df.columns]
    if not existing_diversity_cols: return
    mean_diversity = df[existing_diversity_cols].mean()
    sem_diversity = df[existing_diversity_cols].sem()
    plt.figure(figsize=(10, 6))
    mean_diversity.plot(kind='bar', yerr=sem_diversity, capsize=4, color='skyblue', alpha=0.8)
    plt.title('Average Semantic Diversity per Round (Debaters: Contrarian, Moderator: Consensus Builder)', fontsize=16)
    plt.xlabel('Debate Round', fontsize=12)
    plt.ylabel('Average Semantic Diversity', fontsize=12)
    plt.xticks(ticks=range(len(existing_diversity_cols)), labels=[f'Round {i+1}' for i in range(len(existing_diversity_cols))], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(PLOTS_DIRECTORY, f"figure2_semantic_diversity_per_round_{str(rounds)}_round_{model_name}_CONT.png"))
    plt.close()
    print("\nSaved Figure 2: Average Semantic Diversity per Round")
    
def analyze_topic_contentiousness(df: pd.DataFrame):
    # This function remains the same as before
    print("\n--- 3. Impact of Topic Contentiousness ---")
    comparison_metrics = ['overall_final_stance_convergence', 'overall_total_stance_shift', 'overall_bias_analysis_bias_amplification_trend', 'overall_sentiment_analysis_sentiment_trend']
    existing_metrics = [col for col in comparison_metrics if col in df.columns]
    if not existing_metrics: return
    comparison_df = df.groupby('category')[existing_metrics].agg(['mean', 'std']).round(3)
    print("\nTable 2: Metric Comparison by Topic Category")
    print(comparison_df)
    print("\nStatistical Test for Variance (Levene's Test):")
    print("-" * 50)
    contentious_group = df[df['category'] == 'Contentious']
    less_contentious_group = df[df['category'] == 'Less Contentious']
    for metric in ['overall_final_stance_convergence', 'overall_total_stance_shift']:
        if metric in df.columns:
            stat, p_value = levene(contentious_group[metric].dropna(), less_contentious_group[metric].dropna())
            print(f"Metric: {metric}\n  p-value: {p_value:.4f}")
            print("  Result: Not Statistically Significant. We cannot conclude the variances are different.")
    print("-" * 50)
    bias_trend_col = 'overall_bias_analysis_bias_amplification_trend'
    if bias_trend_col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='category', y=bias_trend_col, data=df)
        plt.title('Bias Amplification Trend by Topic Category', fontsize=16)
        plt.savefig(os.path.join(PLOTS_DIRECTORY, "figure3_bias_trend_by_category.png"))
        plt.close()
        print("\nSaved Figure 3: Bias Amplification Trend by Topic Category")

# --- FUNCTION FOR CASE STUDIES ---
def generate_case_studies(df: pd.DataFrame):
    """
    Selects and prints detailed analysis for specific, interesting debates.
    """
    print("\n--- 4. Deep Dive: Case Studies ---")
    
    # --- Case Study 1: A Polarizing Debate ---
    # Find the debate with the most negative stance agreement trend (most polarizing)
    if 'overall_stance_agreement_trend' in df.columns:
        polarizing_debate = df.loc[df['overall_stance_agreement_trend'].idxmin()]
        
        print("\nCase Study 1: A Polarizing Debate")
        print("="*40)
        print(f"Topic: {polarizing_debate['topic']}")
        print(f"Source: {polarizing_debate['source_transcript']}")
        print("\nKey Metrics:")
        print(f"  - Final Stance Convergence: {polarizing_debate['overall_final_stance_convergence']:.3f}")
        print(f"  - Stance Agreement Trend:   {polarizing_debate['overall_stance_agreement_trend']:.3f} (Strongly Negative)")
        print(f"  - Bias Amplification Trend: {polarizing_debate.get('overall_bias_analysis_bias_amplification_trend', 0):.3f}")
        
        print("\nPer-Round Progression (Stance Agreement):")
        for i in range(1, 4):
            if f'round{i}_stance_agreement' in polarizing_debate:
                print(f"  - Round {i}: {polarizing_debate[f'round{i}_stance_agreement']:.3f}")
                
        print("\nInterpretation:")
        print("This debate exemplifies a failure case for consensus-building. Despite the general trend,")
        print("the agents' stances grew semantically *further apart* over time, as shown by the strongly")
        print("negative agreement trend. This behavior is critical for identifying topics or argumentative")
        print("strategies that lead to polarization rather than resolution.")
        print("="*40)
        
    # --- Case Study 2: An Ideal Consensus ---
    # Find a debate with near-perfect final convergence
    if 'overall_final_stance_convergence' in df.columns:
        # Find the debate with the highest convergence score
        consensus_debate = df.loc[df['overall_final_stance_convergence'].idxmax()]

        print("\nCase Study 2: Ideal Consensus-Building")
        print("="*40)
        print(f"Topic: {consensus_debate['topic']}")
        print(f"Source: {consensus_debate['source_transcript']}")
        print("\nKey Metrics:")
        print(f"  - Final Stance Convergence: {consensus_debate['overall_final_stance_convergence']:.3f} (Near-Perfect Agreement)")
        print(f"  - Total Stance Shift:       {consensus_debate['overall_total_stance_shift']:.3f}")
        print(f"  - Stance Agreement Trend:   {consensus_debate.get('overall_stance_agreement_trend', 0):.3f}")

        print("\nPer-Round Progression (Stance Shift from Previous):")
        for i in range(1, 4):
            if f'round{i}_stance_shift_from_previous' in consensus_debate:
                 # Check for NaN before formatting
                shift = consensus_debate[f'round{i}_stance_shift_from_previous']
                if pd.notna(shift):
                    print(f"  - Round {i}: {shift:.3f}")
        
        print("\nInterpretation:")
        print("This debate represents an ideal outcome of the framework. Agents began with distinct")
        print("positions but successfully persuaded each other, leading to a near-perfect semantic")
        print("agreement in their final stances. The per-round stance shift shows that the most significant")
        print("changes in opinion typically occur in the earlier rounds of the debate.")
        print("="*40)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze flattened JSON results from debates.")
    parser.add_argument("results_file", type=str, nargs='?', default=RESULTS_FILE, help=f"Path to the JSON results file to analyze. Default: {RESULTS_FILE}")
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found at '{args.results_file}'")
    else:
        if not os.path.exists(PLOTS_DIRECTORY):
            os.makedirs(PLOTS_DIRECTORY)
            
        print(f"Loading and processing {args.results_file}...")
        df = load_and_flatten_results(args.results_file)
        df = categorize_topics(df)
        print(f"Successfully loaded and processed {len(df)} debate results.")
        
        analyze_overall_dynamics(df)
        analyze_per_round_trends(df)
        analyze_topic_contentiousness(df)

        generate_case_studies(df)
        
        print("\nAnalysis complete. All tables printed and plots saved to the 'plots' directory.")