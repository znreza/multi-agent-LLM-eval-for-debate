# analyze_psychometrics.py
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- CONFIGURATION ---
RESULTS_FILE = "cmv_800_3_moderator_neutral.json"
PLOTS_DIRECTORY = "plots"

# --- HELPER FUNCTIONS ---

def load_and_process_psychometrics(filepath: str) -> pd.DataFrame:
    """
    Loads the JSON results and extracts detailed psychometric data into a flat DataFrame.
    Each row will represent one agent's action in one round (argument or belief update).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_rows = []
    
    for debate in data:
        # Extract overall metrics for correlation later
        final_convergence = debate["metrics"]["overall_metrics"].get("final_stance_convergence", np.nan)
        
        # Process arguments from the transcript structure
        for turn in debate["transcript"]:
            if turn.get("action") == "argument":
                agent_name = turn["speaker"].replace("Agent ", "")
                row = {
                    "topic": debate["topic"],
                    "agent": agent_name,
                    "persona": turn.get("persona"),
                    "round": turn.get("round"),
                    "final_convergence": final_convergence,
                    "confidence_score": turn.get("structured_analysis", {}).get("confidence_score"),
                    "cognitive_effort": turn.get("structured_analysis", {}).get("cognitive_effort"),
                    "evidence_ratio": turn.get("complexity_metrics", {}).get("evidence_ratio"),
                    "rebuttal_rate": turn.get("complexity_metrics", {}).get("rebuttal_rate")
                }
                processed_rows.append(row)

        # Process ToM from the psychometric_analysis structure
        for round_data in debate["psychometric_analysis"]["rounds"]:
            for tom_entry in round_data.get("theory_of_mind", []):
                agent_name = tom_entry["agent"]
                # Create a metric for how much common ground was identified
                acknowledged_count = len(tom_entry.get("acknowledged_points", []))
                common_ground_count = len(tom_entry.get("common_ground", []))
                
                row = {
                    "topic": debate["topic"],
                    "agent": agent_name,
                    "round": tom_entry.get("round"),
                    "final_convergence": final_convergence,
                    "empathy_score": tom_entry.get("tom_metrics", {}).get("empathy_score"),
                    "common_ground_count": acknowledged_count + common_ground_count
                }
                processed_rows.append(row)
        
        # Process belief updates
        for update in debate["psychometric_analysis"]["belief_updates"]:
            agent_name = update["agent"]
            analysis = update.get("update_analysis", {})
            confidence_before = analysis.get("confidence_before")
            confidence_after = analysis.get("confidence_after")
            
            row = {
                "topic": debate["topic"],
                "agent": agent_name,
                "final_convergence": final_convergence,
                "belief_change_type": analysis.get("belief_change"),
                "confidence_before": confidence_before,
                "confidence_after": confidence_after,
                "delta_confidence": (confidence_after - confidence_before) if confidence_before is not None and confidence_after is not None else None
            }
            processed_rows.append(row)

    df = pd.DataFrame(processed_rows)
    # Merge agent personas back into all rows for easier grouping
    personas = df.dropna(subset=['persona'])[['agent', 'persona']].drop_duplicates()
    df = pd.merge(df.drop(columns=['persona']), personas, on='agent', how='left')
    
    return df

# --- ANALYSIS AND PLOTTING FUNCTIONS ---

def analyze_personas_and_cognition(df: pd.DataFrame):
    """Analyzes argumentation style and cognitive states by persona."""
    print("\n--- 1. Persona-Driven Behaviors ---")
    
    # Filter for rows that contain argument data
    arg_df = df.dropna(subset=['evidence_ratio', 'confidence_score', 'cognitive_effort'])
    
    # Group by persona
    persona_summary = arg_df.groupby('persona')[['evidence_ratio', 'confidence_score', 'cognitive_effort']].mean().round(3)
    
    print("\nTable 3: Average Cognitive Metrics by Persona")
    print(persona_summary)
    
    # Figure 4: Argumentation Styles
    plt.figure(figsize=(12, 7))
    arg_df.groupby('persona')[['evidence_ratio', 'rebuttal_rate']].mean().plot(kind='bar', rot=0)
    plt.title('Argumentation Styles by Agent Persona', fontsize=16)
    plt.ylabel('Average Score', fontsize=12)
    plt.xlabel('Persona', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIRECTORY, "figure4_persona_argument_styles.png"))
    plt.close()
    print("\nSaved Figure 4: Persona-Driven Argumentation Styles")
    
    # Figure 5: Confidence and Effort Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(ax=axes[0], x='persona', y='confidence_score', data=arg_df)
    axes[0].set_title('Distribution of Confidence Scores', fontsize=14)
    sns.boxplot(ax=axes[1], x='persona', y='cognitive_effort', data=arg_df)
    axes[1].set_title('Distribution of Cognitive Effort', fontsize=14)
    fig.suptitle('Cognitive State Distributions by Persona', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOTS_DIRECTORY, "figure5_persona_cognitive_states.png"))
    plt.close()
    print("Saved Figure 5: Confidence and Effort by Persona")
    
def analyze_theory_of_mind(df: pd.DataFrame):
    """Analyzes the link between Theory of Mind and debate outcomes."""
    print("\n--- 2. Theory of Mind and Constructive Dialogue ---")
    
    # Aggregate ToM metrics per debate
    tom_df = df.dropna(subset=['empathy_score', 'common_ground_count'])
    debate_tom_summary = tom_df.groupby('topic').agg(
        avg_empathy=('empathy_score', 'mean'),
        total_common_ground=('common_ground_count', 'sum'),
        final_convergence=('final_convergence', 'first')
    ).reset_index()

    # Figure 6: Correlation Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(x='total_common_ground', y='final_convergence', data=debate_tom_summary,
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.title('Higher Common Ground Identification Correlates with Convergence', fontsize=16)
    plt.xlabel('Total Common Ground Points Identified in Debate', fontsize=12)
    plt.ylabel('Final Stance Convergence', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIRECTORY, "figure6_tom_correlation.png"))
    plt.close()
    print("\nSaved Figure 6: Theory of Mind's Impact on Convergence")
    
def analyze_confidence_dynamics(df: pd.DataFrame):
    """Analyzes how agent confidence changes during belief updates."""
    print("\n--- 3. Confidence Dynamics During Belief Updates ---")
    
    update_df = df.dropna(subset=['delta_confidence', 'belief_change_type'])
    update_df = update_df[update_df['belief_change_type'] != 'none']
    
    avg_delta = update_df['delta_confidence'].mean()
    print(f"\nAverage change in confidence after an update: {avg_delta:.3f}")
    
    # Figure 7: Confidence Delta by Change Type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='belief_change_type', y='delta_confidence', data=update_df,
                order=['minor', 'moderate', 'major'])
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Change in Confidence After Belief Updates', fontsize=16)
    plt.xlabel('Magnitude of Belief Change', fontsize=12)
    plt.ylabel('Δ Confidence (After - Before)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(PLOTS_DIRECTORY, "figure7_confidence_dynamics.png"))
    plt.close()
    print("\nSaved Figure 7: Confidence Dynamics During Belief Updates")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(PLOTS_DIRECTORY):
        os.makedirs(PLOTS_DIRECTORY)
        
    print(f"Loading and processing {RESULTS_FILE} for psychometric analysis...")
    psych_df = load_and_process_psychometrics(RESULTS_FILE)
    print(f"Successfully loaded data for {psych_df['topic'].nunique()} debates.")
    
    analyze_personas_and_cognition(psych_df)
    analyze_theory_of_mind(psych_df)
    analyze_confidence_dynamics(psych_df)
    
    print("\nPsychometric analysis complete. All tables printed and plots saved to the 'plots' directory.")