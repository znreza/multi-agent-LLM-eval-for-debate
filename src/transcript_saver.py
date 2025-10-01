"""
Transcript saving utilities for multi-agent debates.
Saves transcripts in JSON format matching existing experiment format.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


def create_filename(topic: str, moderator_style: str = "neutral") -> str:
    """Create filename based on topic and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_clean = "".join(c for c in topic if c.isalnum() or c.isspace()).strip()
    topic_clean = "_".join(topic_clean.split())
    topic_clean = topic_clean[:30]
    return f"{timestamp}_{topic_clean}_{moderator_style.upper()}.json"


def format_transcript_for_json(topic: str, rounds: List[Dict], moderator_style: str,
                              model_name: str = "llama-3.2-3b-instruct",
                              provider: str = "cpp",
                              moderator_initial_opinion: str = None,
                              final_judgment: str = None,
                              round_summaries: List[str] = None,
                              agent_belief_evolution: Dict[str, List[str]] = None,
                              psychometric_data: Dict[str, Any] = None,
                              raw_debate_string: str = None) -> Dict[str, Any]: # <-- NEW argument
    """
    Format debate rounds into the expected JSON transcript format.
    """
    num_rounds = len(set(r["round"] for r in rounds)) if rounds else 0
    agent_names = sorted(list(set(r["agent"] for r in rounds))) if rounds else []
    
    metadata = {
        "topic": topic, "topic_category": "Multi-Agent AI Debate",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "llm_model": model_name, "provider": provider,
        "num_rounds": num_rounds, "num_agents": len(agent_names),
        "agent_names": agent_names, "moderator_persona": f"{moderator_style.upper()}_WITH_VERDICT",
        "debate_format": "multi-agent-belief-updating"
    }
    
    transcript_entries = [{"speaker": "Moderator", "text": f"The debate on '{topic}' is now open.", "action": "debate_opening"}]
    
    rounds_by_number = {}
    for r in rounds:
        rounds_by_number.setdefault(r["round"], []).append(r)
    
    for round_num in sorted(rounds_by_number.keys()):
        transcript_entries.append({
            "speaker": "Moderator", "text": f"--- Round {round_num + 1} ---",
            "action": "round_header", "round": round_num
        })
        
        for arg in rounds_by_number[round_num]:
            entry = {
                "speaker": f"Agent {arg['agent']}", "text": arg["argument"],
                "persona": arg["context"]["persona"], "incentive": arg["context"]["incentive"],
                "round": round_num, "current_stance": arg.get("stance", "Not specified"),
                "action": "argument"
            }
            if "structured_analysis" in arg and arg["structured_analysis"]:
                entry["structured_analysis"] = arg["structured_analysis"]
            if "complexity_metrics" in arg and arg["complexity_metrics"]:
                entry["complexity_metrics"] = arg["complexity_metrics"]
            if "epistemic_markers" in arg and arg["epistemic_markers"]:
                entry["epistemic_markers"] = arg["epistemic_markers"]
            transcript_entries.append(entry)
        
        if round_summaries and round_num < len(round_summaries):
            transcript_entries.append({
                "speaker": "Moderator", "text": round_summaries[round_num],
                "action": "round_summary", "round": round_num
            })

    if final_judgment:
        transcript_entries.append({"speaker": "Moderator (Verdict)", "text": final_judgment, "action": "final_judgment"})
    else:
        transcript_entries.append({"speaker": "Moderator", "text": "The debate has concluded.", "action": "debate_closing"})

    transcript = {
        "metadata": metadata,
        "moderator_initial_opinion": moderator_initial_opinion or f"Moderating debate on: {topic}.",
        "transcript": transcript_entries
    }

    if agent_belief_evolution:
        transcript["agent_belief_evolution"] = agent_belief_evolution
    if psychometric_data:
        transcript["psychometric_analysis"] = psychometric_data
    
    # <-- NEW: Add the raw debate string at the end of the JSON object
    if raw_debate_string:
        transcript["raw_debate_transcript"] = raw_debate_string
        
    return transcript


def save_transcript(topic: str, rounds: List[Dict], moderator_style: str = "neutral",
                   model_name: str = "llama-3.2-3b-instruct", provider: str = "cpp",
                   output_dir: str = "../experiment_transcripts",
                   moderator_initial_opinion: str = None,
                   final_judgment: str = None,
                   round_summaries: List[str] = None,
                   agent_belief_evolution: Dict[str, List[str]] = None,
                   psychometric_data: Dict[str, Any] = None,
                   raw_debate_string: str = None, # <-- NEW argument
                   experiment_name: str = None) -> str:
    """Saves a debate transcript to a JSON file."""
    model_clean = model_name.replace("/", "_").replace("-", "_")
    full_output_dir = os.path.join(output_dir, model_clean)
    if experiment_name:
        exp_clean = experiment_name.replace(" ", "_").replace("/", "_")
        full_output_dir = os.path.join(full_output_dir, exp_clean)
    
    os.makedirs(full_output_dir, exist_ok=True)
    
    filename = create_filename(topic, moderator_style)
    filepath = os.path.join(full_output_dir, filename)
    
    transcript_data = format_transcript_for_json(
        topic=topic, rounds=rounds, moderator_style=moderator_style,
        model_name=model_name, provider=provider,
        moderator_initial_opinion=moderator_initial_opinion,
        final_judgment=final_judgment, round_summaries=round_summaries,
        agent_belief_evolution=agent_belief_evolution,
        psychometric_data=psychometric_data,
        raw_debate_string=raw_debate_string # <-- NEW: Pass down to formatter
    )
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        print(f"Transcript saved successfully: {filepath}")
    except TypeError as e:
        print(f"Error: Could not serialize transcript data to JSON. {e}")
        with open(f"{filepath}.error.txt", 'w', encoding='utf-8') as f:
            f.write(str(transcript_data))
        print("Saved a debug version of the transcript.")

    return filepath


def save_evaluation_results(results: List[Dict[str, Any]],
                          output_dir: str = "../data/results",
                          filename: str = "analysis_results_metrics_v2.json") -> str:
    """Saves evaluation results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results saved: {filepath}")
    return filepath


def append_evaluation_result(result: Dict[str, Any],
                           output_dir: str = "../data/results",
                           filename: str = "analysis_results_metrics_v2.json") -> str:
    """
    Appends a single evaluation result to an existing results file.
    If the file doesn't exist, it creates it.
    
    Args:
        result: A single evaluation result dictionary to append.
        output_dir: The directory where the results file is stored.
        filename: The name of the results JSON file.
        
    Returns:
        The full path to the updated file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    existing_results = []
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            # Ensure it's a list
            if not isinstance(existing_results, list):
                print(f"Warning: Existing results file '{filepath}' is not a list. Starting fresh.")
                existing_results = []
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not read or decode existing results file '{filepath}'. Starting fresh.")
            existing_results = []

    existing_results.append(result)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation result appended to: {filepath}")
    return filepath