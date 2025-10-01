# main.py
import argparse
import json
import os

from agents import Agent, Moderator
from config import CONFIG
from transcript_saver import save_transcript


def run_debate(topic, agents, moderator, rounds=3, save_transcript_bool=True, experiment_name=None):
    """
    Run one debate case with given agents and moderator with enhanced psychometric tracking.
    Returns: tuple of (round arguments, transcript_filepath, metadata)
    """
    
    # print(f"\n🎯 Starting debate on: {topic}")
    
    # Initialize containers for all debate artifacts
    all_args_history = []
    round_summaries = []
    agent_belief_evolution = {agent.name: [] for agent in agents}
    psychometric_data = {"rounds": [], "belief_updates": []}
    raw_transcript_lines = [] 

    # Get and record moderator's unbiased stance (without system prompt influence)
    # print("📝 Getting moderator's unbiased stance...")
    moderator_initial_opinion = moderator.get_unbiased_stance(topic)
    # print(f"Moderator unbiased stance: {moderator_initial_opinion}")
    raw_transcript_lines.append(f"MODERATOR (Unbiased Stance): {moderator_initial_opinion}\n")

    # Assign initial beliefs and track them
    if len(agents) >= 2:
        initial_belief_a = agents[0].set_initial_belief(topic, stance_hint="pro")
        initial_belief_b = agents[1].set_initial_belief(topic, stance_hint="con")
        agent_belief_evolution[agents[0].name].append(f"Initial (Pro): {initial_belief_a}")
        agent_belief_evolution[agents[1].name].append(f"Initial (Con): {initial_belief_b}")
    for ag in agents[2:]:
        initial_belief = ag.set_initial_belief(topic, stance_hint="neutral")
        agent_belief_evolution[ag.name].append(f"Initial (Neutral): {initial_belief}")

    # Debate rounds
    for r in range(rounds):
        # print(f"\n--- Round {r+1} ---")
        raw_transcript_lines.append(f"\n--- Round {r+1} ---\n") # <-- NEW
        
        round_psychometric = {"round": r, "arguments": [], "theory_of_mind": []}
        current_round_args = []

        # Each agent produces a structured argument
        for ag in agents:
            opponent_args = [a["argument"] for a in all_args_history if a["agent"] != ag.name]
            arg_data = ag.generate_structured_argument(topic, r, opponent_args)
            
            argument_text = arg_data.get('argument', 'Error: Argument parsing failed.')
            # print(f"🗣️ {ag.name} ({ag.persona}): {argument_text[:200]}...")
            raw_transcript_lines.append(f"{ag.name} ({ag.persona}): {argument_text}\n") # <-- NEW

            current_round_args.append(arg_data)
            all_args_history.append(arg_data)
            
            round_psychometric["arguments"].append({
                "agent": ag.name,
                "structured_analysis": arg_data.get("structured_analysis", {}),
                "complexity_metrics": arg_data.get("complexity_metrics", {}),
                "epistemic_markers": arg_data.get("epistemic_markers", [])
            })

            # Theory of mind analysis
            if opponent_args:
                tom_data = ag.analyze_theory_of_mind(topic, r, opponent_args)
                round_psychometric["theory_of_mind"].append(tom_data)
                # print(f"🧠 {ag.name} ToM analysis: {tom_data.get('opponent_understanding', 'ToM parsing failed.')[:100]}...")

        # Agents update beliefs (internal process, not added to raw transcript)
        for ag in agents:
            belief_update_data = ag.update_belief_structured(topic, current_round_args)
            new_belief = belief_update_data.get("updated_belief", ag.beliefs.get(topic))
            if new_belief not in agent_belief_evolution[ag.name][-1]:
                agent_belief_evolution[ag.name].append(f"Round {r+1}: {new_belief}")
                # print(f"🧠 {ag.name} updated belief: {new_belief[:150]}...")
            psychometric_data["belief_updates"].append(belief_update_data)
        
        psychometric_data["rounds"].append(round_psychometric)

        # Moderator summarizes the current round
        summary = moderator.summarize_round(topic, current_round_args)
        round_summaries.append(summary)
        # print(f"⚖️ Moderator summary: {summary}")
        raw_transcript_lines.append(f"MODERATOR (Summary): {summary}\n") # <-- NEW

    # Judge final outcome
    final_judgment = moderator.judge_winner(topic, all_args_history)
    # print(f"🏆 Final judgment: {final_judgment}")
    raw_transcript_lines.append(f"\n--- Final Judgment ---\n") # <-- NEW
    raw_transcript_lines.append(f"MODERATOR (Verdict): {final_judgment}\n") # <-- NEW

    # Join all raw lines into a single string
    raw_debate_string = "".join(raw_transcript_lines) # <-- NEW

    # Save transcript
    transcript_filepath = None
    if save_transcript_bool:
        transcript_filepath = save_transcript(
            topic=topic,
            rounds=all_args_history,
            moderator_style=moderator.style,
            model_name=CONFIG.get("model", "unknown"),
            provider=CONFIG.get("provider", "unknown"),
            moderator_initial_opinion=moderator_initial_opinion,
            final_judgment=final_judgment,
            round_summaries=round_summaries,
            agent_belief_evolution=agent_belief_evolution,
            psychometric_data=psychometric_data,
            raw_debate_string=raw_debate_string, # <-- NEW: Pass the raw string
            experiment_name=experiment_name
        )

    return all_args_history, transcript_filepath, {
        "moderator_initial_opinion": moderator_initial_opinion,
        "final_judgment": final_judgment,
        "round_summaries": round_summaries,
        "agent_belief_evolution": agent_belief_evolution
    }


def main(experiment_name=None):

    if experiment_name is None:
        experiment_name = f"cmv_300_{str(CONFIG['num_rounds'])}_moderator_{CONFIG['moderator_role']}"

    parser = argparse.ArgumentParser(description="Run multi-agent debates")
    parser.add_argument("--experiment", "-e", type=str, default=experiment_name, help="Experiment name for organizing results")
    parser.add_argument("--max-debates", "-m", type=int, default=CONFIG["max_debates"], help="Maximum number of debates to run")
    parser.add_argument("--rounds", "-r", type=int, default=CONFIG["num_rounds"], help="Number of rounds per debate")
    parser.add_argument("--test-run", "-t", action="store_true", help="Use small topics from CONFIG file for testing")
    
    args = parser.parse_args()
    experiment_name = args.experiment
    max_debates = args.max_debates
    num_rounds = args.rounds
    test_run = args.test_run

    print("🚀 Starting Multi-Agent Debate Experiment")
    print(f"📊 Experiment: {experiment_name or 'Default'}")
    print(f"🤖 Model: {CONFIG['model']} ({CONFIG['provider']})")
    print(f"🎲 Max debates: {max_debates}, Rounds per debate: {num_rounds}")
    print(f"🧪 Test run: {test_run}")
    
    # Load topics based on test_run flag
    if test_run:
        debates = CONFIG["topics"][:max_debates]
        print(f"📝 Using {len(debates)} topics from CONFIG file")
    else:
        # Load topics from cmv_topics.json file
        cmv_topics_path = "../data/cmv_topics.json"
        try:
            with open(cmv_topics_path, 'r', encoding='utf-8') as f:
                all_cmv_topics = json.load(f)
            # Take first 800 topics or max_debates, whichever is smaller
            debates = all_cmv_topics[188:min(300, max_debates)]
            print(f"📝 Loaded {len(debates)} topics from {cmv_topics_path}")
        except FileNotFoundError:
            print(f"⚠️ Could not find {cmv_topics_path}, falling back to CONFIG topics")
            debates = CONFIG["topics"][:max_debates]
        except Exception as e:
            print(f"⚠️ Error loading topics from {cmv_topics_path}: {e}")
            print("⚠️ Falling back to CONFIG topics")
            debates = CONFIG["topics"][:max_debates]

    num_agents = min(CONFIG["num_agents"], len(CONFIG["personas"]))
    agents = [
        Agent(f"A{i+1}", persona=CONFIG["personas"][i], incentive=CONFIG["incentives"][i])
        for i in range(num_agents)
    ]
    moderator = Moderator(style=CONFIG["moderator_role"])

    print(f"👥 Agents: {[f'{a.name} ({a.persona})' for a in agents]}")
    print(f"⚖️ Moderator: {moderator.style}")

    transcript_files = []
    all_metadata = []
    
    for i, topic in enumerate(debates):
        print("\n" + "="*60)
        print(f"🔥 DEBATE {i+1}/{len(debates)}: {topic}")
        print("="*60)
        
        _, transcript_file, metadata = run_debate(
            topic, agents, moderator, 
            rounds=num_rounds, 
            save_transcript_bool=True,
            experiment_name=experiment_name
        )
        
        if transcript_file:
            transcript_files.append(transcript_file)
        all_metadata.append(metadata)

    print("\n" + "="*60)
    print("🎉 EXPERIMENT COMPLETED!")
    print("="*60)
    
    print(f"📈 Summary:")
    print(f"   • Total debates conducted: {len(debates)}")
    print(f"   • Transcripts saved: {len(transcript_files)}")
    print(f"   • Experiment: {experiment_name or 'Default'}")
    print(f"   • Model: {CONFIG['model']} ({CONFIG['provider']})")
    
    model_clean = CONFIG['model'].replace("/", "_").replace("-", "_")
    final_output_dir = f"../experiment_transcripts/{model_clean}/"
    if experiment_name:
        final_output_dir += f"{experiment_name.replace(' ', '_')}/"
    
    print(f"\n📁 Files saved in: {final_output_dir}")


if __name__ == "__main__":
    main()