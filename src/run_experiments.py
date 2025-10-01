#!/usr/bin/env python3
"""
Example script showing how to run organized experiments with different models and configurations.
"""

import subprocess
import sys
import os

def run_experiment(experiment_name, max_debates=2, rounds=2, description=""):
    """Run a single experiment configuration"""
    print(f"\n🚀 Starting experiment: {experiment_name}")
    if description:
        print(f"📝 Description: {description}")
    
    cmd = [
        sys.executable, "main.py",
        "--experiment", experiment_name,
        "--max-debates", str(max_debates),
        "--rounds", str(rounds)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Run a series of experiments to demonstrate the enhanced transcript system"""
    
    print("🧪 Multi-Agent Debate Experiment Suite")
    print("=" * 50)
    print("This script demonstrates organized experiment management with:")
    print("• Model-specific folder organization")
    print("• Experiment subfolder structuring") 
    print("• Enhanced transcript capture")
    print("• Command-line argument support")
    print()
    
    # Define experiments to run
    experiments = [
        {
            "name": "baseline_prompts",
            "max_debates": 2,
            "rounds": 2,
            "description": "Baseline experiment with default prompt templates"
        },
        {
            "name": "enhanced_personas",
            "max_debates": 2, 
            "rounds": 3,
            "description": "Testing persona-specific argument generation"
        },
        {
            "name": "belief_tracking",
            "max_debates": 1,
            "rounds": 3,
            "description": "Focus on agent belief evolution over rounds"
        }
    ]
    
    print(f"📋 Planned experiments: {len(experiments)}")
    for exp in experiments:
        print(f"   • {exp['name']}: {exp['description']}")
    
    # Ask for confirmation
    response = input(f"\n❓ Run all {len(experiments)} experiments? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("⏹️ Experiments cancelled")
        return
    
    # Run experiments
    successful = 0
    failed = 0
    
    for exp in experiments:
        success = run_experiment(
            experiment_name=exp["name"],
            max_debates=exp["max_debates"],
            rounds=exp["rounds"],
            description=exp["description"]
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 EXPERIMENT SUITE COMPLETED")
    print("=" * 50)
    print(f"✅ Successful experiments: {successful}")
    print(f"❌ Failed experiments: {failed}")
    
    if successful > 0:
        print(f"\n📁 Check results in:")
        print(f"   experiment_transcripts/[model_name]/[experiment_name]/")
        print(f"\n💡 Example commands to run individual experiments:")
        print(f"   python main.py --experiment my_test --max-debates 3 --rounds 4")
        print(f"   python main.py -e quick_test -m 1 -r 2")
    
    return successful, failed

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)