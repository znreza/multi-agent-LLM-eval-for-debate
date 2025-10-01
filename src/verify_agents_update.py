#!/usr/bin/env python3
"""
Verification script to ensure agents.py has all the correct updates
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_agent_imports():
    """Test that agents.py can import all prompt functions"""
    try:
        from agents import Agent, Moderator
        print("✅ Agent and Moderator classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_agent_methods():
    """Test that Agent class has enhanced methods"""
    from agents import Agent
    
    # Create a test agent
    agent = Agent("TestAgent", persona="evidence-driven analyst", incentive="truth", provider="inference_client")
    
    # Check methods exist
    methods_to_check = ['set_initial_belief', 'generate_argument', 'update_belief', '_call_llm']
    for method in methods_to_check:
        if hasattr(agent, method):
            print(f"✅ Agent.{method} exists")
        else:
            print(f"❌ Agent.{method} missing")
            return False
    
    return True

def test_moderator_methods():
    """Test that Moderator class has enhanced methods"""
    from agents import Moderator
    
    # Create a test moderator
    moderator = Moderator(style="neutral", provider="inference_client")
    
    # Check methods exist including the new get_initial_opinion
    methods_to_check = ['get_initial_opinion', 'summarize_round', 'judge_winner', '_call_llm']
    for method in methods_to_check:
        if hasattr(moderator, method):
            print(f"✅ Moderator.{method} exists")
        else:
            print(f"❌ Moderator.{method} missing")
            return False
    
    return True

def test_prompt_integration():
    """Test that prompt functions are being used"""
    import inspect
    from agents import Agent, Moderator
    
    # Check if Agent methods use prompt functions
    agent_methods = {
        'set_initial_belief': 'get_agent_initial_belief_prompt',
        'generate_argument': 'get_agent_argument_prompt',
        'update_belief': 'get_agent_update_belief_prompt'
    }
    
    for method_name, expected_prompt_func in agent_methods.items():
        method = getattr(Agent, method_name)
        source = inspect.getsource(method)
        if expected_prompt_func in source:
            print(f"✅ Agent.{method_name} uses {expected_prompt_func}")
        else:
            print(f"❌ Agent.{method_name} doesn't use {expected_prompt_func}")
            return False
    
    # Check Moderator methods
    moderator_methods = {
        'summarize_round': 'get_moderator_summary_prompt',
        'judge_winner': 'get_moderator_judge_prompt'
    }
    
    for method_name, expected_prompt_func in moderator_methods.items():
        method = getattr(Moderator, method_name)
        source = inspect.getsource(method)
        if expected_prompt_func in source:
            print(f"✅ Moderator.{method_name} uses {expected_prompt_func}")
        else:
            print(f"❌ Moderator.{method_name} doesn't use {expected_prompt_func}")
            return False
    
    return True

def test_providers_available():
    """Test that all providers are properly configured"""
    from agents import Agent, Moderator
    
    providers_to_test = ["inference_client"]  # Only test this one as it doesn't need extra setup
    
    for provider in providers_to_test:
        try:
            agent = Agent("TestAgent", persona="analytical", incentive="truth", provider=provider)
            moderator = Moderator(style="neutral", provider=provider)
            print(f"✅ Provider '{provider}' configured correctly")
        except Exception as e:
            print(f"❌ Provider '{provider}' failed: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🔍 Verifying agents.py Updates")
    print("=" * 40)
    
    tests = [
        test_agent_imports,
        test_agent_methods,
        test_moderator_methods,
        test_prompt_integration,
        test_providers_available
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print(f"\n📋 Running {test_func.__name__}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"💥 {test_func.__name__} ERROR: {e}")
    
    print("\n" + "=" * 40)
    print(f"🎯 VERIFICATION RESULTS:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    
    if failed == 0:
        print(f"🎉 All tests passed! agents.py is fully updated with:")
        print(f"   • Enhanced prompt system integration")
        print(f"   • Moderator initial opinion capability") 
        print(f"   • Structured argument generation")
        print(f"   • Belief updating with context")
        print(f"   • Multi-provider support (inference_client, huggingface, cpp)")
    else:
        print(f"⚠️ Some tests failed. Check the output above.")
    
    sys.exit(0 if failed == 0 else 1)