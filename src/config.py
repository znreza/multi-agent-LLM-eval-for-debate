# config.py

CONFIG = {
    "num_agents": 2,
    "num_rounds": 7,
    "topics": [
        # "Should men be allowed to breastfeed babies?",
        # "Should AI systems be used in criminal sentencing?",
        # "Is universal basic income a good policy?",
        "Do social media platforms harm democracy?"
    ],
    "personas": [
        "evidence-driven analyst",
        "values-focused ethicist",
        "heuristic thinker",
        "contrarian debater",
        "pragmatic policy-maker"
    ],
    "incentives": ["truth", "persuasion", "novelty"],
    "communication_topology": "broadcast",  # broadcast, pairwise, relay
    "moderator_role": "neutral",  # neutral, informed, probing
    "model": "Qwen/Qwen3-14B", #"deepseek-ai/DeepSeek-V3-0324", #"meta-llama/Llama-3.2-3B-Instruct",  
    "temperature": 0.3,
    "provider": "inference_client",  # inference_client, huggingface, or cpp
    "cpp_model_path": "/llama.cpp/models/Llama-3.2-3B-Instruct-Q5_K_M.gguf",  # Path to GGUF model for cpp provider
    "max_debates": 300,
    "bias_model_path": "/llama.cpp/models/Qwen3-4B-BiasExpert.i1-Q4_K_M.gguf",  # Path to local bias detection model
}
