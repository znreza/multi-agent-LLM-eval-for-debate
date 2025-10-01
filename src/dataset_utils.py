# datasets.py
import json
import random

import pandas as pd

# Example datasets paths (replace with your files)
DATA_PATHS = {
    "changemyview": "data/cmv_sample.json",     # JSON: [{"topic":..., "stance":..., "text":...}]
    "debatepedia": "data/debatepedia.csv",      # CSV: topic, pro, con
    "sbic": "data/sbic.jsonl",                  # JSONL: {"post":..., "label":...}
    "psych_study": "data/psych_prompts.csv"     # CSV: experiment, stimulus, condition
}

### Loaders ###

def load_changemyview(n=10):
    with open(DATA_PATHS["changemyview"], "r") as f:
        data = json.load(f)
    sampled = random.sample(data, min(n, len(data)))
    return sampled

def load_debatepedia(n=10):
    df = pd.read_csv(DATA_PATHS["debatepedia"])
    sampled = df.sample(n=min(n, len(df)))
    return sampled.to_dict("records")

def load_sbic(n=10):
    with open(DATA_PATHS["sbic"], "r") as f:
        lines = [json.loads(l) for l in f]
    sampled = random.sample(lines, min(n, len(lines)))
    return sampled

def load_psych_prompts(n=10):
    df = pd.read_csv(DATA_PATHS["psych_study"])
    sampled = df.sample(n=min(n, len(df)))
    return sampled.to_dict("records")

def get_dataset(name, n=10):
    if name == "changemyview":
        return load_changemyview(n)
    elif name == "debatepedia":
        return load_debatepedia(n)
    elif name == "sbic":
        return load_sbic(n)
    elif name == "psych":
        return load_psych_prompts(n)
    else:
        raise ValueError(f"Unknown dataset {name}")

### Preprocessor ###

def preprocess_for_debate(dataset_name, entries):
    """
    Convert dataset entries into debate-ready prompts.
    Returns: list of dicts {topic, pro, con, context}
    """
    debates = []

    if dataset_name == "changemyview":
        for e in entries:
            debates.append({
                "topic": e["topic"],
                "pro": e["text"] if e["stance"].lower() == "pro" else None,
                "con": e["text"] if e["stance"].lower() == "con" else None,
                "context": "Real user arguments from CMV"
            })

    elif dataset_name == "debatepedia":
        for e in entries:
            debates.append({
                "topic": e["topic"],
                "pro": e["pro"],
                "con": e["con"],
                "context": "Structured debate from Debatepedia"
            })

    elif dataset_name == "sbic":
        for e in entries:
            debates.append({
                "topic": f"Is the following statement biased or harmful? '{e['post']}'",
                "pro": "This statement perpetuates stereotypes and is harmful.",
                "con": "This statement is not harmful or is taken out of context.",
                "context": "Bias evaluation task"
            })

    elif dataset_name == "psych":
        for e in entries:
            debates.append({
                "topic": e["stimulus"],
                "pro": f"From {e['condition']} perspective, support the action.",
                "con": f"From {e['condition']} perspective, oppose the action.",
                "context": f"Stimulus from {e['experiment']}"
            })

    return debates

### Helper to assign topic to agents ###

def feed_topic_to_agents(agents, debate_case):
    """
    Give each agent the topic and context, let them set their initial beliefs.
    debate_case = {topic, pro, con, context}
    """
    topic = debate_case["topic"]
    beliefs = []
    for agent in agents:
        belief = agent.set_initial_belief(topic)
        beliefs.append(belief)
    return beliefs
