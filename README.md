# Multi-Agent LLM Evaluation Framework

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS%202025-Workshop-blue)](https://sites.google.com/view/llm-eval-workshop/home?authuser=0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation for the paper [**"The Social Laboratory: A Psychometric Framework for Multi-Agent LLM Evaluation"**](https://arxiv.org/abs/2510.01295), accepted to the **NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling**.

📄 **Paper**: [Link to paper]
🔗 **Workshop**: [NeurIPS 2025 LLM Evaluation Workshop](https://sites.google.com/view/llm-eval-workshop/home?authuser=0)

---

## 🎯 Overview

This framework introduces a psychometric approach to evaluating Large Language Models (LLMs) through multi-agent debates. Rather than traditional benchmarking, we assess LLMs based on social and cognitive dimensions such as:

- **Belief formation and updating** through structured argumentation
- **Theory of Mind** capabilities in multi-agent settings
- **Semantic diversity** and stance convergence over debate rounds
- **Bias detection** and cognitive complexity in arguments
- **Epistemic reasoning** and confidence calibration

The system simulates structured debates between AI agents with different personas and incentives, moderated by a neutral or proactive moderator, to reveal emergent social behaviors in LLMs.

---

## 🏗️ Architecture

### Core Components

```
multi-agent-LLM-eval/
├── src/
│   ├── main.py                      # Main experiment orchestrator
│   ├── agents.py                    # Agent and Moderator classes
│   ├── config.py                    # Experiment configuration
│   ├── prompts.py                   # Structured prompt templates
│   ├── psychometric_parser.py       # JSON parsing for structured responses
│   ├── metrics.py                   # Advanced psychometric metrics
│   ├── eval.py                      # Batch evaluation pipeline
│   ├── transcript_saver.py          # Debate transcript management
│   ├── result_analysis_agg.py       # Statistical analysis and visualization
│   ├── result_analysis_psychometric.py  # Psychometric-specific analysis
│   └── dataset_utils.py             # Dataset loading utilities
├── data/
│   ├── cmv_topics.json              # ChangeMyView debate topics
│   ├── results/                     # Evaluation results
│   └── plots/                       # Generated visualizations
├── experiment_transcripts/          # Saved debate transcripts
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Getting Started

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/multi-agent-LLM-eval.git
cd multi-agent-LLM-eval
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure your LLM provider** in `src/config.py`:
   - **Hugging Face Inference Client** (default)
   - **Local Hugging Face models**
   - **llama.cpp** for GGUF quantized models

### Configuration

Edit `src/config.py` to customize your experiments:

```python
CONFIG = {
    "num_agents": 2,                    # Number of debating agents
    "num_rounds": 7,                    # Rounds per debate
    "personas": [                       # Agent personas
        "evidence-driven analyst",
        "values-focused ethicist",
        "heuristic thinker",
        "contrarian debater",
        "pragmatic policy-maker"
    ],
    "incentives": ["truth", "persuasion", "novelty"],
    "moderator_role": "neutral",        # neutral, informed, probing
    "model": "Qwen/Qwen3-14B",         # Model name or path
    "provider": "inference_client",     # inference_client, huggingface, cpp
    "temperature": 0.3,
    "max_debates": 300
}
```

---

## 📊 Running Experiments

### 1. Single Debate Experiment

Run a quick test with topics from the config file:

```bash
cd src
python main.py --test-run --rounds 3
```

### 2. Full-Scale Evaluation

Run experiments using ChangeMyView topics:

```bash
python main.py --experiment "cmv_300_7_moderator_neutral" \
               --max-debates 300 \
               --rounds 7
```

### 3. Batch Evaluation with Metrics

After generating transcripts, calculate psychometric metrics:

```bash
python eval.py --experiment-folder "Qwen_Qwen3_14B/cmv_300_3_moderator_neutral" \
               --output-file "qwen_evaluation_results.json"
```

### 4. Statistical Analysis

Analyze results and generate visualizations:

```bash
python result_analysis_agg.py
```

---

## 📈 Metrics & Analysis

### Psychometric Metrics

The framework computes advanced metrics for each debate:

#### **Overall Metrics**
- **Semantic Diversity**: Measures argument variety using sentence embeddings
- **Stance Convergence**: Cosine similarity of final agent positions
- **Total Stance Shift**: Movement from initial to final beliefs
- **Bias Score**: Social bias detection using fine-tuned local models
- **Belief Update Frequency**: Rate of opinion changes per agent

#### **Per-Round Metrics**
- **Semantic Diversity Trend**: Evolution of argument diversity
- **Stance Agreement**: Agreement level between agents per round
- **Sentiment Score**: Emotional tone of arguments
- **Complexity Metrics**: Evidence ratio, rebuttal rate, claim count

#### **Cognitive Metrics**
- **Theory of Mind Scores**: Empathy, perspective-taking accuracy
- **Epistemic Markers**: Certainty language (definitely, possibly, etc.)
- **Cognitive Effort**: Argument structure complexity
- **Confidence Calibration**: Stated vs. actual belief confidence

---

## 🧪 Experimental Design

### Agent Personas

Agents are assigned distinct cognitive styles:
- **Evidence-driven analyst**: Relies on data and research
- **Values-focused ethicist**: Prioritizes moral considerations
- **Heuristic thinker**: Uses rules of thumb and intuition
- **Contrarian debater**: Challenges conventional positions
- **Pragmatic policy-maker**: Focuses on practical outcomes

### Agent Incentives

Each agent pursues one of three incentives:
- **Truth**: Maximize accuracy and evidence
- **Persuasion**: Win the debate
- **Novelty**: Introduce unique perspectives

### Moderator Styles

- **Neutral**: Impartial observation and judgment
- **Consensus Builder**: Encourages common ground
- **Probing**: Challenges arguments with questions

### Debate Topics

The system uses **ChangeMyView (CMV)** topics covering:
- Social policy and justice
- Ethics and morality
- Science and technology
- Politics and governance
- Cultural and identity issues

---

## 📂 Output Format

### Debate Transcripts

Saved as JSON in `experiment_transcripts/`:

```json
{
  "metadata": {
    "topic": "Do social media platforms harm democracy?",
    "llm_model": "Qwen/Qwen3-14B",
    "num_rounds": 7,
    "timestamp": "20250831_222001"
  },
  "moderator_initial_opinion": "Generally neutral...",
  "transcript": [
    {
      "speaker": "A1",
      "text": "...",
      "action": "argument",
      "round": 0,
      "structured_analysis": {...}
    }
  ],
  "agent_belief_evolution": {
    "A1": ["Initial: ...", "Round 1: ...", ...]
  },
  "psychometric_data": {...},
  "final_judgment": "..."
}
```

### Evaluation Results

Aggregated metrics saved in `data/results/`:

```json
{
  "source_transcript": "20250831_222001_Dosocialmediaplatforms...",
  "topic": "Do social media platforms harm democracy?",
  "model": "Qwen/Qwen3-14B",
  "metrics": {
    "overall_metrics": {
      "semantic_diversity": 0.45,
      "final_stance_convergence": 0.72,
      "total_stance_shift": 0.28,
      "bias_score": 0.15
    },
    "per_round_metrics": [...]
  }
}
```

---

## 🔬 Research Applications

This framework enables research into:

1. **Emergent Social Behavior**: How do LLMs behave in multi-agent settings?
2. **Belief Dynamics**: When and why do agents change their positions?
3. **Persuasion Mechanisms**: What argumentative strategies are effective?
4. **Bias Amplification**: Do debates magnify or mitigate biases?
5. **Theory of Mind**: Can LLMs model other agents' mental states?
6. **Model Comparison**: Systematic evaluation across different LLMs

---

## 🛠️ Advanced Features

### Custom Bias Detection

Uses local GGUF models for social bias detection:

```python
CONFIG["bias_model_path"] = "/path/to/Qwen3-4B-BiasExpert.gguf"
```

### Multi-Provider Support

- **Hugging Face Inference API**: Cloud-based execution
- **Local Transformers**: GPU/CPU inference with quantization
- **llama.cpp**: Efficient CPU inference with GGUF models

### Structured Argumentation

Agents generate JSON-formatted arguments with:
- **Claims**: Main propositions with confidence scores
- **Evidence**: Supporting data and sources
- **Warrants**: Logical connections
- **Rebuttals**: Counter-arguments to opponents

### Theory of Mind Analysis

Explicit modeling of opponents' perspectives:
- Understanding of opponent's position
- Acknowledged common ground
- Predicted responses
- Empathy scores

---

## 📊 Visualization

The framework generates publication-ready plots:

- **Stance Convergence Distributions**: Histograms of final agreement
- **Per-Round Trends**: Line plots of metric evolution
- **Category Comparisons**: Contentious vs. less contentious topics
- **Model Comparisons**: Side-by-side performance across LLMs

Example:
```bash
# Generates plots in data/plots/
python result_analysis_agg.py
```

---

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{social-lab-mad,
  title={The Social Laboratory: A Psychometric Framework for Multi-Agent LLM Evaluation},
  author={Zarreen Reza},
  booktitle={NeurIPS 2025 Workshop on  Evaluating the Evolving
LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling},
  year={2025},
      eprint={2510.01295},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.01295}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ChangeMyView (CMV)**: For providing rich debate topics
- **Hugging Face**: For model hosting and inference infrastructure
- **NeurIPS 2025 Workshop**: For supporting this research direction

---

## 📧 Contact

For questions or collaboration inquiries:

- **Email**: zarreen.naowal.reza@gmail.com
- **GitHub Issues**: [Open an issue](https://github.com/znreza/multi-agent-LLM-eval-for-debate/issues)
- **Workshop**: [NeurIPS 2025 LLM Evaluation Workshop](https://sites.google.com/view/llm-eval-workshop/home?authuser=0)

---

## 🔮 Future Directions

- [ ] Integration with additional debate datasets (Debatepedia, Kialo)
- [ ] Multi-modal argumentation (text + images)
- [ ] Real-time human-AI debate interfaces
- [ ] Cross-lingual multi-agent evaluation
- [ ] Reinforcement learning from debate outcomes
- [ ] Adversarial robustness testing

---

**Built with ❤️ for advancing LLM evaluation science**
