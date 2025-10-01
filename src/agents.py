# agents.py

import torch
from config import CONFIG
from huggingface_hub import InferenceClient
from llama_cpp import Llama
from prompts import (get_agent_initial_belief_prompt,
                     get_moderator_initial_opinion_prompt,
                     get_moderator_judge_prompt, get_moderator_summary_prompt,
                     get_structured_argument_prompt,
                     get_structured_belief_update_prompt,
                     get_theory_of_mind_prompt, get_unbiased_stance_prompt)
from psychometric_parser import (calculate_argument_complexity,
                                 calculate_tom_metrics,
                                 extract_epistemic_markers,
                                 parse_belief_update,
                                 parse_structured_argument,
                                 parse_theory_of_mind)
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import clean_llm_response

# Initialize client conditionally

client = InferenceClient(bill_to="agent-solitaire", provider="auto")

model = CONFIG["model"]
temperature = CONFIG["temperature"]
provider = CONFIG["provider"]


class Agent:
    def __init__(self, name, persona="neutral", incentive="none",
                 provider=provider,
                 model=model,
                 hf_model_id=model,
                 cpp_model_path=None):
        self.name = name
        self.persona = persona
        self.incentive = incentive
        self.beliefs = {}
        self.provider = provider
        self.model = model
        self.hf_model_id = hf_model_id
        self.cpp_model_path = cpp_model_path
        self.llama_model = None
        self.model_hf = None
        self.tokenizer = None

        if self.provider == "huggingface":
            # if not TRANSFORMERS_AVAILABLE:
            #     raise ImportError("Transformers and PyTorch are required for the 'huggingface' provider.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            self.model_hf = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                device_map="auto", # Use "auto" for better device mapping
                torch_dtype=torch.bfloat16
            )
        elif self.provider == "cpp":
            # if not LLAMA_CPP_AVAILABLE:
            #     raise ValueError("llama-cpp-python not available. Install with: pip install llama-cpp-python")
            if not cpp_model_path:
                cpp_model_path = CONFIG.get("cpp_model_path", "./models/Llama-3.2-3B-Instruct-Q5_K_M.gguf")
            print(f"Loading Llama.cpp model from: {cpp_model_path}")
            self.llama_model = Llama(
                model_path=cpp_model_path,
                n_gpu_layers=16,   # Reduce GPU layers to avoid memory issues
                n_ctx=10000,        # Increased context window for structured responses
                n_batch=256,       # Smaller batch size
                verbose=False,
                use_mlock=False,   # Don't lock memory
                n_threads=4        # Limit threads
            )

    def _call_llm(self, prompt, max_tokens=1200, use_system_prompt=True, override_temperature=None):
        """Route inference to the chosen provider with an increased token limit."""
        if self.provider == "inference_client":
            if not client:
                raise ValueError("Hugging Face InferenceClient is not initialized.")
            
            if use_system_prompt:
                messages = [
                    {"role": "system", "content": f"You are a {self.persona} with an incentive for {self.incentive}. Respond concisely and adhere to the requested format."},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
                
            use_temperature = override_temperature if override_temperature is not None else temperature
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=use_temperature
            )
            return clean_llm_response(completion.choices[0].message.content)

        elif self.provider == "huggingface":
            if use_system_prompt:
                messages = [
                    {"role": "system", "content": f"You are a {self.persona} with an incentive for {self.incentive}. Respond concisely and adhere to the requested format."},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
                
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model_hf.device)
            
            outputs = self.model_hf.generate(**inputs, max_new_tokens=max_tokens)
            decoded = self.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            return clean_llm_response(decoded[0])

        elif self.provider == "cpp":
            try:
                if use_system_prompt:
                    messages = [
                        {"role": "system", "content": f"You are a {self.persona} with an incentive for {self.incentive}. Respond concisely and adhere to the requested format."},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]
                    
                use_temperature = override_temperature if override_temperature is not None else temperature
                
                # Clear context to avoid memory issues
                self.llama_model.reset()
                
                output = self.llama_model.create_chat_completion(
                    messages=messages,
                    max_tokens=min(max_tokens, 1200),  # Increased limit for structured responses
                    temperature=use_temperature,
                    stop=["<|eot_id|>", "<|end_of_text|>"]
                )
                return clean_llm_response(output['choices'][0]['message']['content'])
            except Exception as e:
                print(f"Warning: llama-cpp error: {e}")
                return f"Error: Could not generate response - {str(e)}"

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def get_unbiased_stance(self, topic):
        """Get unbiased stance on topic without any system prompt influence."""
        prompt = get_unbiased_stance_prompt(topic)
        return self._call_llm(prompt, max_tokens=128, use_system_prompt=False, override_temperature=0.0)

    def set_initial_belief(self, topic, stance_hint=None):
        prompt = get_agent_initial_belief_prompt(
            persona=self.persona,
            incentive=self.incentive,
            topic=topic,
            stance_hint=stance_hint
        )
        belief = self._call_llm(prompt, max_tokens=128)
        self.beliefs[topic] = belief
        return belief

    def generate_structured_argument(self, topic, round_idx, opponent_arguments=None):
        """Generate argument with enhanced psychometric analysis."""
        current_belief = self.beliefs.get(topic, "No prior belief")
        
        prompt = get_structured_argument_prompt(
            persona=self.persona,
            incentive=self.incentive,
            topic=topic,
            round_number=round_idx,
            current_belief=current_belief,
            opponent_arguments=opponent_arguments or []
        )
        
        raw_response = self._call_llm(prompt, max_tokens=1200) # Increased for structured JSON
        main_argument, structured_analysis = parse_structured_argument(raw_response)
        
        complexity_metrics = calculate_argument_complexity(structured_analysis)
        epistemic_markers = extract_epistemic_markers(main_argument)
        
        return {
            "agent": self.name, "topic": topic, "round": round_idx,
            "stance": current_belief, "argument": main_argument,
            "context": {"persona": self.persona, "incentive": self.incentive},
            "structured_analysis": structured_analysis,
            "complexity_metrics": complexity_metrics,
            "epistemic_markers": epistemic_markers,
            "raw_response": raw_response
        }

    def analyze_theory_of_mind(self, topic, round_idx, opponent_arguments):
        """Analyze understanding of opponents' mental states."""
        prompt = get_theory_of_mind_prompt(
            persona=self.persona,
            incentive=self.incentive,
            topic=topic,
            round_number=round_idx,
            opponent_arguments=opponent_arguments
        )
        
        raw_response = self._call_llm(prompt, max_tokens=1000) # Increased for ToM JSON
        tom_analysis = parse_theory_of_mind(raw_response)
        
        tom_metrics = calculate_tom_metrics(tom_analysis)
        tom_analysis.update({
            "agent": self.name, "round": round_idx,
            "tom_metrics": tom_metrics, "raw_response": raw_response
        })
        
        return tom_analysis

    def update_belief_structured(self, topic, round_args):
        """Update belief with enhanced psychometric tracking."""
        other_args = [a["argument"] for a in round_args if a["agent"] != self.name]
        current_belief = self.beliefs.get(topic, 'No current belief')
        
        prompt = get_structured_belief_update_prompt(
            persona=self.persona,
            incentive=self.incentive,
            topic=topic,
            current_belief=current_belief,
            other_arguments=other_args
        )
        
        raw_response = self._call_llm(prompt, max_tokens=1000) # Increased for belief update JSON
        updated_belief, update_analysis = parse_belief_update(raw_response)
        
        self.beliefs[topic] = updated_belief
        
        return {
            "agent": self.name, "topic": topic,
            "previous_belief": current_belief, "updated_belief": updated_belief,
            "update_analysis": update_analysis, "raw_response": raw_response
        }


class Moderator:
    def __init__(self, style="neutral",
                 provider=provider,
                 model=model,
                 hf_model_id=model,
                 cpp_model_path=None):
        self.style = style
        self.provider = provider
        self.model = model
        self.hf_model_id = hf_model_id
        self.cpp_model_path = cpp_model_path
        self.llama_model = None
        self.model_hf = None
        self.tokenizer = None

        if self.provider == "huggingface":
            # if not TRANSFORMERS_AVAILABLE:
            #     raise ImportError("Transformers and PyTorch are required for the 'huggingface' provider.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            self.model_hf = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        elif self.provider == "cpp":
            # if not LLAMA_CPP_AVAILABLE:
            #     raise ValueError("llama-cpp-python not available. Install with: pip install llama-cpp-python")
            if not cpp_model_path:
                cpp_model_path = CONFIG.get("cpp_model_path", "./models/Llama-3.2-3B-Instruct-Q5_K_M.gguf")
            self.llama_model = Llama(
                model_path=cpp_model_path,
                n_gpu_layers=16,   # Reduce GPU layers to avoid memory issues
                n_ctx=10000,        # Increased context window for structured responses
                n_batch=256,       # Smaller batch size
                verbose=False,
                use_mlock=False,   # Don't lock memory
                n_threads=4        # Limit threads
            )

    def _call_llm(self, prompt, max_tokens=256, use_system_prompt=True, override_temperature=None):
        if self.provider == "inference_client":
            if not client:
                raise ValueError("Hugging Face InferenceClient is not initialized.")
            
            if use_system_prompt:
                messages = [
                    {"role": "system", "content": f"You are a {self.style} and impartial debate moderator."},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
                
            use_temperature = override_temperature if override_temperature is not None else temperature
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=use_temperature
            )
            return clean_llm_response(completion.choices[0].message.content)

        elif self.provider == "huggingface":
            if use_system_prompt:
                messages = [
                    {"role": "system", "content": f"You are a {self.style} and impartial debate moderator."},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
                
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model_hf.device)
            
            outputs = self.model_hf.generate(**inputs, max_new_tokens=max_tokens)
            decoded = self.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            return clean_llm_response(decoded[0])

        elif self.provider == "cpp":
            if use_system_prompt:
                messages = [
                    {"role": "system", "content": f"You are a {self.style} and impartial debate moderator."},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
                
            use_temperature = override_temperature if override_temperature is not None else temperature
            output = self.llama_model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=use_temperature,
                stop=["<|eot_id|>", "<|end_of_text|>"]
            )
            return clean_llm_response(output['choices'][0]['message']['content'])

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def get_unbiased_stance(self, topic):
        """Get unbiased stance on topic without any system prompt influence."""
        prompt = get_unbiased_stance_prompt(topic)
        return self._call_llm(prompt, max_tokens=128, use_system_prompt=False, override_temperature=0.0)

    def get_initial_opinion(self, topic):
        prompt = get_moderator_initial_opinion_prompt(topic)
        return self._call_llm(prompt, max_tokens=128)

    def summarize_round(self, topic, round_args):
        prompt = get_moderator_summary_prompt(
            style=self.style,
            topic=topic,
            round_arguments=round_args
        )
        return self._call_llm(prompt)

    def judge_winner(self, topic, all_rounds):
        prompt = get_moderator_judge_prompt(
            style=self.style,
            topic=topic,
            all_arguments=all_rounds
        )
        return self._call_llm(prompt)