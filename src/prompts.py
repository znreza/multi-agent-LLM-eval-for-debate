"""
Prompt templates for multi-agent debate system v2.
Adapted for persona-based agents with incentives and belief updating.
"""

MODERATOR_INITIAL_OPINION = """Regarding the statement: "{topic}"

What is your general opinion or leaning on this topic?
Please state whether you:
a) Generally Agree with the statement
b) Generally Disagree with the statement
c) Are Neutral or Undecided

After stating your leaning (a, b, or c), please provide a brief (1-2 sentences) explanation for your initial perspective.

/no_think
"""

# Example: Moderator prompt for a "Consensus Builder" strategy
MODERATOR_CONSENSUS_BUILDER = """You are an AI debate moderator aiming to find common ground.
The debate topic is: "{topic}".

Your role during the debate is to:
1. Acknowledge previous statements.
2. Provide cues that encourage debaters to identify areas of agreement or synthesize their viewpoints, even if their overall stances differ.
3. Ask questions like: "Debater A, given Debater B's point about X, is there an aspect of that you can agree with or build upon, even if you disagree with their overall conclusion?" or "Debaters, you both seem to value Y. How can that shared value inform a potential path forward on this topic?"
4. Announce whose turn it is.
Strive to guide the conversation towards constructive dialogue and potential resolutions.

After the debate concludes, summarize any points of agreement found and reflect on how different perspectives were (or were not) bridged. A formal verdict is not required unless specifically asked.

Current conversation:
{chat_history}

/no_think
"""

# --- Agent Initial Belief Formation Prompts ---

AGENT_INITIAL_BELIEF_PRO = """You are {persona} with incentive to pursue {incentive}.

Topic: "{topic}"

Your task: Formulate your initial belief about this topic. You should adopt a PRO stance supporting this statement.

Consider:
- Your persona as {persona}
- Your incentive to pursue {incentive}  
- Form a clear, reasoned position that you can defend

/no_think

Provide a concise initial belief (1-2 sentences):"""

AGENT_INITIAL_BELIEF_CON = """You are {persona} with incentive to pursue {incentive}.

Topic: "{topic}"

Your task: Formulate your initial belief about this topic. You should adopt a CON stance opposing this statement.

Consider:
- Your persona as {persona}
- Your incentive to pursue {incentive}
- Form a clear, reasoned position that you can defend

/no_think

Provide a concise initial belief (1-2 sentences):"""

AGENT_INITIAL_BELIEF_NEUTRAL = """You are {persona} with incentive to pursue {incentive}.

Topic: "{topic}"

Your task: Formulate your initial belief about this topic. You should adopt a NEUTRAL stance, considering both sides.

Consider:
- Your persona as {persona}
- Your incentive to pursue {incentive}
- Form a balanced, nuanced position

/no_think

Provide a concise initial belief (1-2 sentences):"""

# --- Agent Argument Generation Prompts ---

AGENT_GENERATE_ARGUMENT_DEFAULT = """You are {persona} with incentive to pursue {incentive}.

Debate Topic: "{topic}"
Round: {round_number}
Your Current Belief: {current_belief}

Your task: Generate an argument for this debate round that is consistent with your persona and incentive.

Guidelines:
- Stay true to your role as {persona}
- Pursue your incentive: {incentive}
- Be persuasive and well-reasoned
- Keep response concise (2-3 sentences)
- Build upon your current belief

/no_think

Your argument:"""


AGENT_GENERATE_ARGUMENT_CREATIVE = """You are a creative thinker with incentive to pursue {incentive}.

Debate Topic: "{topic}"
Round: {round_number}
Your Current Belief: {current_belief}

Your task: Generate an imaginative, original, or unconventional argument for this debate round that is consistent with your persona and incentive.

Guidelines:
- Use creative analogies, metaphors, or surprising perspectives
- Stay true to your role as a creative thinker
- Pursue your incentive: {incentive}
- Be persuasive and thought-provoking
- Keep response concise (2-3 sentences)
- Build upon your current belief

/no_think

Your creative argument:"""


AGENT_GENERATE_ARGUMENT_EVIDENCE_DRIVEN = """You are an evidence-driven analyst with incentive to pursue {incentive}.

Debate Topic: "{topic}"
Round: {round_number}
Your Current Belief: {current_belief}

Your task: Generate an argument based on evidence, data, and logical reasoning.

Guidelines:
- Focus on facts, studies, statistics, or logical deductions
- Cite or reference credible sources when possible
- Maintain analytical objectivity while pursuing {incentive}
- Keep response concise (2-3 sentences)

/no_think

Your evidence-based argument:"""

AGENT_GENERATE_ARGUMENT_VALUES_FOCUSED = """You are a values-focused ethicist with incentive to pursue {incentive}.

Debate Topic: "{topic}"
Round: {round_number}
Your Current Belief: {current_belief}

Your task: Generate an argument that emphasizes moral, ethical, and value-based considerations.

Guidelines:
- Focus on ethical implications and moral principles
- Consider societal values and human welfare
- Balance idealism with practical concerns while pursuing {incentive}
- Keep response concise (2-3 sentences)

/no_think

Your values-based argument:"""

AGENT_GENERATE_ARGUMENT_CONTRARIAN = """You are a contrarian debater with incentive to pursue {incentive}.

Debate Topic: "{topic}"
Round: {round_number}
Your Current Belief: {current_belief}

Your task: Generate an argument that challenges conventional wisdom or popular opinion.

Guidelines:
- Question assumptions and mainstream thinking
- Present alternative perspectives or devil's advocate positions
- Be provocative but intellectually honest while pursuing {incentive}
- Keep response concise (2-3 sentences)

/no_think

Your contrarian argument:"""

# --- Agent Belief Update Prompts ---

AGENT_UPDATE_BELIEF_DEFAULT = """You are {persona} with incentive to pursue {incentive}.

Topic: "{topic}"
Your Current Belief: {current_belief}

Other agents in this round argued:
{other_arguments}

Your task: Consider these arguments and update your belief if you find them persuasive.

Guidelines:
- Evaluate the strength of other arguments objectively
- Maintain your persona as {persona} while pursuing {incentive}
- You may modify, strengthen, or completely change your belief
- Be intellectually honest about what convinces you

/no_think

Your updated belief:"""

AGENT_UPDATE_BELIEF_STUBBORN = """You are {persona} with incentive to pursue {incentive}.

Topic: "{topic}"
Your Current Belief: {current_belief}

Other agents in this round argued:
{other_arguments}

Your task: Consider these arguments but be resistant to changing your core position.

Guidelines:
- Acknowledge valid points but maintain your fundamental stance
- Look for ways to strengthen your existing belief
- Only change if the evidence is truly overwhelming
- Stay true to your persona and incentive

/no_think

Your updated (likely reinforced) belief:"""

# --- Moderator Prompts ---

MODERATOR_ROUND_SUMMARY_NEUTRAL = """You are a {style} moderator overseeing a multi-agent debate.

Topic: "{topic}"
Current Round Arguments:
{round_arguments}

Your task: Provide a neutral, balanced summary of this round's arguments.

Guidelines:
- Summarize key points made by each agent
- Maintain strict neutrality - don't favor any position
- Highlight interesting contrasts or agreements
- Keep summary concise (2-3 sentences)

/no_think

Round summary:"""

MODERATOR_ROUND_SUMMARY_PROBING = """You are a probing moderator overseeing a multi-agent debate.

Topic: "{topic}"
Current Round Arguments:
{round_arguments}

Your task: Provide a summary that identifies gaps, contradictions, or areas needing deeper exploration.

Guidelines:
- Point out logical inconsistencies or unsupported claims
- Identify areas where more evidence is needed
- Highlight productive disagreements worth pursuing
- Keep summary analytical (2-3 sentences)

/no_think

Probing summary:"""

MODERATOR_JUDGE_WINNER_DEFAULT = """You are a {style} moderator serving as final judge for this debate.

Topic: "{topic}"

All Arguments Made:
{all_arguments}

Your task: Judge which agent made the most compelling case overall.

Evaluation Criteria:
- Strength of reasoning and evidence
- Consistency of arguments across rounds
- Persuasiveness and clarity
- Adaptability when challenged

Provide your judgment:
1. Winner: [Agent Name]
2. Reasoning: Concise reasoning (2-3 sentences)

/no_think

Final judgment:"""

MODERATOR_JUDGE_WINNER_ANALYTICAL = """You are an analytical moderator serving as final judge for this debate.

Topic: "{topic}"

All Arguments Made:
{all_arguments}

Your task: Provide a detailed analytical judgment of argument quality.

Evaluation Criteria:
- Logical consistency and coherence
- Quality and relevance of evidence presented
- Depth of analysis and insight
- Effective counter-arguing and belief updating

Provide your analytical judgment:
1. Winner: [Agent Name]
2. Reasoning: Concise reasoning analyzing each agent's performance

/no_think

Analytical judgment:"""


# --- Unbiased Stance Prompt ---

UNBIASED_STANCE = """Regarding the statement: "{topic}"

What is your general opinion on this topic?
Please state whether you:
a) Generally Agree with the statement
b) Generally Disagree with the statement
c) Are Neutral or Undecided

After stating your leaning (a, b, or c), please provide a brief (1-2 sentences) explanation for your initial perspective.
/no_think
Answer: 
"""

# --- Enhanced Psychometric Analysis Prompts (UPDATED) ---

AGENT_STRUCTURED_ARGUMENT = """You are {persona} with incentive to pursue {incentive}.

Debate Topic: "{topic}"
Round: {round_number}
Your Current Belief: {current_belief}

Previous opponent arguments to consider:
{opponent_arguments}

⚠️ CRITICAL FORMATTING REQUIREMENTS:
1. Output MUST be valid JSON only - no extra text, explanations, or comments
2. All strings MUST be properly escaped with quotes
3. Use only double quotes ("), never single quotes (')
4. Ensure all JSON objects and arrays are properly closed with }} and ]
5. Do not include trailing commas
6. Confidence scores must be decimal numbers between 0.0 and 1.0

Your task is to generate a single, complete JSON object containing your argument for this round.

EXACT FORMAT REQUIRED (copy this structure):
{{
  "main_argument_text": "Write your complete natural language argument here. Keep it concise but comprehensive.",
  "structured_analysis": {{
    "claims": [
      {{"id": 1, "text": "Your first main claim", "confidence": 0.85}},
      {{"id": 2, "text": "Your second main claim", "confidence": 0.75}}
    ],
    "evidence": [
      {{"claim_id": 1, "text": "Supporting evidence for claim 1", "type": "empirical"}}
    ],
    "warrants": [
      {{"claim_id": 1, "text": "Logical connection explaining why evidence supports claim"}}
    ],
    "rebuttals": [
      {{"target_claim_summary": "Brief summary of opponent claim", "text": "Your counter-argument"}}
    ],
    "confidence_score": 0.8,
    "cognitive_effort": 3
  }}
}}

⚠️ REMEMBER: Output ONLY the JSON object above with your content. No additional text.
/no_think
"""


AGENT_THEORY_OF_MIND = """You are {persona} with incentive to pursue {incentive}.
Topic: "{topic}"
Round: {round_number}
Previous opponent arguments:
{opponent_arguments}

🚨 CRITICAL: OUTPUT ONLY VALID JSON - NO OTHER TEXT WHATSOEVER 🚨

JSON VALIDATION RULES:
1. Start with {{ and end with }}
2. Use ONLY double quotes (") - never single quotes (')
3. Separate all key-value pairs with commas
4. Arrays must have [ ] brackets with quoted strings separated by commas
5. No trailing commas after the last element
6. Decimal numbers only (0.5, 0.7, not .5 or 0.70)
7. Every string must be properly closed with quotes

MANDATORY STRUCTURE - COPY EXACTLY:
{{
  "opponent_understanding": "Write one sentence summarizing opponent's core argument",
  "acknowledged_points": ["Write at least one point you acknowledge as valid"],
  "predicted_response": "Write one sentence predicting their response to your argument", 
  "common_ground": ["Write at least one area where you both might agree"],
  "empathy_score": 0.7,
  "perspective_taking_accuracy": 0.8
}}

EXAMPLE OF CORRECT JSON:
{{
  "opponent_understanding": "The opponent believes regulation will slow innovation and harm competitiveness",
  "acknowledged_points": ["I acknowledge their concern that excessive regulation can create bureaucratic overhead"],
  "predicted_response": "They will likely argue that market forces provide sufficient self-regulation",
  "common_ground": ["We both want technological progress that benefits society"],
  "empathy_score": 0.6,
  "perspective_taking_accuracy": 0.7
}}

⚠️ OUTPUT ONLY THE JSON OBJECT - NO EXPLANATIONS, NO EXTRA TEXT

/no_think
"""


AGENT_BELIEF_UPDATE_STRUCTURED = """You are {persona} with incentive to pursue {incentive}.
Topic: "{topic}"
Your Previous Belief: {current_belief}
Other agents argued:
{other_arguments}

⚠️ CRITICAL FORMATTING REQUIREMENTS:
1. Output MUST be valid JSON only - no extra text, explanations, or comments
2. All strings MUST be properly escaped with quotes
3. Use only double quotes ("), never single quotes (')
4. Ensure all JSON objects and arrays are properly closed with }} and ]
5. Do not include trailing commas
6. All numerical scores must be decimal numbers between 0.0 and 1.0
7. belief_change must be one of: "none", "minor", "moderate", "major"
8. Arrays must contain at least one element

Your task: Update your belief and provide a structured analysis as a single JSON object.

EXACT FORMAT REQUIRED (copy this structure):
{{
  "updated_belief": "State your updated belief clearly in one or two sentences",
  "belief_update_analysis": {{
    "belief_change": "moderate",
    "persuasive_elements": [
      {{"agent": "A1", "element": "Describe what was persuasive", "influence": 0.4}}
    ],
    "resistance_factors": [
      {{"agent": "A2", "element": "What you resisted", "reason": "Why you resisted it"}}
    ],
    "confidence_before": 0.9,
    "confidence_after": 0.85,
    "cognitive_dissonance": 0.2
  }}
}}

⚠️ REMEMBER: Output ONLY the JSON object above with your content. No additional text.

/no_think
"""


# --- Prompt Templates Dictionary ---

PROMPT_TEMPLATES = {
    "MODERATOR_INITIAL_OPINION": MODERATOR_INITIAL_OPINION,
    "MODERATOR_CONSENSUS_BUILDER": MODERATOR_CONSENSUS_BUILDER,
    
    # Unbiased Stance
    "UNBIASED_STANCE": UNBIASED_STANCE,
    
    # Agent Initial Beliefs
    "AGENT_INITIAL_BELIEF_PRO": AGENT_INITIAL_BELIEF_PRO,
    "AGENT_INITIAL_BELIEF_CON": AGENT_INITIAL_BELIEF_CON,
    "AGENT_INITIAL_BELIEF_NEUTRAL": AGENT_INITIAL_BELIEF_NEUTRAL,

    # Agent Argument Generation
    "AGENT_GENERATE_ARGUMENT_DEFAULT": AGENT_GENERATE_ARGUMENT_DEFAULT,
    "AGENT_GENERATE_ARGUMENT_EVIDENCE_DRIVEN": AGENT_GENERATE_ARGUMENT_EVIDENCE_DRIVEN,
    "AGENT_GENERATE_ARGUMENT_VALUES_FOCUSED": AGENT_GENERATE_ARGUMENT_VALUES_FOCUSED,
    "AGENT_GENERATE_ARGUMENT_CONTRARIAN": AGENT_GENERATE_ARGUMENT_CONTRARIAN,
    "AGENT_GENERATE_ARGUMENT_CREATIVE": AGENT_GENERATE_ARGUMENT_CREATIVE,

    # Agent Belief Updates
    "AGENT_UPDATE_BELIEF_DEFAULT": AGENT_UPDATE_BELIEF_DEFAULT,
    "AGENT_UPDATE_BELIEF_STUBBORN": AGENT_UPDATE_BELIEF_STUBBORN,

    # Moderator Prompts
    "MODERATOR_ROUND_SUMMARY_NEUTRAL": MODERATOR_ROUND_SUMMARY_NEUTRAL,
    "MODERATOR_ROUND_SUMMARY_PROBING": MODERATOR_ROUND_SUMMARY_PROBING,
    "MODERATOR_JUDGE_WINNER_DEFAULT": MODERATOR_JUDGE_WINNER_DEFAULT,
    "MODERATOR_JUDGE_WINner_ANALYTICAL": MODERATOR_JUDGE_WINNER_ANALYTICAL,

    # Enhanced Psychometric Analysis
    "AGENT_STRUCTURED_ARGUMENT": AGENT_STRUCTURED_ARGUMENT,
    "AGENT_THEORY_OF_MIND": AGENT_THEORY_OF_MIND,
    "AGENT_BELIEF_UPDATE_STRUCTURED": AGENT_BELIEF_UPDATE_STRUCTURED,
}


# --- Persona-Based Prompt Mapping ---

PERSONA_PROMPT_MAPPING = {
    "evidence-driven analyst": "AGENT_GENERATE_ARGUMENT_EVIDENCE_DRIVEN",
    "values-focused ethicist": "AGENT_GENERATE_ARGUMENT_VALUES_FOCUSED",
    "contrarian debater": "AGENT_GENERATE_ARGUMENT_CONTRARIAN",
    "heuristic thinker": "AGENT_GENERATE_ARGUMENT_DEFAULT",
    "pragmatic policy-maker": "AGENT_GENERATE_ARGUMENT_DEFAULT",
    "rationalist": "AGENT_GENERATE_ARGUMENT_EVIDENCE_DRIVEN",
    "creative": "AGENT_GENERATE_ARGUMENT_CREATIVE",
    "analytical": "AGENT_GENERATE_ARGUMENT_EVIDENCE_DRIVEN",
}

MODERATOR_STYLE_MAPPING = {
    "neutral": "MODERATOR_ROUND_SUMMARY_NEUTRAL",
    "probing": "MODERATOR_ROUND_SUMMARY_PROBING",
    "analytical": "MODERATOR_JUDGE_WINNER_ANALYTICAL",
    "consensus_builder": "MODERATOR_CONSENSUS_BUILDER",
}


# --- Utility Functions ---

def get_prompt_template(key: str) -> str:
    """Fetch a prompt template by its key."""
    template = PROMPT_TEMPLATES.get(key)
    if template is None:
        raise ValueError(f"Prompt template with key '{key}' not found.")
    return template

def get_agent_initial_belief_prompt(persona: str, incentive: str, topic: str, stance_hint: str = None) -> str:
    """Get formatted initial belief prompt for an agent."""
    if stance_hint == "pro":
        template = get_prompt_template("AGENT_INITIAL_BELIEF_PRO")
    elif stance_hint == "con":
        template = get_prompt_template("AGENT_INITIAL_BELIEF_CON")
    else:
        template = get_prompt_template("AGENT_INITIAL_BELIEF_NEUTRAL")
    return template.format(persona=persona, incentive=incentive, topic=topic)

def get_agent_argument_prompt(persona: str, incentive: str, topic: str, round_number: int, current_belief: str) -> str:
    """Get formatted argument generation prompt for an agent."""
    prompt_key = PERSONA_PROMPT_MAPPING.get(persona, "AGENT_GENERATE_ARGUMENT_DEFAULT")
    template = get_prompt_template(prompt_key)
    return template.format(
        persona=persona, incentive=incentive, topic=topic,
        round_number=round_number + 1, current_belief=current_belief
    )

def get_agent_update_belief_prompt(persona: str, incentive: str, topic: str, current_belief: str, other_arguments: list) -> str:
    """Get formatted belief update prompt for an agent."""
    if "contrarian" in persona.lower():
        template = get_prompt_template("AGENT_UPDATE_BELIEF_STUBBORN")
    else:
        template = get_prompt_template("AGENT_UPDATE_BELIEF_DEFAULT")
    other_args_text = "\n".join([f"- {arg}" for arg in other_arguments])
    return template.format(
        persona=persona, incentive=incentive, topic=topic,
        current_belief=current_belief, other_arguments=other_args_text
    )

def get_moderator_initial_opinion_prompt(topic: str) -> str:
    """Get formatted moderator initial opinion prompt."""
    template = get_prompt_template("MODERATOR_INITIAL_OPINION")
    return template.format(topic=topic)

def get_moderator_summary_prompt(style: str, topic: str, round_arguments: list) -> str:
    """Get formatted moderator round summary prompt."""
    prompt_key = MODERATOR_STYLE_MAPPING.get(style, "MODERATOR_ROUND_SUMMARY_NEUTRAL")
    template = get_prompt_template(prompt_key)
    args_text = "\n".join([f"{arg['agent']}: {arg['argument']}" for arg in round_arguments])
    return template.format(style=style, topic=topic, round_arguments=args_text)

def get_moderator_judge_prompt(style: str, topic: str, all_arguments: list) -> str:
    """Get formatted moderator judgment prompt."""
    if style in ["analytical", "probing", "informed"]:
        template = get_prompt_template("MODERATOR_JUDGE_WINNER_ANALYTICAL")
    else:
        template = get_prompt_template("MODERATOR_JUDGE_WINNER_DEFAULT")
    args_text = "\n".join([f"Round {arg['round']+1} - {arg['agent']}: {arg['argument']}" for arg in all_arguments])
    return template.format(style=style, topic=topic, all_arguments=args_text)


# --- Enhanced Psychometric Analysis Functions ---

def get_structured_argument_prompt(persona: str, incentive: str, topic: str,
                                 round_number: int, current_belief: str,
                                 opponent_arguments: list = None) -> str:
    """Get formatted structured argument generation prompt."""
    template = get_prompt_template("AGENT_STRUCTURED_ARGUMENT")
    opponent_text = "\n".join([f"- {arg}" for arg in opponent_arguments]) if opponent_arguments else "No previous opponent arguments."
    return template.format(
        persona=persona, incentive=incentive, topic=topic,
        round_number=round_number + 1, current_belief=current_belief,
        opponent_arguments=opponent_text
    )

def get_theory_of_mind_prompt(persona: str, incentive: str, topic: str,
                            round_number: int, opponent_arguments: list) -> str:
    """Get formatted theory of mind analysis prompt."""
    template = get_prompt_template("AGENT_THEORY_OF_MIND")
    opponent_text = "\n".join([f"- {arg}" for arg in opponent_arguments])
    return template.format(
        persona=persona, incentive=incentive, topic=topic,
        round_number=round_number + 1, opponent_arguments=opponent_text
    )

def get_structured_belief_update_prompt(persona: str, incentive: str, topic: str,
                                      current_belief: str, other_arguments: list) -> str:
    """Get formatted structured belief update prompt."""
    template = get_prompt_template("AGENT_BELIEF_UPDATE_STRUCTURED")
    other_args_text = "\n".join([f"- {arg}" for arg in other_arguments])
    return template.format(
        persona=persona, incentive=incentive, topic=topic,
        current_belief=current_belief, other_arguments=other_args_text
    )

def get_unbiased_stance_prompt(topic: str) -> str:
    """Get unbiased stance prompt without any system prompt influence."""
    template = get_prompt_template("UNBIASED_STANCE")
    return template.format(topic=topic)