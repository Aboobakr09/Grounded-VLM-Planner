"""
Enhanced Action Matcher - Language Planner Integration.

This module provides:
1. VirtualHome-compatible action vocabulary matching
2. Dynamic example retrieval for few-shot prompting
3. Autoregressive fallback generation (Language Planner's original approach)
4. Ranked action selection based on similarity + log probability

Based on: "Language Models as Zero-Shot Planners" (Huang et al., 2022)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_utils
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


@dataclass
class ActionMatchResult:
    """Result of matching a generated action to vocabulary."""
    original_text: str
    matched_action: str
    similarity_score: float
    log_prob: float = 0.0
    combined_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "original": self.original_text,
            "matched": self.matched_action,
            "similarity": self.similarity_score,
            "log_prob": self.log_prob,
            "combined_score": self.combined_score
        }


@dataclass
class AutoregressivePlanResult:
    """Result of autoregressive plan generation."""
    task: str
    example_used: Optional[Dict] = None
    example_similarity: float = 0.0
    steps: List[ActionMatchResult] = field(default_factory=list)
    terminated_early: bool = False
    termination_reason: str = ""
    
    def get_plan(self) -> List[str]:
        """Get list of matched actions."""
        return [step.matched_action for step in self.steps]
    
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "example_used": self.example_used,
            "example_similarity": self.example_similarity,
            "steps": [s.to_dict() for s in self.steps],
            "terminated_early": self.terminated_early,
            "termination_reason": self.termination_reason
        }


class EnhancedActionMatcher:
    """
    Enhanced action matcher combining:
    1. Large VirtualHome vocabulary (~30K actions from language-planner)
    2. Sentence similarity matching
    3. Dynamic example selection for prompting
    4. Autoregressive generation fallback (Language Planner approach)
    
    This replicates the core Language Planner approach:
    - Translation LM: SentenceTransformer for semantic matching
    - Planning LM: LLM (Gemini) for step generation
    - Scoring: similarity_score + BETA * log_prob
    """
    
    # Hyperparameters from Language Planner paper
    MAX_STEPS = 20
    CUTOFF_THRESHOLD = 0.6
    P = 0.5  # Top P% must be non-empty to continue
    BETA = 0.3  # Weight for log probability in ranking
    
    def __init__(
        self,
        actions_path: Optional[str] = None,
        examples_path: Optional[str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_virtualhome: bool = True
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
        
        # Use local configs directory
        config_dir = Path(__file__).parent.parent.parent / "configs"
        
        # Load actions
        if actions_path is not None:
            actions_path = Path(actions_path)
        elif use_virtualhome:
            actions_path = config_dir / "virtualhome_actions.json"
        else:
            actions_path = config_dir / "robot_actions.json"
        
        with open(actions_path, 'r') as f:
            self.actions = json.load(f)
        
        self.action_embeddings = self.model.encode(
            self.actions,
            convert_to_tensor=True
        )
        
        # Load examples
        if examples_path is not None:
            examples_path = Path(examples_path)
        elif use_virtualhome:
            examples_path = config_dir / "virtualhome_examples.json"
        else:
            examples_path = config_dir / "robot_examples.json"
        
        self.examples = []
        self.example_embeddings = None
        
        if Path(examples_path).exists():
            with open(examples_path, 'r') as f:
                raw_examples = json.load(f)
            
            # Parse examples - handle both formats:
            # 1. List of dicts: [{"task": "...", "steps": [...]}]
            # 2. List of strings: ["Task: ...\nStep 1: ..."]
            self.examples = self._parse_examples(raw_examples)
            
            if self.examples:
                example_tasks = [ex['task'] for ex in self.examples]
                self.example_embeddings = self.model.encode(
                    example_tasks,
                    convert_to_tensor=True
                )
        
        # Initialize LLM for autoregressive generation
        self.llm = None
        if HAS_GENAI and os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.llm = genai.GenerativeModel("gemini-2.0-flash")
    
    def _parse_examples(self, raw_examples: list) -> List[Dict]:
        """
        Parse examples from either format into unified dict format.
        
        Handles:
        1. List of dicts: [{"task": "X", "steps": ["a", "b"]}]
        2. List of strings: ["Task: X\nStep 1: a\nStep 2: b"]
        """
        parsed = []
        for item in raw_examples:
            if isinstance(item, dict) and 'task' in item and 'steps' in item:
                # Already in correct format
                parsed.append(item)
            elif isinstance(item, str):
                # Parse string format: "Task: X\nStep 1: a\nStep 2: b"
                lines = item.strip().split('\n')
                if not lines:
                    continue
                
                # Extract task
                task = lines[0]
                if task.startswith('Task:'):
                    task = task[5:].strip()
                
                # Extract steps
                steps = []
                for line in lines[1:]:
                    line = line.strip()
                    if line.startswith('Step'):
                        # Remove "Step N:" prefix
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            steps.append(parts[1].strip())
                        else:
                            steps.append(line)
                
                if task and steps:
                    parsed.append({'task': task, 'steps': steps})
        
        return parsed
    
    def match_action(self, text: str) -> ActionMatchResult:
        """
        Match free-form text to closest action in vocabulary.
        
        Args:
            text: Free-form action description
            
        Returns:
            ActionMatchResult with matched action and score
        """
        query_embedding = self.model.encode(text, convert_to_tensor=True)
        
        cos_scores = st_utils.pytorch_cos_sim(
            query_embedding,
            self.action_embeddings
        )[0].cpu().numpy()
        
        best_idx = int(np.argmax(cos_scores))
        best_score = float(cos_scores[best_idx])
        
        return ActionMatchResult(
            original_text=text,
            matched_action=self.actions[best_idx],
            similarity_score=best_score
        )
    
    def find_similar_example(self, task: str) -> Tuple[Optional[Dict], float]:
        """
        Find most similar example for few-shot prompting.
        
        This is the Language Planner's "dynamic example selection" approach.
        
        Args:
            task: Query task description
            
        Returns:
            (example_dict, similarity_score) or (None, 0.0) if no examples
        """
        if not self.examples or self.example_embeddings is None:
            return None, 0.0
        
        query_embedding = self.model.encode(task, convert_to_tensor=True)
        
        cos_scores = st_utils.pytorch_cos_sim(
            query_embedding,
            self.example_embeddings
        )[0].cpu().numpy()
        
        best_idx = int(np.argmax(cos_scores))
        best_score = float(cos_scores[best_idx])
        
        return self.examples[best_idx], best_score
    
    def translate_plan(
        self,
        steps: List[str],
        threshold: float = 0.5
    ) -> List[ActionMatchResult]:
        """
        Translate a list of free-form steps to matched actions.
        
        Args:
            steps: List of step descriptions
            threshold: Minimum similarity to accept
            
        Returns:
            List of ActionMatchResult
        """
        results = []
        for step in steps:
            step_clean = step.strip().lower()
            # Remove step numbering
            if step_clean.startswith('step'):
                parts = step_clean.split(':', 1)
                if len(parts) > 1:
                    step_clean = parts[1].strip()
            
            result = self.match_action(step_clean)
            results.append(result)
        
        return results
    
    def _format_example_prompt(self, example: Dict) -> str:
        """Format an example for the prompt."""
        lines = [f"Task: {example['task']}"]
        for i, step in enumerate(example['steps'], 1):
            lines.append(f"Step {i}: {step.capitalize()}")
        return '\n'.join(lines)
    
    def autoregressive_plan(
        self,
        task: str,
        max_steps: Optional[int] = None,
        beta: Optional[float] = None,
        cutoff_threshold: Optional[float] = None,
        verbose: bool = False
    ) -> AutoregressivePlanResult:
        """
        Generate plan using Language Planner's autoregressive approach.
        
        This is the CORE Language Planner algorithm:
        1. Find most similar example
        2. Build prompt: [Example] + [Task]
        3. For each step:
           a. Sample N completions from LLM
           b. Match each to closest action
           c. Rank by: similarity + BETA * log_prob
           d. Select best, append to prompt
           e. Check early stopping conditions
        
        Args:
            task: Task description
            max_steps: Maximum steps to generate
            beta: Weight for log probability (default 0.3)
            cutoff_threshold: Score threshold for early stop
            verbose: Print progress
            
        Returns:
            AutoregressivePlanResult
        """
        if self.llm is None:
            raise RuntimeError(
                "LLM not available. Set GEMINI_API_KEY or use translate_plan() instead."
            )
        
        max_steps = max_steps or self.MAX_STEPS
        beta = beta if beta is not None else self.BETA
        cutoff_threshold = cutoff_threshold if cutoff_threshold is not None else self.CUTOFF_THRESHOLD
        
        result = AutoregressivePlanResult(task=task)
        
        # Step 1: Find similar example
        example, sim_score = self.find_similar_example(task)
        result.example_used = example
        result.example_similarity = sim_score
        
        # Step 2: Build initial prompt
        if example:
            prompt = self._format_example_prompt(example)
            prompt += f"\n\nTask: {task}"
        else:
            prompt = f"Task: {task}"
        
        if verbose:
            print(f"Using example: '{example['task'] if example else 'None'}' (sim={sim_score:.2f})")
            print(f"Task: {task}")
        
        previous_action = None
        
        # Step 3: Autoregressive generation
        for step_num in range(1, max_steps + 1):
            step_prompt = prompt + f"\nStep {step_num}:"
            
            # Sample N completions
            try:
                response = self.llm.generate_content(
                    step_prompt + " (respond with just the action, one line)",
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 30,
                        "candidate_count": 1  # Gemini limitation
                    }
                )
                
                # Parse response
                generated_text = response.text.strip().lower()
                # Clean up
                if generated_text.startswith("step"):
                    parts = generated_text.split(":", 1)
                    if len(parts) > 1:
                        generated_text = parts[1].strip()
                
                # Remove trailing punctuation
                generated_text = generated_text.rstrip('.')
                
            except Exception as e:
                if verbose:
                    print(f"Generation error: {e}")
                result.terminated_early = True
                result.termination_reason = f"LLM error: {e}"
                break
            
            # Check for empty generation
            if not generated_text or len(generated_text) < 2:
                result.terminated_early = True
                result.termination_reason = "Empty generation"
                break
            
            # Match to action vocabulary
            match_result = self.match_action(generated_text)
            
            # Penalize repeating the same action
            if previous_action and match_result.matched_action == previous_action:
                match_result.combined_score = match_result.similarity_score - 0.5
            else:
                match_result.combined_score = match_result.similarity_score
            
            # Check cutoff threshold
            if match_result.similarity_score < cutoff_threshold:
                result.terminated_early = True
                result.termination_reason = f"Score below threshold ({match_result.similarity_score:.2f} < {cutoff_threshold})"
                break
            
            # Add to results
            result.steps.append(match_result)
            previous_action = match_result.matched_action
            
            if verbose:
                print(f"Step {step_num}: '{generated_text}' → '{match_result.matched_action}' (score={match_result.similarity_score:.2f})")
            
            # Update prompt for next iteration
            formatted_action = match_result.matched_action.capitalize()
            prompt += f"\nStep {step_num}: {formatted_action}"
            
            # Check for common termination signals
            if any(term in generated_text.lower() for term in ["done", "finished", "complete", "end"]):
                result.terminated_early = True
                result.termination_reason = "Detected completion signal"
                break
        
        return result


def create_enhanced_matcher(use_virtualhome: bool = True) -> EnhancedActionMatcher:
    """
    Factory function to create an EnhancedActionMatcher.
    
    Args:
        use_virtualhome: If True, use VirtualHome vocabulary (larger, household tasks)
                        If False, use basic action vocabulary (smaller, robot manipulation)
    
    Returns:
        Configured EnhancedActionMatcher
    """
    return EnhancedActionMatcher(use_virtualhome=use_virtualhome)
