"""
Action Translator - Language Planner's semantic matching component.
Translates free-form LLM actions to admissible action vocabulary.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_utils
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class ActionTranslator:
    """
    Translates LLM-generated actions to admissible action vocabulary
    using semantic similarity (Language Planner approach).
    """
    
    def __init__(
        self,
        actions_path: Optional[str] = None,
        examples_path: Optional[str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )
        
        # Load translation model
        self.model = SentenceTransformer(model_name)
        
        # Load admissible actions
        if actions_path is None:
            base_dir = Path(__file__).parent.parent.parent
            actions_path = base_dir / "configs" / "available_actions.json"
        
        with open(actions_path, 'r') as f:
            self.actions = json.load(f)
        
        # Embed actions
        self.action_embeddings = self.model.encode(
            self.actions,
            convert_to_tensor=True
        )
        
        # Load examples
        self.examples = []
        self.example_embeddings = None
        
        if examples_path is None:
            base_dir = Path(__file__).parent.parent.parent
            examples_path = base_dir / "configs" / "available_examples.json"
        
        if Path(examples_path).exists():
            with open(examples_path, 'r') as f:
                self.examples = json.load(f)
            
            # Embed example tasks
            example_tasks = [ex['task'] for ex in self.examples]
            self.example_embeddings = self.model.encode(
                example_tasks,
                convert_to_tensor=True
            )
    
    def translate_action(self, generated_action: str) -> Tuple[str, float]:
        """
        Translate free-form action to closest admissible action.
        
        Args:
            generated_action: Free-form action from LLM
        
        Returns:
            (admissible_action, similarity_score)
        """
        query_embedding = self.model.encode(
            generated_action,
            convert_to_tensor=True
        )
        
        # Compute cosine similarity
        cos_scores = st_utils.pytorch_cos_sim(
            query_embedding,
            self.action_embeddings
        )[0].cpu().numpy()
        
        # Get best match
        best_idx = np.argmax(cos_scores)
        best_score = float(cos_scores[best_idx])
        
        return self.actions[best_idx], best_score
    
    def find_similar_example(self, task: str) -> Tuple[dict, float]:
        """
        Find most similar example task for few-shot prompting.
        
        Args:
            task: Query task description
        
        Returns:
            (example_dict, similarity_score)
        """
        if not self.examples or self.example_embeddings is None:
            return None, 0.0
        
        query_embedding = self.model.encode(task, convert_to_tensor=True)
        
        cos_scores = st_utils.pytorch_cos_sim(
            query_embedding,
            self.example_embeddings
        )[0].cpu().numpy()
        
        best_idx = np.argmax(cos_scores)
        best_score = float(cos_scores[best_idx])
        
        return self.examples[best_idx], best_score
    
    def translate_plan(
        self,
        generated_steps: List[str],
        threshold: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        """
        Translate entire plan to admissible actions.
        
        Args:
            generated_steps: List of free-form actions from LLM
            threshold: Minimum similarity score to accept
        
        Returns:
            List of (original_step, translated_action, score)
        """
        results = []
        
        for step in generated_steps:
            # Clean step text
            step_clean = step.strip().lower()
            if step_clean.startswith('step'):
                # Remove step numbering
                parts = step_clean.split(':', 1)
                if len(parts) > 1:
                    step_clean = parts[1].strip()
            
            translated, score = self.translate_action(step_clean)
            results.append((step, translated, score))
        
        return results
