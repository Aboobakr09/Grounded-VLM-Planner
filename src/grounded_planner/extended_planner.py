"""
Extended Planner - Language Planner + Grounding Extensions
Combines Language Planner's action translation with our grounding verification.
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from grounded_planner.scene_graph import SceneGraph
from grounded_planner.grounding import GroundingVerifier
from grounded_planner.prompts import COT_PROMPT, CONSTRAINT_PROMPT
from grounded_planner.action_translator import ActionTranslator
from grounded_planner.planner import WaypointGenerator


@dataclass
class ExtendedPlanResult:
    task: str
    task_type: str = ""
    
    # Language Planner components
    similar_example: Optional[Dict] = None
    example_similarity: float = 0.0
    raw_steps: List[str] = field(default_factory=list)  # Free-form LLM output
    translated_steps: List[Tuple[str, str, float]] = field(default_factory=list)  # (raw, translated, score)
    
    # Our extensions
    grounding_success: bool = True
    grounding_message: str = ""
    missing_objects: List[str] = field(default_factory=list)
    low_confidence_objects: List[str] = field(default_factory=list)
    inferred_relations: List[Dict] = field(default_factory=list)
    
    # Planning outputs
    scene_graph: Dict = field(default_factory=dict)
    waypoints: List[Dict] = field(default_factory=list)
    constraints: str = ""
    
    execution_ready: bool = False  # True only if grounding passed AND actions translated


class ExtendedPlanner:
    """
    Extended planner that combines:
    1. Language Planner's task decomposition + action translation
    2. Our grounding verification + scene graph + geometric consistency
    """
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        if not HAS_GENAI:
            raise ImportError("google-generativeai package required")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        
        # Our extensions
        self.verifier = GroundingVerifier()
        
        # Language Planner components
        try:
            self.translator = ActionTranslator()
            self.has_translator = True
        except ImportError:
            print("sentence-transformers not installed. Action translation disabled.")
            print("   Install: pip install sentence-transformers")
            self.has_translator = False
    
    def plan(self, task: str, scene: SceneGraph, use_grounding: bool = True) -> ExtendedPlanResult:
        """
        Execute extended planning pipeline:
        1. Find similar example (Language Planner)
        2. Grounding verification (Our extension)
        3. CoT decomposition
        4. Action translation (Language Planner)
        5. Waypoint generation
        6. Constraint generation
        """
        result = ExtendedPlanResult(task=task)
        result.task_type = WaypointGenerator.detect_task_type(task)
        
        # Stage 0a: Find similar example (Language Planner approach)
        if self.has_translator and self.translator.examples:
            example, sim_score = self.translator.find_similar_example(task)
            result.similar_example = example
            result.example_similarity = sim_score
        
        # Stage 0b: Grounding Verification (OUR EXTENSION)
        if use_grounding:
            grounding = self.verifier.verify_task_grounding(task, scene.to_dict())
            result.grounding_success = grounding.success
            result.grounding_message = grounding.message
            result.missing_objects = grounding.missing_objects
            result.low_confidence_objects = [o['name'] for o in grounding.low_confidence_objects]
            result.inferred_relations = [r.to_dict() for r in grounding.inferred_relations]
            
            if not grounding.success:
                # ABORT: Grounding failed
                result.execution_ready = False
                return result
        
        # Stage 1: CoT Decomposition
        result.raw_steps = self._decompose_task(task, result.similar_example)
        
        # Stage 2: Action Translation (Language Planner approach)
        if self.has_translator and result.raw_steps:
            result.translated_steps = self.translator.translate_plan(result.raw_steps)
        
        # Stage 3: Scene Graph
        result.scene_graph = scene.to_dict()
        
        # Stage 4: Waypoint Generation
        result.waypoints = WaypointGenerator.generate(task, scene)
        
        # Stage 5: Constraint Generation
        result.constraints = self._generate_constraints(task, scene)
        
        # Check if ready for execution
        result.execution_ready = (
            result.grounding_success and
            len(result.waypoints) > 0
        )
        
        return result
    
    def _decompose_task(self, task: str, example: Optional[Dict] = None) -> List[str]:
        """
        CoT decomposition with optional example (Language Planner's dynamic example selection).
        """
        # Build prompt with example if available
        if example:
            example_text =f"""Example Task: {example['task']}
Steps:
""" + "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(example['steps'])])
            
            prompt = f"""{example_text}

Now decompose this task:
Task: {task}
Steps:
"""
        else:
            prompt = COT_PROMPT.format(task=task)
        
        # Generate decomposition
        response = self.model.generate_content(prompt)
        
        # Parse steps
        steps = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line.startswith('Step '):
                step_text = re.sub(r'^Step \d+:\s*', '', line)
                steps.append(step_text)
        
        return steps
    
    def _generate_constraints(self, task: str, scene: SceneGraph) -> str:
        """Generate ReKep-style constraints."""
        prompt = CONSTRAINT_PROMPT.format(
            task=task,
            objects=json.dumps(scene.to_dict(), indent=2)
        )
        response = self.model.generate_content(prompt)
        
        code_match = re.search(r'```python\s*(.+?)```', response.text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response.text[:500]
