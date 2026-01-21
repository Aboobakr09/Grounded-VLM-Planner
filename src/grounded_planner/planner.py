"""Planner: Combined planning with grounding verification."""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from grounded_planner.scene_graph import SceneGraph
from grounded_planner.grounding import GroundingVerifier
from grounded_planner.prompts import COT_PROMPT, CONSTRAINT_PROMPT


@dataclass
class PlanResult:
    task: str
    task_type: str = ""
    reasoning: List[str] = field(default_factory=list)
    scene_graph: Dict = field(default_factory=dict)
    constraints: str = ""
    waypoints: List[Dict] = field(default_factory=list)
    grounding_success: bool = True
    grounding_message: str = ""
    inferred_relations: List[Dict] = field(default_factory=list)


class WaypointGenerator:
    
    BLOCK_HEIGHT = 0.05
    APPROACH_HEIGHT = 0.10
    LIFT_HEIGHT = 0.15
    
    @staticmethod
    def detect_task_type(task: str) -> str:
        task_lower = task.lower()
        if 'pick up' in task_lower:
            return 'pick'
        elif 'stack' in task_lower:
            return 'stack'
        elif 'all' in task_lower and ('box' in task_lower or 'in' in task_lower):
            return 'multi_place'
        elif 'left' in task_lower or 'right' in task_lower:
            return 'spatial'
        elif 'tower' in task_lower or 'build' in task_lower:
            return 'tower'
        return 'generic'
    
    @classmethod
    def generate(cls, task: str, scene: SceneGraph) -> List[Dict]:
        task_type = cls.detect_task_type(task)
        generators = {
            'pick': cls._generate_pick,
            'stack': cls._generate_stack,
            'multi_place': cls._generate_multi_place,
            'spatial': cls._generate_spatial,
            'tower': cls._generate_tower,
        }
        generator = generators.get(task_type, cls._generate_generic)
        return generator(task, scene)
    
    @classmethod
    def _pick_and_place(cls, source_pos: List, target_pos: Optional[List], source_name: str, target_name: Optional[str], target_height_offset: float = 0.0) -> List[Dict]:
        waypoints = []
        
        waypoints.append({'pos': [source_pos[0], source_pos[1], source_pos[2] + cls.APPROACH_HEIGHT], 'action': 'approach', 'label': f'above {source_name}'})
        waypoints.append({'pos': source_pos.copy(), 'action': 'descend', 'label': f'at {source_name}'})
        waypoints.append({'pos': source_pos.copy(), 'action': 'grasp', 'label': f'grasp {source_name}'})
        waypoints.append({'pos': [source_pos[0], source_pos[1], source_pos[2] + cls.LIFT_HEIGHT], 'action': 'lift', 'label': f'lift {source_name}'})
        
        if target_pos:
            waypoints.append({'pos': [target_pos[0], target_pos[1], target_pos[2] + cls.LIFT_HEIGHT + target_height_offset], 'action': 'move', 'label': f'above {target_name}'})
            waypoints.append({'pos': [target_pos[0], target_pos[1], target_pos[2] + cls.BLOCK_HEIGHT + target_height_offset], 'action': 'descend', 'label': f'place on {target_name}'})
            waypoints.append({'pos': [target_pos[0], target_pos[1], target_pos[2] + cls.APPROACH_HEIGHT + target_height_offset], 'action': 'release', 'label': 'release'})
        
        return waypoints
    
    @classmethod
    def _generate_pick(cls, task: str, scene: SceneGraph) -> List[Dict]:
        for obj in scene.objects:
            if obj in task.lower() and 'block' in obj:
                pos = scene.get_position(obj)
                return cls._pick_and_place(pos, None, obj, None)
        return []
    
    @classmethod
    def _generate_stack(cls, task: str, scene: SceneGraph) -> List[Dict]:
        blocks = scene.get_objects_by_type('block')
        source, target = None, None
        for block in blocks:
            color = block.replace('_block', '')
            if color in task.lower():
                if source is None:
                    source = block
                else:
                    target = block
        if source and target:
            return cls._pick_and_place(scene.get_position(source), scene.get_position(target), source, target)
        return []
    
    @classmethod
    def _generate_multi_place(cls, task: str, scene: SceneGraph) -> List[Dict]:
        blocks = scene.get_objects_by_type('block')
        box_pos = scene.get_position('box') if 'box' in scene.objects else None
        if not box_pos:
            return []
        all_waypoints = []
        for i, block in enumerate(blocks):
            block_pos = scene.get_position(block)
            wps = cls._pick_and_place(block_pos, box_pos, block, 'box', target_height_offset=i * cls.BLOCK_HEIGHT)
            all_waypoints.extend(wps)
        return all_waypoints
    
    @classmethod
    def _generate_spatial(cls, task: str, scene: SceneGraph) -> List[Dict]:
        blocks = scene.get_objects_by_type('block')
        source, reference = None, None
        for block in blocks:
            color = block.replace('_block', '')
            if color in task.lower():
                if source is None:
                    source = block
                else:
                    reference = block
        if source and reference:
            src_pos = scene.get_position(source)
            ref_pos = scene.get_position(reference)
            offset = -0.1 if 'left' in task.lower() else 0.1
            target_pos = [ref_pos[0] + offset, ref_pos[1], ref_pos[2]]
            return cls._pick_and_place(src_pos, target_pos, source, f'left of {reference}')
        return []
    
    @classmethod
    def _generate_tower(cls, task: str, scene: SceneGraph) -> List[Dict]:
        blocks = scene.get_objects_by_type('block')
        if len(blocks) < 2:
            return []
        all_waypoints = []
        base = blocks[0]
        base_pos = scene.get_position(base)
        for i, block in enumerate(blocks[1:], start=1):
            block_pos = scene.get_position(block)
            target_pos = [base_pos[0], base_pos[1], base_pos[2] + i * cls.BLOCK_HEIGHT]
            wps = cls._pick_and_place(block_pos, target_pos, block, f'tower level {i}')
            all_waypoints.extend(wps)
        return all_waypoints
    
    @classmethod
    def _generate_generic(cls, task: str, scene: SceneGraph) -> List[Dict]:
        blocks = scene.get_objects_by_type('block')
        if blocks:
            pos = scene.get_position(blocks[0])
            return cls._pick_and_place(pos, None, blocks[0], None)
        return []


class CombinedPlanner:
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        if not HAS_GENAI:
            raise ImportError("google-generativeai package required")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.verifier = GroundingVerifier()
    
    def plan(self, task: str, scene: SceneGraph) -> PlanResult:
        result = PlanResult(task=task)
        result.task_type = WaypointGenerator.detect_task_type(task)
        
        grounding = self.verifier.verify_task_grounding(task, scene.to_dict())
        result.grounding_success = grounding.success
        result.grounding_message = grounding.message
        result.inferred_relations = [r.to_dict() for r in grounding.inferred_relations]
        
        if not grounding.success:
            return result
        
        result.reasoning = self._decompose_task(task)
        result.scene_graph = scene.to_dict()
        result.waypoints = WaypointGenerator.generate(task, scene)
        result.constraints = self._generate_constraints(task, scene)
        
        return result
    
    def _decompose_task(self, task: str) -> List[str]:
        prompt = COT_PROMPT.format(task=task)
        response = self.model.generate_content(prompt)
        steps = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line.startswith('Step '):
                step_text = re.sub(r'^Step \d+:\s*', '', line)
                steps.append(step_text)
        return steps
    
    def _generate_constraints(self, task: str, scene: SceneGraph) -> str:
        prompt = CONSTRAINT_PROMPT.format(task=task, objects=json.dumps(scene.to_dict(), indent=2))
        response = self.model.generate_content(prompt)
        code_match = re.search(r'```python\s*(.+?)```', response.text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response.text[:500]
