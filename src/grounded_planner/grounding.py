"""Grounding Verification: Task-Scene Consistency with Uncertainty."""

from dataclasses import dataclass, field
from typing import List, Dict, Set
import math


@dataclass
class TypedRelation:
    type: str
    subject: str
    object: str
    confidence: float = 1.0
    
    def __repr__(self) -> str:
        return f"{self.type}({self.subject}, {self.object})"
    
    def to_dict(self) -> Dict:
        return {"type": self.type, "subject": self.subject, "object": self.object, "confidence": self.confidence}


@dataclass
class GroundingResult:
    success: bool
    missing_objects: List[str] = field(default_factory=list)
    low_confidence_objects: List[Dict] = field(default_factory=list)
    violated_relations: List[str] = field(default_factory=list)
    inferred_relations: List[TypedRelation] = field(default_factory=list)
    message: str = ""
    confidence_score: float = 1.0
    
    def __repr__(self) -> str:
        if self.success:
            return f"GroundingResult(success=True, confidence={self.confidence_score:.2f})"
        return f"GroundingResult(success=False, reason='{self.message}')"


class GroundingVerifier:
    
    CONFIDENCE_THRESHOLD = 0.6
    SPATIAL_THRESHOLD = 0.05
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
    
    def extract_required_objects(self, task: str) -> Set[str]:
        task_lower = task.lower()
        required = set()
        
        if 'red' in task_lower:
            required.add('red_block')
        if 'blue' in task_lower:
            required.add('blue_block')
        if 'green' in task_lower:
            required.add('green_block')
        if 'yellow' in task_lower:
            required.add('yellow_block')
        if 'box' in task_lower:
            required.add('box')
        
        if 'all blocks' in task_lower or ('all' in task_lower and 'block' in task_lower):
            required.update(['red_block', 'blue_block', 'green_block'])
        
        if 'tower' in task_lower or '3 blocks' in task_lower:
            required.update(['red_block', 'blue_block', 'green_block'])
        
        return required
    
    def compute_spatial_consistency(self, obj_data: Dict, all_objects: Dict) -> float:
        pos = obj_data.get('pos', [0, 0, 0])
        score = 1.0
        
        if pos[2] < 0:
            score *= 0.5
        if pos[2] > 1.5:  # Raised for household (fridge shelf, cabinet)
            score *= 0.7
        
        if abs(pos[0]) > 0.5 or abs(pos[1]) > 0.5:
            score *= 0.8
        
        for other_name, other_data in all_objects.items():
            if other_name == obj_data.get('name'):
                continue
            other_pos = other_data.get('pos', [0, 0, 0])
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos[:2], other_pos[:2])))
            if dist < 0.02:
                score *= 0.3
        
        return max(0.0, min(1.0, score))
    
    def compute_combined_confidence(self, obj_data: Dict, all_objects: Dict) -> float:
        vlm_confidence = obj_data.get('confidence', 0.8)
        spatial_score = self.compute_spatial_consistency(obj_data, all_objects)
        return min(vlm_confidence, spatial_score)
    
    def infer_spatial_relations(self, scene_objects: Dict) -> List[TypedRelation]:
        relations = []
        objects = {k: v for k, v in scene_objects.items() if k not in ['table', 'gripper']}
        
        for obj1_name, obj1_data in objects.items():
            pos1 = obj1_data.get('pos', [0, 0, 0])
            
            relations.append(TypedRelation(type="on", subject=obj1_name, object="table",
                confidence=1.0 if abs(pos1[2] - 0.025) < 0.01 else 0.7))
            
            for obj2_name, obj2_data in objects.items():
                if obj1_name == obj2_name:
                    continue
                
                pos2 = obj2_data.get('pos', [0, 0, 0])
                
                if pos1[0] < pos2[0] - self.SPATIAL_THRESHOLD:
                    relations.append(TypedRelation(type="left_of", subject=obj1_name, object=obj2_name,
                        confidence=min(1.0, abs(pos1[0] - pos2[0]) / 0.2)))
                
                if pos1[0] > pos2[0] + self.SPATIAL_THRESHOLD:
                    relations.append(TypedRelation(type="right_of", subject=obj1_name, object=obj2_name,
                        confidence=min(1.0, abs(pos1[0] - pos2[0]) / 0.2)))
                
                xy_dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if xy_dist < 0.08 and pos1[2] > pos2[2] + 0.03:
                    relations.append(TypedRelation(type="stacked_on", subject=obj1_name, object=obj2_name,
                        confidence=0.9 if xy_dist < 0.03 else 0.6))
                
                dist_3d = math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
                if dist_3d < 0.15:
                    relations.append(TypedRelation(type="near", subject=obj1_name, object=obj2_name,
                        confidence=1.0 - (dist_3d / 0.15)))
        
        return relations
    
    def verify_task_grounding(self, task: str, scene_objects: Dict) -> GroundingResult:
        required = self.extract_required_objects(task)
        detected = set(k for k in scene_objects.keys() if k not in ['table', 'gripper'])
        
        missing = required - detected
        
        low_confidence = []
        for obj_name, obj_data in scene_objects.items():
            if obj_name in ['table', 'gripper']:
                continue
            
            obj_data_with_name = {**obj_data, 'name': obj_name}
            combined_conf = self.compute_combined_confidence(obj_data_with_name, scene_objects)
            
            if combined_conf < self.confidence_threshold:
                low_confidence.append({
                    'name': obj_name,
                    'vlm_confidence': obj_data.get('confidence', 0.8),
                    'spatial_consistency': self.compute_spatial_consistency(obj_data_with_name, scene_objects),
                    'combined': combined_conf
                })
        
        relations = self.infer_spatial_relations(scene_objects)
        
        violated = []
        if 'left' in task.lower():
            left_rels = [r for r in relations if r.type == 'left_of']
            if not left_rels:
                violated.append("No left_of relations detected for spatial task")
        
        success = len(missing) == 0 and len(low_confidence) == 0 and len(violated) == 0
        
        if missing:
            message = f"Missing objects: {missing}"
        elif low_confidence:
            message = f"Low confidence: {[o['name'] for o in low_confidence]}"
        elif violated:
            message = "; ".join(violated)
        else:
            message = "Grounding verified successfully"
        
        avg_confidence = 1.0
        if detected:
            confidences = []
            for obj_name in detected:
                obj_data = scene_objects[obj_name]
                obj_data_with_name = {**obj_data, 'name': obj_name}
                confidences.append(self.compute_combined_confidence(obj_data_with_name, scene_objects))
            avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        
        return GroundingResult(
            success=success,
            missing_objects=list(missing),
            low_confidence_objects=low_confidence,
            violated_relations=violated,
            inferred_relations=relations,
            message=message,
            confidence_score=avg_confidence
        )
