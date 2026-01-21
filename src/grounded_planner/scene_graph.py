"""SceneGraph: Data structure for 3D object positions with confidence tracking."""

from typing import List, Dict, Optional


class SceneGraph:
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.objects: Dict[str, Dict] = {}
    
    def add_object(self, name: str, pos: List[float], relations: Optional[List[str]] = None, confidence: float = 1.0) -> None:
        self.objects[name] = {
            'pos': pos.copy() if isinstance(pos, list) else list(pos),
            'relations': relations or [],
            'confidence': confidence,
        }
    
    def get_position(self, name: str) -> List[float]:
        return self.objects.get(name, {}).get('pos', [0, 0, 0])
    
    def get_confidence(self, name: str) -> float:
        return self.objects.get(name, {}).get('confidence', 1.0)
    
    def get_objects_by_type(self, obj_type: str) -> List[str]:
        return [name for name in self.objects.keys() if obj_type in name]
    
    def get_low_confidence_objects(self, threshold: float = 0.7) -> List[str]:
        return [name for name, data in self.objects.items() if data.get('confidence', 1.0) < threshold]
    
    def to_dict(self) -> Dict:
        return self.objects.copy()
    
    def __repr__(self) -> str:
        lines = [f"SceneGraph: {self.name}"]
        for name, data in self.objects.items():
            pos = data['pos']
            conf = data.get('confidence', 1.0)
            lines.append(f"  {name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] (conf={conf:.2f})")
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.objects)
