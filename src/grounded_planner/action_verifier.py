"""
Action Verifier: Pre-Execution Precondition Checking.
Basically, It verifies whether an action can be executed based on the current state before executing waypoints.
"""

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    is_valid: bool
    reason: str = ""
    prerequisite: str = ""


class ActionVerifier:
    """
    This class will verifies actions and their preconditions before execution.
    """
    
    # Action preconditions: {action: {requirement: check}}
    PRECONDITIONS = {
        'grasp': ['gripper_empty', 'object_reachable'],
        'release': ['gripper_holding'],
        'approach': ['position_valid'],
        'lift': ['gripper_holding'],
        'descend': ['position_valid'],
        'move': ['gripper_holding'],
    }
    
    def __init__(self):
        self.gripper_holding: Optional[str] = None
        self.gripper_pos: List[float] = [0, 0, 0.15]
    
    def verify_action(self, action: str, target: str, scene_objects: Dict) -> VerificationResult:
        """
        This function will verify if action can be executed based on current state.
        
        Args:
            action: Action type (grasp, release, etc.)
            target: Target object name
            scene_objects: Scene graph objects dict
            
        Returns:
            VerificationResult with validity and reason
        """
        action_lower = action.lower()
        
        if action_lower == 'grasp':
            if self.gripper_holding is not None:
                return VerificationResult(
                    is_valid=False,
                    reason=f"Gripper already holding {self.gripper_holding}",
                    prerequisite=f"release {self.gripper_holding}"
                )
            if target and target not in scene_objects:
                return VerificationResult(
                    is_valid=False,
                    reason=f"Object {target} not in scene",
                    prerequisite=""
                )
        
        if action_lower == 'release':
            if self.gripper_holding is None:
                return VerificationResult(
                    is_valid=False,
                    reason="Gripper not holding anything",
                    prerequisite=""
                )
        
        if action_lower in ['lift', 'move']:
            if self.gripper_holding is None:
                return VerificationResult(
                    is_valid=False,
                    reason="Cannot lift/move without holding object",
                    prerequisite="grasp <object> first"
                )
        
        return VerificationResult(is_valid=True)
    
    def update_state(self, action: str, target: str = "") -> None:
        """Once an action is executed, this function will update the internal state."""
        action_lower = action.lower()
        
        if action_lower == 'grasp':
            self.gripper_holding = target
        elif action_lower == 'release':
            self.gripper_holding = None
    
    def verify_waypoint_sequence(
        self, 
        waypoints: List[Dict], 
        scene_objects: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Verifies entire waypoint sequence.
        """
        issues = []
        
        for i, wp in enumerate(waypoints):
            action = wp.get('action', '')
            label = wp.get('label', '')
            
            # Extract target from label
            target = ""
            if 'grasp' in label or 'at' in label:
                parts = label.split()
                if len(parts) > 1:
                    target = parts[-1]
            
            result = self.verify_action(action, target, scene_objects)
            
            if not result.is_valid:
                issues.append(f"Step {i+1} ({action}): {result.reason}")
            else:
                self.update_state(action, target)
        
        return len(issues) == 0, issues
    
    def reset(self) -> None:
        self.gripper_holding = None
        self.gripper_pos = [0, 0, 0.15]


if __name__ == "__main__":
    print("=" * 60)
    print("Action Verifier Test")
    print("=" * 60)
    
    verifier = ActionVerifier()
    scene = {'red_block': {'pos': [0.3, 0.2, 0.025]}}
    
    print("\nTest 1: Grasp red_block")
    result = verifier.verify_action("grasp", "red_block", scene)
    print(f"  Valid: {result.is_valid}")
    verifier.update_state("grasp", "red_block")
    
    print("\nTest 2: Grasp again (should fail)")
    result = verifier.verify_action("grasp", "blue_block", scene)
    print(f"  Valid: {result.is_valid}")
    print(f"  Reason: {result.reason}")
    
    print("\nTest 3: Release")
    result = verifier.verify_action("release", "", scene)
    print(f"  Valid: {result.is_valid}")
