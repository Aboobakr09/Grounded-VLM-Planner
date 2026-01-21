"""
Common Sense Rules Engine for Safe Robot Execution which provides action-level safety verification complementing grounding verification.
"""

from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class SafetyCheck:
    is_safe: bool
    warning: str = ""
    suggested_fix: str = ""


class CommonSenseRules:
    """
    This enforces common-sense rules for safe action execution. like:    
    1. Unsafe combinations (e.g., microwave + metal)
    2. Prerequisite suggestions (e.g., open before grab)
    3. Cleanup tracking (e.g., close containers after use)
    """
    
    # Containers that should be closed after use
    CONTAINERS = {'fridge', 'cabinet', 'drawer', 'box', 'microwave', 'oven'}
    
    # Appliances that should be turned off
    APPLIANCES = {'stove', 'microwave', 'oven'}
    
    # Unsafe action-object combinations
    UNSAFE_COMBINATIONS = {
        ('microwave', 'metal'): "Don't microwave metal objects",
        ('microwave', 'egg'): "Don't microwave eggs in shell",
        ('oven', 'plastic'): "Don't put plastic in oven",
        ('stack', 'liquid'): "Can't stack liquids",
        ('grasp', 'hot'): "Don't grasp hot objects without protection",
    }
    
    PREREQUISITES = {
        'grab': {'container': 'open'},      
        'place': {'container': 'open'},     
        'cook': {'appliance': 'turn_on'},   
    }
    
    def __init__(self):
        self.opened_containers: Set[str] = set()
        self.active_appliances: Set[str] = set()
        self.held_objects: Set[str] = set()
    
    def check_action_safety(self, action: str, context: Dict = None) -> SafetyCheck:
        action_lower = action.lower()
        
        # Check unsafe combinations
        for (target, dangerous), warning in self.UNSAFE_COMBINATIONS.items():
            if target in action_lower and dangerous in action_lower:
                return SafetyCheck(
                    is_safe=False,
                    warning=warning,
                    suggested_fix=f"Remove {dangerous} before {target}"
                )
        
        # Check if trying to grab from closed container
        if 'grab' in action_lower or 'grasp' in action_lower:
            for container in self.CONTAINERS:
                if container in action_lower and container not in self.opened_containers:
                    return SafetyCheck(
                        is_safe=False,
                        warning=f"Cannot grab from closed {container}",
                        suggested_fix=f"open {container}"
                    )
        
        return SafetyCheck(is_safe=True)
    
    def track_action(self, action: str) -> None:
        action_lower = action.lower()
        
        # Track opened/closed containers and active/inactive appliances
        if 'open' in action_lower:
            for container in self.CONTAINERS:
                if container in action_lower:
                    self.opened_containers.add(container)
        elif 'close' in action_lower:
            for container in self.CONTAINERS:
                if container in action_lower:
                    self.opened_containers.discard(container)
        
        if 'turn_on' in action_lower or 'turnon' in action_lower:
            for appliance in self.APPLIANCES:
                if appliance in action_lower:
                    self.active_appliances.add(appliance)
        elif 'turn_off' in action_lower or 'turnoff' in action_lower:
            for appliance in self.APPLIANCES:
                if appliance in action_lower:
                    self.active_appliances.discard(appliance)
        
        if 'grasp' in action_lower or 'grab' in action_lower:
            words = action_lower.split()
            if len(words) > 1:
                self.held_objects.add(words[-1])
        elif 'release' in action_lower or 'place' in action_lower:
            self.held_objects.clear()
    
    def get_cleanup_actions(self) -> List[str]:
        #Get cleanup actions to restore safe state.
        cleanup = []
        
        for container in sorted(self.opened_containers):
            cleanup.append(f"close {container}")
        
        for appliance in sorted(self.active_appliances):
            cleanup.append(f"turn_off {appliance}")
        
        return cleanup
    
    def suggest_prerequisite(self, action: str) -> str:
        action_lower = action.lower()
        
        if 'grab' in action_lower or 'grasp' in action_lower:
            for container in self.CONTAINERS:
                if container in action_lower and container not in self.opened_containers:
                    return f"open {container}"
        
        return ""
    
    def reset(self) -> None:
        self.opened_containers.clear()
        self.active_appliances.clear()
        self.held_objects.clear()


def apply_common_sense(waypoints: List[Dict], rules: CommonSenseRules = None) -> Tuple[List[Dict], List[str]]:
    if rules is None:
        rules = CommonSenseRules()
    
    filtered = []
    warnings = []
    
    for wp in waypoints:
        action = wp.get('action', '')
        label = wp.get('label', '')
        
        check = rules.check_action_safety(f"{action} {label}")
        
        if not check.is_safe:
            warnings.append(f"{action}: {check.warning}")
            if check.suggested_fix:
                warnings.append(f"   Fix: {check.suggested_fix}")
            continue
        
        filtered.append(wp)
        rules.track_action(f"{action} {label}")
    
    cleanup = rules.get_cleanup_actions()
    for cleanup_action in cleanup:
        filtered.append({
            'pos': [0, 0, 0.15],
            'action': 'cleanup',
            'label': cleanup_action
        })
        warnings.append(f"Added cleanup: {cleanup_action}")
    
    return filtered, warnings


if __name__ == "__main__":
    print("-" * 60)
    print("Common Sense Rules Engine Test")
    print("-" * 60)
    
    rules = CommonSenseRules()
    
    print("\nTest 1: Safe grasp")
    check = rules.check_action_safety("grasp red_block")
    print(f"  Result: {'Safe' if check.is_safe else 'Unsafe'}")
    
    print("\nTest 2: Microwave metal")
    check = rules.check_action_safety("put metal_fork in microwave")
    print(f"  Result: {'Safe' if check.is_safe else 'Unsafe'}")
    print(f"  Warning: {check.warning}")
    
    print("\nTest 3: Track container state")
    rules.track_action("open fridge")
    rules.track_action("grab milk")
    print(f"  Open containers: {rules.opened_containers}")
    print(f"  Cleanup needed: {rules.get_cleanup_actions()}")
