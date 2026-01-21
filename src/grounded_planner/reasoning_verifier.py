"""
Reasoning Verifier - Look-Ahead action validity checker.
Ported from Language Planner to provide pre-execution verification.

Checks action validity BEFORE execution using:
1. Rule-based precondition checking (fast, deterministic)
2. Prerequisite suggestion for failed actions
"""

from typing import Optional, Tuple, List, Dict
import re


class StateTracker:
    """
    Lightweight state tracker for action verification.
    Tracks: location, hands (held objects), object states.
    """
    
    GRAB_VERBS = {'grab', 'grasp', 'pick', 'take', 'get'}
    RELEASE_VERBS = {'put', 'place', 'drop', 'release'}
    OPEN_VERBS = {'open'}
    CLOSE_VERBS = {'close'}
    MOVEMENT_VERBS = {'walk', 'go', 'navigate', 'move'}
    
    def __init__(self, initial_location: str = "table"):
        self.state = {
            'location': initial_location,
            'hands': [],  # Objects currently held
            'object_states': {},  # {object: {'state': 'open'/'closed'}}
        }
    
    def update(self, action: str) -> None:
        """Update state based on executed action."""
        action_lower = action.lower().strip()
        parts = action_lower.split()
        
        if not parts:
            return
        
        verb = parts[0]
        obj = parts[1] if len(parts) > 1 else None
        
        # Movement
        if verb in self.MOVEMENT_VERBS and obj:
            self.state['location'] = obj
        
        # Grabbing
        elif verb in self.GRAB_VERBS and obj:
            if obj not in self.state['hands']:
                self.state['hands'].append(obj)
        
        # Releasing
        elif verb in self.RELEASE_VERBS and obj:
            if obj in self.state['hands']:
                self.state['hands'].remove(obj)
            elif self.state['hands']:
                self.state['hands'].pop()  # Release whatever we're holding
        
        # Opening
        elif verb in self.OPEN_VERBS and obj:
            self.state['object_states'][obj] = {'state': 'open'}
        
        # Closing
        elif verb in self.CLOSE_VERBS and obj:
            self.state['object_states'][obj] = {'state': 'closed'}
    
    def hands_full(self) -> bool:
        return len(self.state['hands']) >= 2
    
    def is_holding(self, obj: str) -> bool:
        return obj in self.state['hands']
    
    def is_object_open(self, obj: str) -> Optional[bool]:
        obj_state = self.state['object_states'].get(obj)
        if obj_state:
            return obj_state.get('state') == 'open'
        return None
    
    def get_object_state(self, obj: str) -> Optional[Dict]:
        return self.state['object_states'].get(obj)
    
    def get_state_description(self) -> str:
        return f"Location: {self.state['location']}, Holding: {self.state['hands']}, Objects: {self.state['object_states']}"


class ReasoningVerifier:
    """
    Chain-of-Thought verifier that checks action validity BEFORE execution.
    
    Provides:
    - Rule-based precondition checking
    - Prerequisite suggestions for failed actions
    - Look-ahead verification for plan steps
    """
    
    # Containers that must be opened before grabbing contents
    CONTAINERS = {
        'fridge', 'refrigerator', 'cabinet', 'drawer', 'closet',
        'box', 'microwave', 'oven', 'dishwasher'
    }
    
    # Objects typically found in containers
    CONTAINER_CONTENTS = {
        'fridge': ['egg', 'milk', 'juice', 'butter', 'cheese', 'apple', 'food'],
        'refrigerator': ['egg', 'milk', 'juice', 'butter', 'cheese', 'apple', 'food'],
        'cabinet': ['plate', 'glass', 'cup', 'bowl', 'mug'],
        'drawer': ['fork', 'knife', 'spoon', 'utensil'],
    }
    
    # Location requirements for objects
    OBJECT_LOCATIONS = {
        'fridge': 'kitchen',
        'stove': 'kitchen',
        'microwave': 'kitchen',
        'sink': 'kitchen',
        'bed': 'bedroom',
        'toilet': 'bathroom',
        'tv': 'living_room',
    }
    
    def __init__(self):
        pass
    
    def verify(self, action: str, state: StateTracker) -> Tuple[bool, str]:
        """
        Verify if an action is valid given the current state.
        
        Returns:
            tuple: (is_valid, reason)
        """
        verb, objects = self._parse_action(action)
        
        if not verb:
            return True, ""
        
        return self._rule_based_check(verb, objects, state)
    
    def _parse_action(self, action: str) -> Tuple[Optional[str], List[str]]:
        """Parse action string into verb and objects."""
        action_lower = action.lower().strip()
        parts = action_lower.split()
        
        if parts:
            return parts[0], parts[1:] if len(parts) > 1 else []
        
        return None, []
    
    def _rule_based_check(self, verb: str, objects: List[str], state: StateTracker) -> Tuple[bool, str]:
        """
        Fast rule-based precondition checking.
        """
        # Check grab actions
        if verb in StateTracker.GRAB_VERBS and objects:
            obj = objects[0]
            
            # Check if hands are full
            if state.hands_full():
                return False, f"Cannot grab {obj}: hands are full (holding {', '.join(state.state['hands'])})"
            
            # Check if object is in a container that needs to be opened
            for container, contents in self.CONTAINER_CONTENTS.items():
                if obj in contents:
                    if not state.is_object_open(container):
                        return False, f"Cannot grab {obj}: need to open {container} first"
                    break
        
        # Check open actions
        if verb in StateTracker.OPEN_VERBS and objects:
            obj = objects[0]
            
            # Check if already open
            if state.is_object_open(obj) is True:
                return False, f"The {obj} is already open"
        
        # Check close actions
        if verb in StateTracker.CLOSE_VERBS and objects:
            obj = objects[0]
            
            obj_state = state.get_object_state(obj)
            if obj_state and obj_state.get('state') == 'closed':
                return False, f"The {obj} is already closed"
        
        # Check put/place actions
        if verb in StateTracker.RELEASE_VERBS and objects:
            obj = objects[0]
            
            if not state.is_holding(obj) and not state.state['hands']:
                return False, f"Cannot put {obj}: not holding anything"
        
        return True, ""
    
    def get_prerequisite_suggestion(self, action: str, failure_reason: str, state: StateTracker) -> Optional[str]:
        """
        Suggest a prerequisite action to fix the failure.
        """
        reason_lower = failure_reason.lower()
        
        # Need to open container
        if "need to open" in reason_lower:
            match = re.search(r'need to open (\w+)', reason_lower)
            if match:
                return f"open {match.group(1)}"
        
        # Hands are full
        if "hands are full" in reason_lower:
            if state.state['hands']:
                obj = state.state['hands'][0]
                return f"put {obj}"
        
        # Not holding object
        if "not holding" in reason_lower:
            return None  # Need to grab first
        
        return None
    
    def verify_plan(self, plan_steps: List[str]) -> List[Dict]:
        """
        Verify a complete plan, checking each step.
        
        Returns:
            List of verification results for each step.
        """
        state = StateTracker()
        results = []
        
        for step in plan_steps:
            is_valid, reason = self.verify(step, state)
            
            result = {
                'step': step,
                'valid': is_valid,
                'reason': reason,
            }
            
            if not is_valid:
                fix = self.get_prerequisite_suggestion(step, reason, state)
                result['suggested_fix'] = fix
            else:
                state.update(step)
            
            results.append(result)
        
        return results


if __name__ == "__main__":
    print("=" * 60)
    print("Reasoning Verifier Test")
    print("=" * 60)
    
    tracker = StateTracker()
    verifier = ReasoningVerifier()
    
    test_plan = [
        "grab egg",           # Should fail: fridge not open
        "open fridge",        # Should pass
        "grab egg",           # Should pass now
        "open fridge",        # Should fail: already open
        "close fridge",       # Should pass
    ]
    
    print("\nTest Plan Verification:")
    results = verifier.verify_plan(test_plan)
    
    for r in results:
        status = "✓ VALID" if r['valid'] else f"✗ INVALID: {r['reason']}"
        print(f"\n  Action: {r['step']}")
        print(f"  Result: {status}")
        if r.get('suggested_fix'):
            print(f"  Fix: {r['suggested_fix']}")
