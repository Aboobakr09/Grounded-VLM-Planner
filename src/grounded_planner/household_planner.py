"""
Household Waypoint Generator - Extended for navigation and articulation
"""

from typing import List, Dict, Optional, Tuple
from grounded_planner.scene_graph import SceneGraph


class HouseholdWaypointGenerator:
    """
    Extended waypoint generator for household manipulation tasks.
    Supports navigation, articulation, and multi-location interactions.
    """
    
    # Physical constants
    BLOCK_HEIGHT = 0.05
    APPROACH_HEIGHT = 0.10
    LIFT_HEIGHT = 0.15
    
    # Household locations (default positions)
    LOCATIONS = {
        'table': [0.0, 0.0, 0.0],
        'fridge': [2.0, 0.0, 0.0],
        'cabinet': [-1.0, 1.5, 1.0],
        'counter': [1.0, -1.0, 0.0],
        'sink': [1.5, -1.0, 0.0],
    }
    
    @staticmethod
    def detect_task_type(task: str) -> str:
        """Detect task type including household actions."""
        task_lower = task.lower()
        
        # Household-level tasks
        if 'from fridge' in task_lower or 'in fridge' in task_lower:
            return 'fridge_interaction'
        if 'from cabinet' in task_lower or 'in cabinet' in task_lower:
            return 'cabinet_interaction'
        if 'walk to' in task_lower or 'go to' in task_lower:
            return 'navigation'
        if 'open' in task_lower and ('door' in task_lower or 'fridge' in task_lower or 'cabinet' in task_lower):
            return 'open_articulated'
        if 'close' in task_lower and ('door' in task_lower or 'fridge' in task_lower or 'cabinet' in task_lower):
            return 'close_articulated'
        
        # Tabletop tasks (original)
        if 'pick up' in task_lower or 'pick' in task_lower:
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
    def generate(cls, task: str, scene: SceneGraph, current_location: str = 'table') -> List[Dict]:
        """
        Generate waypoints for household tasks.
        
        Args:
            task: Task description
            scene: Scene graph
            current_location: Current robot location
        """
        task_type = cls.detect_task_type(task)
        
        generators = {
            # Household tasks
            'navigation': cls._generate_navigation,
            'fridge_interaction': cls._generate_fridge_task,
            'cabinet_interaction': cls._generate_cabinet_task,
            'open_articulated': cls._generate_open,
            'close_articulated': cls._generate_close,
            # Tabletop tasks
            'pick': cls._generate_pick,
            'stack': cls._generate_stack,
            'multi_place': cls._generate_multi_place,
            'spatial': cls._generate_spatial,
            'tower': cls._generate_tower,
        }
        
        generator = generators.get(task_type, cls._generate_generic)
        return generator(task, scene, current_location)
    
    # Household Primitives
    
    @classmethod
    def _generate_navigation(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
        """Navigate to a location."""
        waypoints = []
        
        # Extract target location
        target = None
        for loc_name in cls.LOCATIONS.keys():
            if loc_name in task.lower():
                target = loc_name
                break
        
        if not target:
            return []
        
        target_pos = cls.LOCATIONS[target]
        
        # Generate navigation waypoint
        waypoints.append({
            'pos': target_pos,
            'action': 'navigate',
            'label': f'navigate to {target}',
            'location': target
        })
        
        return waypoints
    
    @classmethod
    def _generate_open(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
        """Open articulated object (fridge, cabinet, door)."""
        waypoints = []
        
        # Determine what to open
        target = None
        if 'fridge' in task.lower():
            target = 'fridge'
        elif 'cabinet' in task.lower():
            target = 'cabinet'
        elif 'door' in task.lower():
            target = 'door'
        
        if not target:
            return []
        
        # Navigate if not there
        if current_location != target:
            waypoints.extend(cls._generate_navigation(f"walk to {target}", scene, current_location))
        
        # Approach handle
        handle_pos = cls.LOCATIONS.get(target, [0, 0, 0.5])
        waypoints.append({
            'pos': handle_pos,
            'action': 'approach_handle',
            'label': f'approach {target} handle'
        })
        
        # Grasp handle
        waypoints.append({
            'pos': handle_pos,
            'action': 'grasp_handle',
            'label': f'grasp {target} handle'
        })
        
        # Pull open (trajectory)
        open_pos = [handle_pos[0] + 0.3, handle_pos[1], handle_pos[2]]
        waypoints.append({
            'pos': open_pos,
            'action': 'pull_open',
            'label': f'open {target}',
            'articulated_object': target
        })
        
        return waypoints
    
    @classmethod
    def _generate_close(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
        """Close articulated object."""
        waypoints = []
        
        # Determine what to close
        target = None
        if 'fridge' in task.lower():
            target = 'fridge'
        elif 'cabinet' in task.lower():
            target = 'cabinet'
        elif 'door' in task.lower():
            target = 'door'
        
        if not target:
            return []
        
        # Approach handle
        handle_pos = cls.LOCATIONS.get(target, [0, 0, 0.5])
        waypoints.append({
            'pos': handle_pos,
            'action': 'approach_handle',
            'label': f'approach {target} handle'
        })
        
        # Push close
        closed_pos = [handle_pos[0] - 0.3, handle_pos[1], handle_pos[2]]
        waypoints.append({
            'pos': closed_pos,
            'action': 'push_close',
            'label': f'close {target}',
            'articulated_object': target
        })
        
        return waypoints
    
    @classmethod
    def _generate_fridge_task(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
        """
        Complete fridge interaction: navigate, open, look, grasp, close.
        Example: "get milk from fridge"
        """
        waypoints = []
        
        # Step 1: Navigate to fridge
        if current_location != 'fridge':
            waypoints.extend(cls._generate_navigation("walk to fridge", scene, current_location))
        
        # Step 2: Open fridge
        waypoints.extend(cls._generate_open("open fridge", scene, 'fridge'))
        
        # Step 3: Look inside (active perception)
        fridge_interior = cls.LOCATIONS['fridge']
        waypoints.append({
            'pos': [fridge_interior[0] + 0.5, fridge_interior[1], fridge_interior[2] + 0.5],
            'action': 'look_inside',
            'label': 'look inside fridge',
            'perception_required': True
        })
        
        # Step 4: Identify and grasp object
        # Extract object name from task
        object_name = None
        # Extract object name from task by checking scene objects
        object_name = None
        # Sort objects by length descending to match longest valid name first (e.g. "red apple" vs "apple")
        candidate_objects = sorted(scene.objects.keys(), key=len, reverse=True)
        for obj in candidate_objects:
            if obj in task.lower():
                object_name = obj
                break
        
        if object_name and object_name in scene.objects:
            obj_pos = scene.get_position(object_name)
            waypoints.append({
                'pos': obj_pos,
                'action': 'approach',
                'label': f'approach {object_name}'
            })
            waypoints.append({
                'pos': obj_pos,
                'action': 'grasp',
                'label': f'grasp {object_name}'
            })
            waypoints.append({
                'pos': [obj_pos[0], obj_pos[1], obj_pos[2] + cls.LIFT_HEIGHT],
                'action': 'lift',
                'label': f'lift {object_name}'
            })
        
        # Step 5: Close fridge (cleanup)
        waypoints.extend(cls._generate_close("close fridge", scene, 'fridge'))
        
        return waypoints
    
    @classmethod
    def _generate_cabinet_task(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
        """
        Cabinet interaction: navigate, open, grasp, close.
        Example: "get plate from cabinet"
        """
        waypoints = []
        
        # Navigate
        if current_location != 'cabinet':
            waypoints.extend(cls._generate_navigation("walk to cabinet", scene, current_location))
        
        # Open
        waypoints.extend(cls._generate_open("open cabinet", scene, 'cabinet'))
        
        # Look
        waypoints.append({
            'pos': cls.LOCATIONS['cabinet'],
            'action': 'look_inside',
            'label': 'look inside cabinet',
            'perception_required': True
        })
        
        # Grasp object if visible
        for obj_name in scene.objects:
            if obj_name in task.lower():
                obj_pos = scene.get_position(obj_name)
                waypoints.append({'pos': obj_pos, 'action': 'grasp', 'label': f'grasp {obj_name}'})
                break
        
        # Close
        waypoints.extend(cls._generate_close("close cabinet", scene, 'cabinet'))
        
        return waypoints
    
    # Tabletop Primitives (Original)
    
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
    def _generate_pick(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
        candidate_objects = sorted(scene.objects.keys(), key=len, reverse=True)
        for obj in candidate_objects:
            # Skip fixed furniture/locations
            if obj in cls.LOCATIONS:
                continue
            if obj in task.lower():
                pos = scene.get_position(obj)
                return cls._pick_and_place(pos, None, obj, None)
        return []
    
    @classmethod
    def _generate_stack(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
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
    def _generate_multi_place(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
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
    def _generate_spatial(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
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
    def _generate_tower(cls, task: str, scene: SceneGraph, current_location: str) ->List[Dict]:
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
    def _generate_generic(cls, task: str, scene: SceneGraph, current_location: str) -> List[Dict]:
        blocks = scene.get_objects_by_type('block')
        if blocks:
            pos = scene.get_position(blocks[0])
            return cls._pick_and_place(pos, None, blocks[0], None)
        return []
