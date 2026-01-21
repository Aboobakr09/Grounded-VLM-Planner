"""
demo usage:
    python demo_full_pipeline.py "pick up the red block"
    python demo_full_pipeline.py "stack red block on blue block"
    python demo_full_pipeline.py "grab milk from fridge" --household
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from grounded_planner.scene_graph import SceneGraph
from grounded_planner.grounding import GroundingVerifier
from grounded_planner.planner import WaypointGenerator
from grounded_planner.household_planner import HouseholdWaypointGenerator

try:
    from grounded_planner.extended_planner import ExtendedPlanner
    HAS_EXTENDED_PLANNER = True
except Exception as e:
    HAS_EXTENDED_PLANNER = False
    print(f"ExtendedPlanner unavailable: {e}")

try:
    from grounded_planner.action_translator import ActionTranslator
    HAS_TRANSLATOR = True
except Exception as e:
    HAS_TRANSLATOR = False
    print(f"ActionTranslator unavailable: {e}")

def create_tabletop_scene() -> SceneGraph:
    """Standard tabletop scene with colored blocks."""
    scene = SceneGraph(name="tabletop_blocks")
    scene.add_object("table", [0.0, 0.0, 0.0], confidence=1.0)
    scene.add_object("red_block", [0.15, 0.10, 0.025], confidence=0.95)
    scene.add_object("blue_block", [-0.12, -0.08, 0.025], confidence=0.92)
    scene.add_object("green_block", [0.05, -0.15, 0.025], confidence=0.88)
    scene.add_object("box", [0.20, -0.20, 0.05], confidence=0.97)
    return scene


def create_household_scene() -> SceneGraph:
    """Household scene with fridge, cabinet, and objects."""
    scene = SceneGraph(name="household_kitchen")
    scene.add_object("table", [0.0, 0.0, 0.0], confidence=1.0)
    scene.add_object("fridge", [2.0, 0.0, 0.0], confidence=0.99)
    scene.add_object("cabinet", [-1.0, 1.5, 1.0], confidence=0.99)
    scene.add_object("milk", [2.0, 0.1, 0.5], confidence=0.93)
    scene.add_object("egg", [2.0, -0.1, 0.45], confidence=0.91)
    scene.add_object("apple", [0.1, 0.2, 0.03], confidence=0.96)
    scene.add_object("red_block", [0.3, 0.2, 0.025], confidence=0.97)
    scene.add_object("blue_block", [-0.3, -0.2, 0.025], confidence=0.97)
    return scene

def print_header(title: str) -> None:
    print(f"\n{title}\n")


def print_section(title: str) -> None:
    print(f"\n{title}")

def run_full_pipeline(task: str, use_household: bool = False, verbose: bool = True):

    results = {
        "task": task,
        "mode": "household" if use_household else "tabletop",
    }
    
    print_header(f"Running task: '{task}'")
    
    # Scene Graph Layer
    print_section("Step 1: What do we see? (Scene Graph)")
    
    if use_household:
        scene = create_household_scene()
    else:
        scene = create_tabletop_scene()
    
    print(f"Looking at scene: {scene.name}")
    print(f"Found {len(scene.objects)} objects:")
    for name, data in scene.objects.items():
        pos = data['pos']
        conf = data.get('confidence', 1.0)
        print(f"  - {name} at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] (confidence: {conf:.0%})")
    
    results["scene_graph"] = scene.to_dict()
    
    # Grounding Layer
    print_section("Step 2: Can we actually do this? (Grounding Check)")
    
    verifier = GroundingVerifier()
    grounding = verifier.verify_task_grounding(task, scene.to_dict())
    
    print(f"Grounding passed: {grounding.success}")
    print(f"   Why: {grounding.message}")
    print(f"   Overall confidence: {grounding.confidence_score:.0%}")
    
    if grounding.missing_objects:
        print(f"   Problem: can't find {grounding.missing_objects}")
    if grounding.low_confidence_objects:
        print(f"   Warning: not sure about {[o['name'] for o in grounding.low_confidence_objects]}")
    
    # Spatial relations
    print(f"\n   Figured out {len(grounding.inferred_relations)} spatial relations:")
    for rel in grounding.inferred_relations[:8]:  # Show top 8
        print(f"      • {rel.type}({rel.subject}, {rel.object}) [conf={rel.confidence:.2f}]")
    if len(grounding.inferred_relations) > 8:
        print(f"      ... and {len(grounding.inferred_relations) - 8} more")
    
    results["grounding"] = {
        "success": grounding.success,
        "message": grounding.message,
        "confidence_score": grounding.confidence_score,
        "missing_objects": grounding.missing_objects,
        "low_confidence_objects": [o['name'] for o in grounding.low_confidence_objects],
        "inferred_relations": [r.to_dict() for r in grounding.inferred_relations],
    }
    
    # CoT
    print_section("Step 3: Breaking down the task")
    
    cot_steps = []
    
    if HAS_EXTENDED_PLANNER and os.getenv("GEMINI_API_KEY"):
        try:
            planner = ExtendedPlanner()
            cot_steps = planner._decompose_task(task, example=None)
            print(f"   Got {len(cot_steps)} steps from the LLM:")
            for i, step in enumerate(cot_steps, 1):
                print(f"   Step {i}: {step}")
        except Exception as e:
            print(f"   LLM failed ({e}), falling back to rules...")
            print("   Using predefined step sequences instead.")
    else:
        print("   No API key found, using predefined steps...")
    
    # Fallback through rule-based decomposition
    if not cot_steps:
        task_type = WaypointGenerator.detect_task_type(task)
        if "pick" in task_type:
            cot_steps = [
                "Locate target object in scene",
                "Move gripper above object",
                "Lower gripper to object height",
                "Close gripper to grasp",
                "Lift object up"
            ]
        elif "stack" in task_type:
            cot_steps = [
                "Identify source and target objects",
                "Move gripper above source object",
                "Grasp source object",
                "Lift source object",
                "Move above target object",
                "Lower source onto target",
                "Release gripper"
            ]
        elif use_household and ("fridge" in task.lower() or "grab" in task.lower()):
            cot_steps = [
                "Navigate to fridge location",
                "Open fridge door",
                "Locate target object inside",
                "Grasp target object",
                "Lift object out",
                "Close fridge door"
            ]
        else:
            cot_steps = [
                "Locate target object",
                "Approach object",
                "Grasp object",
                "Complete action"
            ]
        
        for i, step in enumerate(cot_steps, 1):
            print(f"   Step {i}: {step}")
    
    results["cot_steps"] = cot_steps
    
    # Language Planner
    print_section("Step 4: Mapping to robot actions")
    
    translated_actions = []
    
    if HAS_TRANSLATOR:
        try:
            translator = ActionTranslator()
            
            # Find similar example
            example, sim_score = translator.find_similar_example(task)
            if example:
                print(f"   Found similar example: '{example['task']}' (match: {sim_score:.0%})")
            
            # CoT to actions
            translations = translator.translate_plan(cot_steps)
            print(f"\n   Translated to robot actions:")
            for orig, translated, score in translations:
                print(f"      '{orig[:40]:<40}' → '{translated}' (score={score:.2f})")
                translated_actions.append({
                    "original": orig,
                    "translated": translated,
                    "score": score
                })
        except Exception as e:
            print(f"   Translation failed: {e}")
    else:
        print("   Skipping - need sentence-transformers library for this")
    
    results["translated_actions"] = translated_actions
    
    # Waypoint Gen.
    print_section("Step 5: Where should the robot move?")
    
    if use_household:
        waypoints = HouseholdWaypointGenerator.generate(task, scene, current_location="table")
    else:
        waypoints = WaypointGenerator.generate(task, scene)
    
    print(f"   Generated {len(waypoints)} waypoints:")
    for i, wp in enumerate(waypoints):
        pos = wp.get('pos', [0, 0, 0])
        action = wp.get('action', 'unknown')
        label = wp.get('label', '')
        print(f"   [{i+1}] {action:<12} pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]  # {label}")    
    results["waypoints"] = waypoints
    
    # Summary
    print_header("Done! Here's what we got:")
    print(f"  Task: {task}")
    print(f"  Mode: {'Household' if use_household else 'Tabletop'}")
    print(f"  Objects in scene: {len(scene.objects)}")
    print(f"  Grounding check: {'PASSED' if grounding.success else 'FAILED'} ({grounding.confidence_score:.0%} confident)")
    print(f"  Task steps: {len(cot_steps)}")
    print(f"  Robot actions: {len(translated_actions)}")
    print(f"  Motion waypoints: {len(waypoints)}")
    print(f"  Ready to execute: {'Yes' if grounding.success and len(waypoints) > 0 else 'No'}")

    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline Demo for Grounded VLM Planner"
    )
    parser.add_argument(
        "task",
        type=str,
        nargs="?",
        default="stack red block on blue block",
        help="Task description (e.g., 'pick up the red block')"
    )
    parser.add_argument(
        "--household",
        action="store_true",
        help="Use household scene instead of tabletop"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()    
    results = run_full_pipeline(args.task, use_household=args.household)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
