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

# NEW: Enhanced Action Matcher with VirtualHome vocabulary
try:
    from grounded_planner.enhanced_action_matcher import (
        EnhancedActionMatcher, 
        create_enhanced_matcher
    )
    HAS_ENHANCED_MATCHER = True
except Exception as e:
    HAS_ENHANCED_MATCHER = False
    print(f"EnhancedActionMatcher unavailable: {e}")

# Execution Safety Gate - key differentiator from vanilla Language Planner
try:
    from grounded_planner.execution_gate import (
        ExecutionGate,
        DomainDetector,
        TaskDomain
    )
    HAS_EXECUTION_GATE = True
except Exception as e:
    HAS_EXECUTION_GATE = False
    print(f"ExecutionGate unavailable: {e}")

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
    
    # Step 4: DIAGNOSTIC ONLY - Language Planner Translation Analysis
    # ⚠️ This uses VirtualHome vocabulary which is NOT suitable for robot manipulation!
    # It demonstrates why unconstrained semantic matching is dangerous.
    print_section("Step 4: [DIAGNOSTIC] VirtualHome Translation Analysis")
    print("   ⚠️  WARNING: VirtualHome vocab is NOT designed for robot manipulation!")
    print("   ⚠️  The following shows WHY this approach is dangerous:\n")
    
    dangerous_translations = []
    
    if HAS_TRANSLATOR:
        try:
            translator = ActionTranslator()
            
            # CoT to VirtualHome actions (this will produce bad results)
            translations = translator.translate_plan(cot_steps)
            
            low_confidence_count = 0
            for orig, translated, score in translations:
                status = "❌ UNSAFE" if score < 0.7 else "⚠️  LOW" if score < 0.85 else "✓"
                if score < 0.7:
                    low_confidence_count += 1
                print(f"      {status} '{orig[:35]:<35}' → '{translated}' ({score:.0%})")
                dangerous_translations.append({
                    "original": orig,
                    "translated": translated,
                    "score": score,
                    "unsafe": score < 0.7
                })
            
            if low_confidence_count > 0:
                print(f"\n   🚨 {low_confidence_count}/{len(translations)} translations are UNSAFE!")
                print("   🚨 These should NEVER be used for robot execution!")
                print("   → This is why Step 4b uses domain-specific vocabulary instead.")
        except Exception as e:
            print(f"   Translation failed: {e}")
    else:
        print("   Skipping - sentence-transformers not available")
    
    # Store for analysis but clearly mark as non-executable
    results["diagnostic_translations"] = dangerous_translations
    results["translation_warning"] = "VirtualHome translations are for DIAGNOSTIC ONLY - do not execute!"
    
    # Enhanced Action Matching (Language Planner Integration)
    vocab_name = "VirtualHome" if use_household else "Robot Manipulation"
    print_section(f"Step 4b: {vocab_name} vocabulary matching (Language Planner)")
    
    matched_actions = []
    
    if HAS_ENHANCED_MATCHER:
        try:
            # Create enhanced matcher: use robot vocab for tabletop, VirtualHome for household
            enhanced_matcher = create_enhanced_matcher(use_virtualhome=use_household)
            
            # Find similar example
            example, sim = enhanced_matcher.find_similar_example(task)
            if example:
                print(f"   Similar example: '{example['task']}' (match: {sim:.0%})")
            
            # Match CoT steps to action vocabulary
            matches = enhanced_matcher.translate_plan(cot_steps)
            print(f"\n   Matched to {vocab_name} actions:")
            for match in matches:
                print(f"      '{match.original_text[:35]:<35}' → '{match.matched_action}' (sim={match.similarity_score:.2f})")
                matched_actions.append(match.to_dict())
            
            # Optional: Try autoregressive generation if confidence is low
            avg_score = sum(m.similarity_score for m in matches) / len(matches) if matches else 0
            if avg_score < 0.6 and enhanced_matcher.llm:
                print(f"\n   Low confidence ({avg_score:.2f}). Trying autoregressive generation...")
                try:
                    autoreg_result = enhanced_matcher.autoregressive_plan(task, max_steps=8, verbose=False)
                    if autoreg_result.steps:
                        print(f"   Autoregressive plan ({len(autoreg_result.steps)} steps):")
                        for i, step in enumerate(autoreg_result.steps, 1):
                            print(f"      Step {i}: {step.matched_action} (score={step.similarity_score:.2f})")
                        results["autoregressive_plan"] = autoreg_result.to_dict()
                        
                        # CRITICAL FIX: Use autoregressive plan for execution gate if available
                        matched_actions = [step.to_dict() for step in autoreg_result.steps]
                        print(f"   → Using autoregressive plan for execution gate")
                        
                except Exception as ae:
                    print(f"   Autoregressive failed: {ae}")
                    
        except Exception as e:
            print(f"   Enhanced matching failed: {e}")
    else:
        print("   Skipping - EnhancedActionMatcher not available")
    
    results["matched_actions"] = matched_actions
    
    # Step 4c: EXECUTION GATE - The Key Safety Layer
    # This is what separates this system from vanilla Language Planner
    print_section("Step 4c: Execution Gate (Safety Verification)")
    
    execution_allowed = False
    gate_result = None
    
    if HAS_EXECUTION_GATE and matched_actions:
        try:
            # Detect domain and apply appropriate ontology
            detected_domain = DomainDetector.detect(task, scene.to_dict())
            ontology = "robot_primitives" if not use_household else "virtualhome"
            
            gate = ExecutionGate(ontology=ontology)
            gate_result = gate.evaluate(
                task=task,
                matched_actions=matched_actions,
                scene_objects=scene.to_dict()
            )
            
            # Report
            print(f"   Domain detected: {detected_domain.value}")
            print(f"   Ontology: {ontology}")
            print(f"   Threshold: {gate_result.domain_config.similarity_threshold:.0%}")
            print(f"   Domain executable: {gate_result.domain_config.executable}")
            
            print(f"\n   Action-by-action gate decisions:")
            for d in gate_result.decisions:
                status = "✓ PASS" if d.executable else "✗ BLOCK"
                print(f"      [{status}] '{d.action}' ({d.similarity_score:.0%})")
                if d.block_reason:
                    print(f"              └─ {d.block_reason}")
            
            execution_allowed = gate_result.all_executable
            
            print(f"\n   Gate Result: {len(gate_result.decisions) - gate_result.blocked_count}/{len(gate_result.decisions)} actions passed")
            
            if execution_allowed:
                print("   ✅ EXECUTION ALLOWED: All actions passed safety gate")
            else:
                print(f"   🚨 EXECUTION BLOCKED: {gate_result.blocked_count} action(s) failed gate")
                print("   → Waypoints will still be generated for visualization only")
            
            results["execution_gate"] = gate_result.to_dict()
            
        except Exception as e:
            print(f"   Execution gate failed: {e}")
            execution_allowed = False
    else:
        print("   Skipping - ExecutionGate not available or no actions to evaluate")
    
    results["execution_allowed"] = execution_allowed
    
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
    print(f"  Task steps (CoT): {len(cot_steps)}")
    print(f"  Diagnostic translations: {len(results.get('diagnostic_translations', []))}")
    print(f"  Matched actions ({vocab_name}): {len(matched_actions)}")
    print(f"  Motion waypoints: {len(waypoints)}")
    
    ready_to_execute = grounding.success and len(waypoints) > 0
    if HAS_EXECUTION_GATE:
        ready_to_execute = ready_to_execute and execution_allowed
        
    print(f"  Ready to execute: {'Yes' if ready_to_execute else 'No'}")
    if not execution_allowed and HAS_EXECUTION_GATE:
        print(f"  ⚠️  Execution blocked by safety gate")

    
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
