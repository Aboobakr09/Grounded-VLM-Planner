# Grounded VLM Planner - Project Documentation

## What This Project Does

This system lets robots understand and execute natural language commands safely. The key idea is simple: before the robot tries to do anything, we check if it can actually do it. If you say "pick up the red block" but there's no red block visible, the system stops and tells you what's missing instead of failing mysteriously.

## The Problem We're Solving

Both papers have gaps when it comes to perception uncertainty:

- **Language Planner** is great at breaking down tasks, but it assumes all objects exist. No checking.
- **ReKep** tracks keypoints in real-time with closed-loop feedback, but can fail when keypoints get misidentified or occluded.

Neither one explicitly checks "does this object actually exist in the scene?" before starting. That's what we add:
- Do the required objects actually exist?
- Is the vision model confident enough?
- Do the positions make physical sense?

If something's off, the robot stops and tells you why.

## How It Works

The pipeline has five main stages:

### Stage 1: Perception
The system looks at the camera image using Gemini Vision. It finds objects like "red_block at position [0.3, 0.2, 0.025] with 92% confidence". All detected objects get stored in a scene graph.

### Stage 2: Grounding Verification
This is our main contribution. We extract what objects the task needs (from "stack red on blue" we get "red_block" and "blue_block"), then check if they exist in the scene graph. We also verify confidence scores and compute spatial relations like "red_block is left of blue_block".

### Stage 3: Task Decomposition
The LLM breaks down the task into steps. "Stack red on blue" becomes:
1. Locate red and blue blocks
2. Move gripper above red block
3. Lower and grasp
4. Lift up
5. Move above blue block
6. Lower and release

### Stage 4: Action Translation
Each step gets mapped to a robot-executable action using semantic similarity. "Grasp the red block" becomes "grasp red_block" from our action vocabulary.

### Stage 5: Waypoint Generation
Finally, we generate actual 3D positions for the robot to move to. Each waypoint has coordinates like [0.3, 0.2, 0.125] and an action like "approach" or "grasp".

## The Core Modules

| Module | Purpose |
|--------|---------|
| `scene_graph.py` | Stores objects with their 3D positions and confidence scores. Think of it as a structured map of what the robot sees. |
| `grounding.py` | The verification layer. Checks if the task can actually be done given what's in the scene. Returns clear error messages when things fail. |
| `perception.py` | Connects to Gemini Vision to extract objects from camera images. Parses the VLM's JSON output into our scene graph format. |
| `planner.py` | Orchestrates the whole pipeline. Runs grounding check first, then generates reasoning steps and waypoints. |
| `household_planner.py` | Extends the system beyond tabletop manipulation. Handles navigation, opening fridges and cabinets, and picking items from inside containers. |
| `common_sense.py` | Safety rules that catch dangerous actions. Won't let you microwave metal or put plastic in the oven. Also tracks what containers are open and adds cleanup actions. |
| `reasoning_verifier.py` | Checks action preconditions. Can't grasp something if the gripper is already full. Can't grab from a fridge that's closed. |
| `action_translator.py` | Uses sentence embeddings to match free-form LLM outputs to our fixed action vocabulary. This comes from Language Planner's approach. |

## Safety Architecture

We have three safety layers:

**Layer 1: Grounding** - Catches perception errors before planning starts. If an object is missing or uncertain, we abort immediately.

**Layer 2: Common Sense** - Catches dangerous action combinations. Microwave + metal = blocked. Also ensures containers get closed after use.

**Layer 3: Preconditions** - Catches impossible action sequences. Can't release if you're not holding anything. Can't grab if hands are full.

## What We Built On

This project extends **Language Planner** (ICML 2022) which showed that LLMs can decompose tasks and translate them to executable actions. Their limitation was no vision - they assumed all objects exist.

We also drew inspiration from **ReKep** (CoRL 2024) which uses VLM-generated constraints for manipulation. Their limitation was heavy infrastructure requirements.

Our contribution is adding the grounding verification layer that checks scene-task consistency before any planning happens, while keeping the system lightweight enough to run with just an API key.

## Quick Reference

Run a demo:
```bash
python demo_full_pipeline.py "stack red block on blue block"
```

Household mode:
```bash
python demo_full_pipeline.py "grab milk from fridge" --household
```

API key setup:
```bash
echo "GEMINI_API_KEY=your_key" > .env
```

## Limitations

There are some things this system can't do yet:
- **Single-object pick tasks** - "pick up the red block" will generate 0 waypoints because of a bug in the pick parser. Stack and multi-object tasks work fine though.
- **Hardcoded threshold** - The 0.6 confidence cutoff is just a guess. It should be tuned based on actual VLM error rates, or made adaptive.
- **Occlusion** - If an object is hidden behind something, grounding will fail even if the object actually exists.
- **Static scenes** - The system assumes nothing moves during execution. It won't replan if things shift around.
- **No depth sensor** - VLM position estimates can be off by 2-5cm. Real deployment would need depth cameras for accuracy.
- **Single arm only** - There's no support for bimanual manipulation.
- **No force control** - The system can't handle delicate objects or detect if a grasp failed.
- **Template constraints** - Constraints aren't learned from data, they're just pattern-matched.

These are engineering limitations, not fundamental ones. The architecture can scale with better sensors and perception.

## References

- Huang, W., et al. (2022). *Language Models as Zero-Shot Planners.* ICML 2022.
- Huang, W., et al. (2024). *ReKep: Relational Keypoint Constraints.* CoRL 2024.
