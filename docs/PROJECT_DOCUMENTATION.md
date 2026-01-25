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

The pipeline has five main stages, plus a safety verification layer:

### Stage 1: Perception
The system looks at the camera image using Gemini Vision. It finds objects like "red_block at position [0.3, 0.2, 0.025] with 92% confidence". All detected objects get stored in a scene graph.

### Stage 2: Grounding Verification
This is our main contribution. We extract what objects the task needs (from "stack red on blue" we get "red_block" and "blue_block"), then check if they exist in the scene graph. We also verify confidence scores and compute spatial relations like "red_block is left of blue_block".

### Stage 3: Task Decomposition
The LLM breaks down the task into steps using Chain-of-Thought reasoning. "Stack red on blue" becomes:
1. Locate red and blue blocks
2. Move gripper above red block
3. Lower and grasp
4. Lift up
5. Move above blue block
6. Lower and release

### Stage 4: Action Translation & Execution Gate
Each step gets mapped to a robot-executable action using semantic similarity. But here's what makes us different from vanilla Language Planner: we run every action through an **Execution Gate**.

The gate checks:
- Is the similarity score high enough? (70% threshold for robot manipulation)
- Is this action allowed in the current domain?
- Does the action make sense given the scene?

If any action fails the gate, execution is **blocked** and we tell you why.

### Stage 5: Waypoint Generation
Finally, we generate actual 3D positions for the robot to move to. Each waypoint has coordinates like [0.3, 0.2, 0.125] and an action like "approach" or "grasp".

## The Core Modules

| Module | Purpose |
|--------|---------|
| `scene_graph.py` | Stores objects with their 3D positions and confidence scores. Think of it as a structured map of what the robot sees. |
| `grounding.py` | The verification layer. Checks if the task can actually be done given what's in the scene. Returns clear error messages when things fail. |
| `perception.py` | Connects to Gemini Vision to extract objects from camera images. Parses the VLM's JSON output into our scene graph format. |
| `planner.py` | Generates waypoints for tabletop manipulation. Handles pick, stack, and tower tasks. |
| `extended_planner.py` | Orchestrates the LLM-based task decomposition using Chain-of-Thought prompting. |
| `enhanced_action_matcher.py` | Maps natural language steps to valid robot actions using vector similarity and optional autoregressive fallback. |
| `execution_gate.py` | The final safety barrier. Validates actions against domain rules and similarity thresholds. Blocks unsafe actions. |
| `household_planner.py` | Extends the system beyond tabletop manipulation. Handles navigation, opening fridges and cabinets, and picking items from inside containers. |
| `common_sense.py` | Safety rules that catch dangerous actions. Won't let you microwave metal or put plastic in the oven. Also tracks what containers are open and adds cleanup actions. |
| `reasoning_verifier.py` | Checks action preconditions. Can't grasp something if the gripper is already full. Can't grab from a fridge that's closed. |
| `action_translator.py` | Uses sentence embeddings to match free-form LLM outputs to our fixed action vocabulary. This comes from Language Planner's approach. |
| `action_verifier.py` | Lightweight gripper state checker. Verifies the gripper isn't already holding something before grasping. |
| `prompts.py` | All the LLM prompts used for task decomposition, scene extraction, and constraint generation. |

## Safety Architecture

We have three safety layers:

**Layer 1: Grounding** - Catches perception errors before planning starts. If an object is missing or uncertain, we abort immediately.

**Layer 2: Common Sense** - Catches dangerous action combinations. Microwave + metal = blocked. Also ensures containers get closed after use.

**Layer 3: Execution Gate** - Catches low-confidence translations and domain violations. If the system isn't confident enough in the mapping from language to action, it refuses to execute.

## What We Built On

**Language Planner** (ICML 2022) showed that LLMs can decompose tasks and translate them to robot actions. Super influential work. But they focused on planning—no verification that objects actually exist.

**ReKep** (CoRL 2024) does VLM-generated keypoint constraints with closed-loop feedback. Their system tracks keypoints in real-time, which is cool, but it can fail when keypoints get misidentified or when point trackers lose objects.

Our contribution is complementary: we check scene-task consistency *before* planning even starts. Catches perception errors early instead of mid-execution.

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

There are some things this system can't handle yet:

- **Hardcoded threshold** - The 0.6 confidence threshold is a guess. Should be tuned based on real VLM error rates.
- **No occlusion handling** - If an object is hidden behind something, grounding will fail even though the object exists.
- **Static scenes only** - Assumes nothing moves during execution. No replanning if things change.
- **Single arm** - Only supports one robot arm, no bimanual manipulation.
- **Simple geometric checks** - Just checking if positions are within bounds. No collision detection.
- **API dependency** - Requires Gemini API for LLM features. Uses fallback rules when API quota is exceeded.

## References

- Huang, W., et al. (2022). *Language Models as Zero-Shot Planners.* ICML 2022.
- Huang, W., et al. (2024). *ReKep: Relational Keypoint Constraints.* CoRL 2024.
