"""LLM Prompts: All prompts used in the grounded planner pipeline."""

COT_PROMPT = """
You are a robot task planner. Given a task, decompose it into simple steps.

## Example 1
Task: pick up the red block
Reasoning:
Step 1: Locate the red block on the table
Step 2: Move gripper above red block
Step 3: Lower gripper to red block
Step 4: Close gripper to grasp red block
Step 5: Lift red block up

## Example 2
Task: stack red block on blue block
Reasoning:
Step 1: Locate red block and blue block positions
Step 2: Move gripper above red block
Step 3: Lower and grasp red block
Step 4: Lift red block up
Step 5: Move red block above blue block
Step 6: Lower red block onto blue block
Step 7: Open gripper to release

## Example 3
Task: put all blocks in the box
Reasoning:
Step 1: Identify all blocks on table (red, blue, green)
Step 2: Grasp red block
Step 3: Move red block above box
Step 4: Release red block into box
Step 5: Grasp blue block
Step 6: Move blue block above box
Step 7: Release blue block into box
Step 8: Grasp green block
Step 9: Move green block above box
Step 10: Release green block into box

## Your Task
Task: {task}
Reasoning:
"""

# Dynamic few-shot prompt (Language Planner approach: use most similar example)
DYNAMIC_COT_PROMPT = """
You are a robot task planner. Given a task, decompose it into simple steps.
Use the provided example as a guide for formatting.

## Example (most similar task)
{example}

## Your Task
Task: {task}
Reasoning:
"""


SCENE_EXTRACTION_PROMPT = """
You are analyzing a top-down image of a tabletop with colored blocks.

Identify all objects in the image and estimate their 3D positions.
For each object, also provide a confidence score (0.0 to 1.0) indicating how certain you are.

Assume the table surface is at z=0, and blocks are 5cm (0.05m) tall.
Use the table center as origin (0, 0).
Estimate x, y positions in meters based on the image (typical table is 0.8m x 0.6m).

Return a JSON object with this exact format:
{
    "objects": [
        {"name": "red_block", "pos": [x, y, 0.025], "confidence": 0.95},
        {"name": "blue_block", "pos": [x, y, 0.025], "confidence": 0.88},
        ...
    ]
}

Rules:
- Block names should be: red_block, blue_block, green_block, yellow_block
- If there's a box, name it "box"
- pos is [x, y, z] in meters
- confidence is 0.0 to 1.0 (1.0 = very certain, 0.5 = uncertain)
- z should be 0.025 for blocks on table (half of 5cm height)
- z should be 0.05 for a box
- Estimate positions relative to table center (0, 0)
- Left of center = negative x, Right = positive x
- Top of image = negative y, Bottom = positive y

Confidence guidelines:
- 0.9+ : Object clearly visible, color unambiguous
- 0.7-0.9 : Object visible but partially occluded or color slightly unclear
- 0.5-0.7 : Object detected but uncertain about identity or position
- <0.5 : Very uncertain, might not exist

Respond ONLY with valid JSON, no other text.
"""


CONSTRAINT_PROMPT = """
You are generating constraints for a robot manipulation task.

Given:
- Task: {task}
- Objects: {objects}

Generate Python constraint functions that define when task stages are complete.
Each constraint returns a cost (0 = satisfied, >0 = not satisfied).

Example for "stack red block on blue block":

```python
import numpy as np

def stage1_subgoal(gripper_pos, objects):
    target = np.array(objects['red_block']['pos'])
    return np.linalg.norm(gripper_pos - target)

def stage2_subgoal(gripper_pos, objects):
    blue_pos = np.array(objects['blue_block']['pos'])
    target = blue_pos + [0, 0, 0.10]
    return np.linalg.norm(gripper_pos - target)

def stage3_subgoal(gripper_pos, objects):
    blue_pos = np.array(objects['blue_block']['pos'])
    target = blue_pos + [0, 0, 0.05]
    return np.linalg.norm(gripper_pos - target)
```

Now generate constraints for:
Task: {task}
Objects: {objects}

Respond with Python code only.
"""
