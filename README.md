# Grounded VLM Planner
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Description

Our application allows robots to understand natural language commands and execute them safely by checking if the required objects actually exist in the scene before taking action.

The system takes a task like "stack red block on blue block" and breaks it down into steps the robot can follow. But here's the key difference from other systems: before the robot even starts planning, The system verifies that the red block and blue block are actually visible to the camera. If something's missing or uncertain, the robot stops and explains why it can't proceed.

There are several problems this system is trying to solve. Current vision-language systems often "hallucinate" - they assume objects exist even when they don't, leading to failed executions. Additionally, most systems don't explain their failures, leaving developers guessing what went wrong. Lastly, existing solutions like ReKep require expensive hardware and complex setups, while our system runs with just a Gemini API key.

## Key Features

### Grounding Verification
Before any planning happens, the system checks if the task is actually doable.
- It extracts what objects are needed from the task description
- It checks if those objects were detected by the camera
- It verifies the detection confidence is high enough (above 60%)
- If anything fails, it aborts and tells you exactly what's missing

### Chain-of-Thought Decomposition
Once grounding passes, the system breaks down the task into simple steps.
- "Stack red on blue" becomes: locate blocks → grasp red → lift → move over blue → place → release
- Uses Gemini 2.0 to generate these steps
- Each step is clear and actionable

### Scene Graph
The system builds a structured representation of what it sees.
- Every object has a 3D position like [0.3, 0.2, 0.025] meters
- Every object has a confidence score from the vision model
- Spatial relations are computed (what's left of what, what's on top of what)

### Waypoint Generation
The system generates actual robot motion waypoints.
- Approach the object from above
- Lower down to grasp
- Lift up after grasping
- Move to destination
- Lower and release

Each waypoint has a 3D position the robot should move to.

### Safety Layers
There are three safety checks throughout the pipeline.
- **Layer 1 (Grounding)**: Are the required objects actually there?
- **Layer 2 (Common Sense)**: Is the action safe? (won't let you microwave metal)
- **Layer 3 (Preconditions)**: Can this action be done right now? (can't grasp if gripper is full)

### Household Support
The system handles more than just tabletop manipulation.
- Navigation to different locations (fridge, cabinet, table)
- Opening and closing doors and drawers
- Picking items from inside the fridge
- All with the same grounding verification

## Instructions

Users can run the demo by navigating to the project folder and running the main demo script. The system requires a Gemini API key to function since it uses Gemini Vision for understanding scenes.

**Running the Demo**

Open your terminal and go to the project folder:
```
cd Grounded-VLM-Planner
```

Run the demo with a task:
```
python demo_full_pipeline.py "stack red block on blue block"
```

The output will show you five stages:
1. Scene Graph - what objects were detected and their positions
2. Grounding Verification - whether all required objects exist
3. Chain-of-Thought - the step-by-step breakdown from the LLM
4. Action Translation - mapping steps to executable actions
5. Waypoints - the actual 3D positions for robot motion

**Household Mode**

For tasks involving navigation and fridge/cabinet interactions:
```
python demo_full_pipeline.py "grab milk from fridge" --household
```

**Using as a Library**

You can also import the planner in your own Python code:
```python
from grounded_planner import SceneGraph, CombinedPlanner

scene = SceneGraph("demo")
scene.add_object("red_block", [0.3, 0.2, 0.025], confidence=0.92)
scene.add_object("blue_block", [0.5, 0.4, 0.025], confidence=0.88)

planner = CombinedPlanner()
result = planner.plan("stack red block on blue block", scene)

if result.grounding_success:
    print(f"Generated {len(result.waypoints)} waypoints")
else:
    print(f"Failed: {result.grounding_message}")
```

## Development Requirements

The application runs on any machine with Python 3.9+ and a Gemini API key.

**Technical Requirements**
- Language: Python 3.9+
- Main Dependencies: google-generativeai, numpy, sentence-transformers
- API: Gemini 2.0 Flash (requires API key)

**Installation**

Clone the repository:
```
git clone https://github.com/Aboobakr09/Grounded-VLM-Planner.git
cd Grounded-VLM-Planner
```

Install dependencies:
```
pip install -e .
```

Set up your API key:
```
echo "GEMINI_API_KEY=your_key_here" > .env
```

**Project Structure**
```
grounded_vlm_planner/
├── src/grounded_planner/    
│   ├── scene_graph.py       
│   ├── grounding.py         
│   ├── perception.py        
│   ├── planner.py           
│   ├── household_planner.py 
│   ├── common_sense.py      
│   └── reasoning_verifier.py 
├── configs/                 
├── docs/
│   └── Project_Documentation.md                   
├── demo_full_pipeline.py    
└── requirements.txt         
```

**Expected Output**
Running a simple stacking task:
```
python demo_full_pipeline.py "stack red block on blue block"
```
The system executes a 5-stage verification and planning pipeline:
```
Running task: 'stack red block on blue block'


Step 1: What do we see? (Scene Graph)
Looking at scene: tabletop_blocks
Found 5 objects:
  - table at [0.00, 0.00, 0.00] (confidence: 100%)
  - red_block at [0.15, 0.10, 0.03] (confidence: 95%)
  - blue_block at [-0.12, -0.08, 0.03] (confidence: 92%)
  - green_block at [0.05, -0.15, 0.03] (confidence: 88%)
  - box at [0.20, -0.20, 0.05] (confidence: 97%)

Step 2: Can we actually do this? (Grounding Check)
Grounding passed: True
   Why: Grounding verified successfully
   Overall confidence: 93%

   Figured out 15 spatial relations:
      • on(red_block, table) [conf=1.00]
      • right_of(red_block, blue_block) [conf=1.00]
      • right_of(red_block, green_block) [conf=0.50]
      • left_of(red_block, box) [conf=0.25]
      • on(blue_block, table) [conf=1.00]
      • left_of(blue_block, red_block) [conf=1.00]
      • left_of(blue_block, green_block) [conf=0.85]
      • left_of(blue_block, box) [conf=1.00]
      ... and 7 more

Step 3: Breaking down the task
   Got 8 steps from the LLM:
   Step 1: Locate the red block and the blue block
   Step 2: Move the gripper above the red block
   Step 3: Lower the gripper to grasp the red block
   Step 4: Close the gripper to grasp the red block
   Step 5: Lift the red block
   Step 6: Move the red block above the blue block
   Step 7: Lower the red block onto the blue block
   Step 8: Open the gripper to release the red block

Step 4: Mapping to robot actions
   Found similar example: 'stack red block on blue block' (match: 100%)

   Translated to robot actions:
      'Locate the red block and the blue block ' → 'grasp red_block' (score=0.61)
      'Move the gripper above the red block    ' → 'grasp red_block' (score=0.64)
      'Lower the gripper to grasp the red block' → 'grasp red_block' (score=0.71)
      'Close the gripper to grasp the red block' → 'grasp red_block' (score=0.70)
      'Lift the red block                      ' → 'grasp red_block' (score=0.67)
      'Move the red block above the blue block ' → 'grasp blue_block' (score=0.61)
      'Lower the red block onto the blue block ' → 'grasp red_block' (score=0.56)
      'Open the gripper to release the red bloc' → 'grasp red_block' (score=0.62)

Step 5: Where should the robot move?
   Generated 7 waypoints:
   [1] approach     pos=[0.15, 0.10, 0.12]  # above red_block
   [2] descend      pos=[0.15, 0.10, 0.03]  # at red_block
   [3] grasp        pos=[0.15, 0.10, 0.03]  # grasp red_block
   [4] lift         pos=[0.15, 0.10, 0.17]  # lift red_block
   [5] move         pos=[-0.12, -0.08, 0.17]  # above blue_block
   [6] descend      pos=[-0.12, -0.08, 0.08]  # place on blue_block
   [7] release      pos=[-0.12, -0.08, 0.12]  # release

Done! Here's what we got:

  Task: stack red block on blue block
  Mode: Tabletop
  Objects in scene: 5
  Grounding check: PASSED (93% confident)
  Task steps: 8
  Robot actions: 8
  Motion waypoints: 7
  Ready to execute: Yes
```

  
## License

This project is licensed under the MIT License, which allows anyone to use, modify, or contribute to the code freely.
