# Grounded VLM Planner
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Description

Our application allows robots to understand natural language commands and execute them safely by checking if the required objects actually exist in the scene before taking action.

The system takes a task like "stack red block on blue block" and breaks it down into steps the robot can follow. But here's the key difference from other systems: before the robot even starts planning, It verify that the red block and blue block are actually visible to the camera. If something's missing or uncertain, the robot stops and explains why it can't proceed.

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
cd grounded_vlm_planner
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
git clone https://github.com/aboobakr09/grounded_vlm_planner.git
cd grounded_vlm_planner
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
│   └── Documentation.md                   
├── demo_full_pipeline.py    
└── requirements.txt         
```

## License

This project is licensed under the MIT License, which allows anyone to use, modify, or contribute to the code freely.
