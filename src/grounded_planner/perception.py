"""Perception: Vision-Language Scene Extraction using Gemini Vision."""

import os
import re
import json
import base64
from typing import Optional

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from grounded_planner.scene_graph import SceneGraph
from grounded_planner.prompts import SCENE_EXTRACTION_PROMPT


class VisionSceneExtractor:
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        if not HAS_GENAI:
            raise ImportError("google-generativeai package required")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def extract(self, image_path: str) -> SceneGraph:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        mime_type = 'image/png'
        if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        
        response = self.model.generate_content([
            {'mime_type': mime_type, 'data': base64.b64encode(image_bytes).decode()},
            SCENE_EXTRACTION_PROMPT
        ])
        
        return self._parse_response(response.text, image_path)
    
    def _parse_response(self, text: str, scene_name: str) -> SceneGraph:
        scene = SceneGraph(name=scene_name)
        
        try:
            text = text.strip()
            if text.startswith('```'):
                text = re.sub(r'```json?\s*', '', text)
                text = re.sub(r'```\s*$', '', text)
            
            data = json.loads(text)
            
            for obj in data.get('objects', []):
                scene.add_object(
                    name=obj['name'],
                    pos=obj['pos'],
                    relations=obj.get('relations', []),
                    confidence=obj.get('confidence', 0.8)
                )
            
            if 'table' not in scene.objects:
                scene.add_object('table', [0, 0, 0], [], confidence=1.0)
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            scene.add_object('table', [0, 0, 0], [], confidence=1.0)
        
        return scene
