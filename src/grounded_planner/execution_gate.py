"""
Execution Gate - Ontology-Constrained Action Verification

This module provides the critical execution safety layer that separates
this system from vanilla Language Planner approaches.

Key Contribution:
"Unlike prior systems, we explicitly prevent cross-domain action grounding 
by enforcing ontology-specific execution gates."

Design Principles:
1. No action ontology should ever cross embodiment boundaries
2. Threshold gates block EXECUTION, not planning
3. Mismatch is treated as signal ("I don't know"), not noise
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class TaskDomain(Enum):
    """Supported task domains with distinct ontologies."""
    TABLETOP_MANIPULATION = "tabletop_manipulation"
    HOUSEHOLD_NAVIGATION = "household_navigation"
    MIXED_SYMBOLIC = "mixed_symbolic"
    UNKNOWN = "unknown"


@dataclass
class DomainConfig:
    """Configuration for a specific task domain."""
    domain: TaskDomain
    allowed_ontologies: List[str]
    similarity_threshold: float
    executable: bool
    description: str


# Domain configurations
DOMAIN_CONFIGS: Dict[TaskDomain, DomainConfig] = {
    TaskDomain.TABLETOP_MANIPULATION: DomainConfig(
        domain=TaskDomain.TABLETOP_MANIPULATION,
        allowed_ontologies=["robot_primitives"],
        similarity_threshold=0.7,  # Strict for physical execution
        executable=True,
        description="Gripper-based block manipulation"
    ),
    TaskDomain.HOUSEHOLD_NAVIGATION: DomainConfig(
        domain=TaskDomain.HOUSEHOLD_NAVIGATION,
        allowed_ontologies=["virtualhome"],
        similarity_threshold=0.6,  # VirtualHome is more symbolic
        executable=False,  # Symbolic planning only
        description="High-level household activity planning"
    ),
    TaskDomain.MIXED_SYMBOLIC: DomainConfig(
        domain=TaskDomain.MIXED_SYMBOLIC,
        allowed_ontologies=["robot_primitives", "virtualhome"],
        similarity_threshold=0.65,
        executable=False,  # Requires human verification
        description="Mixed domain - requires verification"
    ),
    TaskDomain.UNKNOWN: DomainConfig(
        domain=TaskDomain.UNKNOWN,
        allowed_ontologies=[],
        similarity_threshold=1.0,  # Block everything
        executable=False,
        description="Unknown domain - cannot execute"
    ),
}


@dataclass
class ExecutionDecision:
    """Result of execution gate evaluation."""
    action: str
    original_text: str
    similarity_score: float
    
    # Gate results
    passes_threshold: bool
    ontology_allowed: bool
    domain_consistent: bool
    
    # Final decision
    executable: bool
    block_reason: str = ""
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "original": self.original_text,
            "similarity": self.similarity_score,
            "passes_threshold": self.passes_threshold,
            "ontology_allowed": self.ontology_allowed,
            "domain_consistent": self.domain_consistent,
            "executable": self.executable,
            "block_reason": self.block_reason
        }


@dataclass 
class ExecutionGateResult:
    """Complete result from execution gate."""
    task: str
    detected_domain: TaskDomain
    domain_config: DomainConfig
    decisions: List[ExecutionDecision] = field(default_factory=list)
    
    @property
    def all_executable(self) -> bool:
        """True only if ALL actions pass the gate."""
        return all(d.executable for d in self.decisions)
    
    @property
    def blocked_count(self) -> int:
        return sum(1 for d in self.decisions if not d.executable)
    
    @property
    def executable_actions(self) -> List[str]:
        return [d.action for d in self.decisions if d.executable]
    
    @property
    def blocked_actions(self) -> List[Tuple[str, str]]:
        """Returns list of (action, reason) for blocked actions."""
        return [(d.action, d.block_reason) for d in self.decisions if not d.executable]
    
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "detected_domain": self.detected_domain.value,
            "domain_config": {
                "threshold": self.domain_config.similarity_threshold,
                "executable": self.domain_config.executable,
                "allowed_ontologies": self.domain_config.allowed_ontologies
            },
            "all_executable": self.all_executable,
            "blocked_count": self.blocked_count,
            "decisions": [d.to_dict() for d in self.decisions]
        }


class DomainDetector:
    """
    Detects task domain from natural language description.
    
    Pipeline:
    Task → Domain Classifier → Allowed Ontologies → Action Matching
    """
    
    # Domain indicators (keyword-based for now, could be ML-based)
    TABLETOP_INDICATORS = {
        'block', 'stack', 'pick up', 'gripper', 'grasp', 'lift',
        'place', 'red_block', 'blue_block', 'green_block', 'box',
        'tower', 'table', 'approach', 'descend', 'release'
    }
    
    HOUSEHOLD_INDICATORS = {
        'fridge', 'cabinet', 'door', 'walk', 'navigate', 'kitchen',
        'living room', 'bedroom', 'bathroom', 'couch', 'chair',
        'television', 'remote', 'milk', 'egg', 'cook', 'wash',
        'clothes', 'shower', 'computer', 'phone'
    }
    
    @classmethod
    def detect(cls, task: str, scene_objects: Optional[Dict] = None) -> TaskDomain:
        """
        Detect task domain from task description and optional scene context.
        
        Args:
            task: Natural language task description
            scene_objects: Optional scene graph objects dict
            
        Returns:
            Detected TaskDomain
        """
        task_lower = task.lower()
        
        # Count domain indicators
        tabletop_score = sum(1 for ind in cls.TABLETOP_INDICATORS if ind in task_lower)
        household_score = sum(1 for ind in cls.HOUSEHOLD_INDICATORS if ind in task_lower)
        
        # Boost from scene context
        if scene_objects:
            obj_names = ' '.join(scene_objects.keys()).lower()
            tabletop_score += sum(1 for ind in cls.TABLETOP_INDICATORS if ind in obj_names)
            household_score += sum(1 for ind in cls.HOUSEHOLD_INDICATORS if ind in obj_names)
        
        # Decision logic
        if tabletop_score > 0 and household_score == 0:
            return TaskDomain.TABLETOP_MANIPULATION
        elif household_score > 0 and tabletop_score == 0:
            return TaskDomain.HOUSEHOLD_NAVIGATION
        elif tabletop_score > 0 and household_score > 0:
            return TaskDomain.MIXED_SYMBOLIC
        else:
            return TaskDomain.UNKNOWN
    
    @classmethod
    def get_config(cls, domain: TaskDomain) -> DomainConfig:
        return DOMAIN_CONFIGS[domain]


class ExecutionGate:
    """
    The execution safety gate.
    
    Evaluates each action against:
    1. Similarity threshold (ontology-specific)
    2. Ontology allowlist (domain-specific)
    3. Domain consistency
    
    This is the key differentiator from vanilla Language Planner.
    """
    
    def __init__(self, ontology: str = "robot_primitives"):
        """
        Args:
            ontology: Which action ontology is being used
                     ("robot_primitives" or "virtualhome")
        """
        self.ontology = ontology
    
    def evaluate(
        self,
        task: str,
        matched_actions: List[Dict],  # [{original, matched, similarity}]
        scene_objects: Optional[Dict] = None,
        force_domain: Optional[TaskDomain] = None
    ) -> ExecutionGateResult:
        """
        Evaluate all matched actions through the execution gate.
        
        Args:
            task: Original task description
            matched_actions: List of action match results
            scene_objects: Optional scene graph for domain detection
            force_domain: Override domain detection
            
        Returns:
            ExecutionGateResult with per-action decisions
        """
        # Step 1: Detect domain
        if force_domain:
            domain = force_domain
        else:
            domain = DomainDetector.detect(task, scene_objects)
        
        config = DomainDetector.get_config(domain)
        
        result = ExecutionGateResult(
            task=task,
            detected_domain=domain,
            domain_config=config
        )
        
        # Step 2: Evaluate each action
        for action_match in matched_actions:
            original = action_match.get("original", "")
            matched = action_match.get("matched", action_match.get("matched_action", ""))
            similarity = action_match.get("similarity", action_match.get("similarity_score", 0.0))
            
            # Gate checks
            passes_threshold = similarity >= config.similarity_threshold
            ontology_allowed = self.ontology in config.allowed_ontologies
            domain_consistent = config.executable  # Domain must allow execution
            
            # Determine executability
            executable = passes_threshold and ontology_allowed and domain_consistent
            
            # Build block reason if not executable
            block_reason = ""
            if not executable:
                reasons = []
                if not passes_threshold:
                    reasons.append(f"similarity {similarity:.0%} < threshold {config.similarity_threshold:.0%}")
                if not ontology_allowed:
                    reasons.append(f"ontology '{self.ontology}' not allowed for {domain.value}")
                if not domain_consistent:
                    reasons.append(f"domain {domain.value} is not executable")
                block_reason = "; ".join(reasons)
            
            decision = ExecutionDecision(
                action=matched,
                original_text=original,
                similarity_score=similarity,
                passes_threshold=passes_threshold,
                ontology_allowed=ontology_allowed,
                domain_consistent=domain_consistent,
                executable=executable,
                block_reason=block_reason
            )
            
            result.decisions.append(decision)
        
        return result
    
    def print_report(self, result: ExecutionGateResult) -> None:
        """Print execution gate report."""
        print(f"\n   Domain: {result.detected_domain.value}")
        print(f"   Threshold: {result.domain_config.similarity_threshold:.0%}")
        print(f"   Domain executable: {result.domain_config.executable}")
        print(f"   Ontology: {self.ontology}")
        
        print(f"\n   Action Decisions:")
        for d in result.decisions:
            status = "✓ EXEC" if d.executable else "✗ BLOCK"
            print(f"      [{status}] '{d.action}' ({d.similarity_score:.0%})")
            if d.block_reason:
                print(f"              └─ {d.block_reason}")
        
        print(f"\n   Summary: {len(result.decisions) - result.blocked_count}/{len(result.decisions)} actions executable")
        
        if not result.all_executable:
            print(f"   🚨 EXECUTION BLOCKED: {result.blocked_count} action(s) failed gate")


if __name__ == "__main__":
    print("=" * 60)
    print("Execution Gate Test")
    print("=" * 60)
    
    # Test domain detection
    print("\n1. Domain Detection:")
    test_tasks = [
        "stack red block on blue block",
        "get milk from fridge",
        "pick up block and put in fridge",
        "do something random"
    ]
    
    for task in test_tasks:
        domain = DomainDetector.detect(task)
        config = DomainDetector.get_config(domain)
        print(f"   '{task[:40]:<40}' → {domain.value} (exec={config.executable})")
    
    # Test execution gate
    print("\n2. Execution Gate:")
    
    test_actions = [
        {"original": "close gripper", "matched": "close gripper", "similarity": 1.0},
        {"original": "move above block", "matched": "move gripper above object", "similarity": 0.79},
        {"original": "do something", "matched": "find wall", "similarity": 0.35},
    ]
    
    gate = ExecutionGate(ontology="robot_primitives")
    result = gate.evaluate("stack red block on blue block", test_actions)
    gate.print_report(result)
