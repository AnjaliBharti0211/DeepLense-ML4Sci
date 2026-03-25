"""
Workflow orchestration patterns inspired by HEPTAPOD.

This module implements agentic workflow patterns from:
- Menzo et al. 2025 "HEPTAPOD: Orchestrating High Energy Physics Workflows
  Towards Autonomous Agency" (arXiv:2512.15867)

Key patterns:
1. Schema-validated tool database
2. Run-card driven configuration
3. Structured error handling with recovery
4. Human-in-the-loop checkpoints
5. State/context propagation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, Field


# =============================================================================
# Workflow Status and State Management (HEPTAPOD-inspired)
# =============================================================================


class TaskStatus(str, Enum):
    """Task status markers (from HEPTAPOD todo list pattern)."""

    PENDING = "pending"  # [ ]
    IN_PROGRESS = "in_progress"  # [*]
    COMPLETED = "completed"  # [x]
    FAILED = "failed"  # [!]
    BLOCKED = "blocked"  # [~]
    SKIPPED = "skipped"  # [-]


class WorkflowMode(str, Enum):
    """
    Workflow interaction modes (from HEPTAPOD).

    - TODO: Predefined task list execution
    - PLANNER: Autonomous workflow planning
    - EXPLORER: Interactive query mode
    """

    TODO = "todo"
    PLANNER = "planner"
    EXPLORER = "explorer"


@dataclass
class WorkflowTask:
    """
    A single task in the workflow (HEPTAPOD todo list item).

    Attributes:
        task_id: Unique identifier
        description: Human-readable task description
        status: Current task status
        dependencies: List of task IDs this depends on
        metadata: Additional task metadata
    """

    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None

    def mark_in_progress(self) -> None:
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def mark_completed(self) -> None:
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.utcnow()

    def to_status_line(self) -> str:
        """Format as HEPTAPOD-style status line."""
        markers = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.IN_PROGRESS: "[*]",
            TaskStatus.COMPLETED: "[x]",
            TaskStatus.FAILED: "[!]",
            TaskStatus.BLOCKED: "[~]",
            TaskStatus.SKIPPED: "[-]",
        }
        return f"{markers[self.status]} {self.description}"


@dataclass
class WorkflowState:
    """
    Complete workflow state with task tracking.

    Implements HEPTAPOD's context propagation pattern where
    tools write metadata into conversation state.
    """

    workflow_id: str
    mode: WorkflowMode = WorkflowMode.TODO
    tasks: list[WorkflowTask] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_task(
        self,
        task_id: str,
        description: str,
        dependencies: list[str] | None = None,
    ) -> WorkflowTask:
        """Add a new task to the workflow."""
        task = WorkflowTask(
            task_id=task_id,
            description=description,
            dependencies=dependencies or [],
        )
        self.tasks.append(task)
        return task

    def get_task(self, task_id: str) -> WorkflowTask | None:
        """Get a task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_pending_tasks(self) -> list[WorkflowTask]:
        """Get all pending tasks with satisfied dependencies."""
        completed_ids = {
            t.task_id for t in self.tasks if t.status == TaskStatus.COMPLETED
        }
        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.dependencies)
        ]

    def update_context(self, key: str, value: Any) -> None:
        """Update workflow context (HEPTAPOD context propagation)."""
        self.context[key] = value

    def to_todo_list(self) -> str:
        """Format as HEPTAPOD-style todo list."""
        lines = ["## Workflow Tasks", ""]
        for task in self.tasks:
            lines.append(task.to_status_line())
        return "\n".join(lines)


# =============================================================================
# Structured Error Handling (HEPTAPOD-inspired)
# =============================================================================


class StructuredError(BaseModel):
    """
    Structured error for machine-interpretable failure modes.

    From HEPTAPOD: "Tools return structured error objects that
    agents can interpret" with predictable failure patterns.
    """

    error_type: str = Field(description="Error category")
    message: str = Field(description="Human-readable error message")
    recoverable: bool = Field(
        default=True, description="Whether automatic recovery is possible"
    )
    suggested_action: str | None = Field(
        default=None, description="Suggested recovery action"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional error context"
    )

    def to_json_response(self) -> dict[str, Any]:
        """Format as JSON response for agent consumption."""
        return {f"Error: {self.error_type}": self.message, **self.context}


class StructuredResult(BaseModel):
    """
    Structured result for tool outputs.

    From HEPTAPOD: All tools return "structured JSON outputs"
    with consistent field structure.
    """

    status: str = Field(default="ok", description="Status: 'ok' or 'error'")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Result data"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Result metadata"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings"
    )

    @classmethod
    def success(cls, data: dict[str, Any], **metadata: Any) -> StructuredResult:
        """Create successful result."""
        return cls(status="ok", data=data, metadata=metadata)

    @classmethod
    def error(cls, error: StructuredError) -> StructuredResult:
        """Create error result."""
        return cls(
            status="error",
            data=error.to_json_response(),
            metadata={"recoverable": error.recoverable},
        )


# =============================================================================
# Run-Card Configuration (HEPTAPOD-inspired)
# =============================================================================


class RunCardConfig(BaseModel):
    """
    Run-card style configuration with placeholder substitution.

    From HEPTAPOD: "Run cards provide complete, standalone
    configurations that can be version-controlled" with
    placeholder substitution pattern.
    """

    name: str = Field(description="Configuration name")
    version: str = Field(default="1.0", description="Config version")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Configuration parameters"
    )
    placeholders: dict[str, str] = Field(
        default_factory=dict, description="Placeholder -> value mappings"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def resolve_placeholders(self, values: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve placeholders in parameters.

        Pattern: [[PLACEHOLDER_NAME]] -> actual value
        """
        resolved = {}
        for key, value in self.parameters.items():
            if isinstance(value, str) and value.startswith("[[") and value.endswith("]]"):
                placeholder_name = value[2:-2]
                if placeholder_name in values:
                    resolved[key] = values[placeholder_name]
                else:
                    resolved[key] = value  # Keep unresolved
            else:
                resolved[key] = value
        return resolved

    def to_runcard_string(self) -> str:
        """Format as run-card configuration string."""
        lines = [
            f"# {self.name} Configuration",
            f"# Version: {self.version}",
            "",
        ]
        for key, value in self.parameters.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
        return "\n".join(lines)


# =============================================================================
# Human-in-the-Loop Checkpoints (HEPTAPOD-inspired)
# =============================================================================


class ApprovalCheckpoint(BaseModel):
    """
    Human approval checkpoint for workflow control.

    From HEPTAPOD: Run cards "serve as natural boundary at
    which the researcher may inspect, modify, or approve
    the configuration."
    """

    checkpoint_id: str = Field(description="Unique checkpoint identifier")
    description: str = Field(description="What is being approved")
    requires_approval: bool = Field(default=True)
    auto_approve_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for auto-approval",
    )
    approval_status: str | None = Field(
        default=None, description="'approved', 'rejected', or None"
    )
    reviewer_notes: str | None = Field(default=None)
    reviewed_at: datetime | None = Field(default=None)

    def approve(self, notes: str | None = None) -> None:
        """Mark checkpoint as approved."""
        self.approval_status = "approved"
        self.reviewer_notes = notes
        self.reviewed_at = datetime.utcnow()

    def reject(self, notes: str) -> None:
        """Mark checkpoint as rejected."""
        self.approval_status = "rejected"
        self.reviewer_notes = notes
        self.reviewed_at = datetime.utcnow()

    def can_auto_approve(self, confidence: float) -> bool:
        """Check if automatic approval is allowed."""
        return not self.requires_approval or confidence >= self.auto_approve_threshold


# =============================================================================
# Tool Registry (HEPTAPOD-inspired RuntimeField/StateField pattern)
# =============================================================================


@dataclass
class ToolField:
    """
    Tool field specification (HEPTAPOD RuntimeField/StateField pattern).

    From HEPTAPOD:
    - RuntimeField: Agent-provided values (paths, configs, params)
    - StateField: Orchestration-layer injected values
    """

    name: str
    description: str
    field_type: str  # "runtime" or "state"
    required: bool = True
    default: Any = None


@dataclass
class ToolSpec:
    """
    Tool specification for registry.

    From HEPTAPOD: "Tools serve three roles:
    1. Formalizing how agents invoke domain-specific capabilities
    2. Returning structured outputs downstream steps can consume
    3. Defining clear execution boundaries for reproducibility"
    """

    name: str
    description: str
    docstring: str  # Used by LLM to understand when/why to use tool
    fields: list[ToolField] = field(default_factory=list)
    returns: str = "StructuredResult"
    category: str = "general"

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON schema for LLM consumption."""
        properties = {}
        required = []

        for f in self.fields:
            properties[f.name] = {
                "type": "string",  # Simplified - would be more complex in practice
                "description": f.description,
            }
            if f.required:
                required.append(f.name)

        return {
            "name": self.name,
            "description": self.docstring,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """
    Registry of available tools for agent orchestration.

    Implements HEPTAPOD's "schema-validated tools database
    exposing domain-specific capabilities."
    """

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool specification."""
        self._tools[spec.name] = spec
        if spec.category not in self._categories:
            self._categories[spec.category] = []
        self._categories[spec.category].append(spec.name)

    def get(self, name: str) -> ToolSpec | None:
        """Get tool by name."""
        return self._tools.get(name)

    def list_by_category(self, category: str) -> list[ToolSpec]:
        """List tools by category."""
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names]

    def to_llm_tools(self) -> list[dict[str, Any]]:
        """Export all tools as LLM-compatible function specs."""
        return [spec.to_schema() for spec in self._tools.values()]


# =============================================================================
# Pre-configured DeepLense Tools Registry
# =============================================================================


def create_deeplense_tool_registry() -> ToolRegistry:
    """Create tool registry with DeepLense-specific tools."""
    registry = ToolRegistry()

    # Simulation tools
    registry.register(
        ToolSpec(
            name="parse_simulation_request",
            description="Parse natural language into simulation config",
            docstring=(
                "Use this tool to extract simulation parameters from "
                "natural language descriptions. Returns structured config "
                "with confidence score and any clarification questions."
            ),
            fields=[
                ToolField(
                    name="prompt",
                    description="Natural language simulation request",
                    field_type="runtime",
                ),
            ],
            category="parsing",
        )
    )

    registry.register(
        ToolSpec(
            name="run_simulation",
            description="Execute gravitational lensing simulation",
            docstring=(
                "Use this tool to generate lens images after configuration "
                "is complete. Requires confirmation for >10 images."
            ),
            fields=[
                ToolField(
                    name="config",
                    description="Simulation configuration",
                    field_type="runtime",
                ),
                ToolField(
                    name="confirm",
                    description="Confirm execution for large batches",
                    field_type="runtime",
                    required=False,
                    default=True,
                ),
            ],
            category="simulation",
        )
    )

    registry.register(
        ToolSpec(
            name="validate_config",
            description="Validate simulation configuration",
            docstring=(
                "Use this tool to check parameter validity and "
                "physical consistency before running simulations."
            ),
            fields=[
                ToolField(
                    name="config",
                    description="Configuration to validate",
                    field_type="runtime",
                ),
            ],
            category="validation",
        )
    )

    registry.register(
        ToolSpec(
            name="explain_physics",
            description="Explain gravitational lensing concepts",
            docstring=(
                "Use this tool to provide educational explanations "
                "about dark matter, lensing physics, or parameters."
            ),
            fields=[
                ToolField(
                    name="topic",
                    description="Physics topic to explain",
                    field_type="runtime",
                ),
            ],
            category="education",
        )
    )

    return registry
