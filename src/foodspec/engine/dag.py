"""
FoodSpec Pipeline DAG (Directed Acyclic Graph)

Represents execution as a dependency graph.
Ensures proper ordering and enables visualization.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Execution status of a node"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeType(str, Enum):
    """Types of pipeline nodes"""
    PREPROCESS = "preprocess"
    QC = "qc"
    FEATURES = "features"
    MODEL = "model"
    TRUST = "trust"
    VIZ = "visualization"
    REPORT = "reporting"
    VALIDATION = "validation"


@dataclass
class Node:
    """
    Single execution node in pipeline.

    Represents one stage of the workflow.
    """

    name: str                           # Unique identifier
    node_type: NodeType                 # Type of operation
    func: Optional[Callable] = None     # Execution function
    inputs: List[str] = field(default_factory=list)  # Dependency names
    outputs: List[str] = field(default_factory=list) # Output artifact names
    params: Dict[str, Any] = field(default_factory=dict)  # Configuration
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[Any] = None        # Execution result
    error: Optional[str] = None         # Error message if failed

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.name == other.name

    def to_dict(self) -> Dict[str, Any]:
        """Export to dict"""
        return {
            "name": self.name,
            "type": self.node_type.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "params": self.params,
            "status": self.status.value,
            "error": self.error,
        }


@dataclass
class PipelineDAG:
    """
    Directed Acyclic Graph for FoodSpec pipeline.

    Manages:
    - Node registration
    - Dependency resolution
    - Topological ordering
    - Execution orchestration
    """

    nodes: Dict[str, Node] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)

    @property
    def pipeline_dag(self) -> "PipelineDAG":
        """Backwards-compatible alias used in tests."""
        return self

    def add_node(
        self,
        name: str,
        node_type: NodeType,
        func: Optional[Callable] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Node:
        """
        Add node to DAG.

        Args:
            name: Unique node name
            node_type: Type of operation
            func: Execution function
            inputs: List of input dependencies
            outputs: List of output artifact names
            params: Configuration parameters

        Returns:
            Created Node object
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")

        node = Node(
            name=name,
            node_type=node_type,
            func=func,
            inputs=inputs or [],
            outputs=outputs or [],
            params=params or {},
        )

        self.nodes[name] = node
        logger.info(f"Added node: {name} ({node_type.value})")

        return node

    def get_node(self, name: str) -> Optional[Node]:
        """Get node by name"""
        return self.nodes.get(name)

    def topological_sort(self) -> List[str]:
        """
        Compute topological ordering of nodes.

        Returns:
            List of node names in execution order

        Raises:
            ValueError: If cycle detected
        """
        visited: Set[str] = set()
        visiting: Set[str] = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return

            if name in visiting:
                raise ValueError(f"Cycle detected involving node '{name}'")

            visiting.add(name)
            node = self.nodes[name]

            # Visit dependencies first
            for dep in node.inputs:
                if dep not in self.nodes:
                    raise ValueError(f"Dependency '{dep}' not found for node '{name}'")
                visit(dep)

            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in self.nodes:
            visit(name)

        self.execution_order = order
        logger.info(f"Topological sort: {' → '.join(order)}")

        return order

    def validate(self) -> bool:
        """
        Validate DAG consistency.

        Checks:
        - All dependencies exist
        - No cycles
        - All nodes have names

        Returns:
            True if valid

        Raises:
            ValueError: If invalid
        """
        # Check all nodes have names
        for name, node in self.nodes.items():
            if not name or not node.name:
                raise ValueError("Node has empty name")

        # Check all dependencies exist
        for name, node in self.nodes.items():
            for dep in node.inputs:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{name}' depends on nonexistent '{dep}'")

        # Check for cycles
        self.topological_sort()

        logger.info("✓ DAG is valid")
        return True

    def get_execution_order(self) -> List[str]:
        """Get nodes in execution order"""
        if not self.execution_order:
            self.topological_sort()
        return self.execution_order

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all nodes in topological order.

        Args:
            context: Shared execution context

        Returns:
            Updated context with execution results
        """
        logger.info("=" * 70)
        logger.info("PIPELINE EXECUTION")
        logger.info("=" * 70)

        order = self.get_execution_order()
        results = {}

        for node_name in order:
            node = self.nodes[node_name]

            logger.info(f"\n[{order.index(node_name) + 1}/{len(order)}] Running: {node_name}")

            try:
                node.status = NodeStatus.RUNNING

                if node.func is None:
                    logger.warning("  → Node has no function, skipping")
                    node.status = NodeStatus.SKIPPED
                    results[node_name] = None
                    continue

                # Call node function with context
                logger.info(f"  → Executing with params: {node.params}")
                result = node.func(context, **node.params)

                node.result = result
                node.status = NodeStatus.SUCCESS
                results[node_name] = result

                logger.info("  ✓ Success")

            except Exception as e:
                node.status = NodeStatus.FAILED
                node.error = str(e)
                results[node_name] = None

                logger.error(f"  ✗ Failed: {e}")
                raise RuntimeError(f"Pipeline failed at node '{node_name}': {e}")

        logger.info("\n" + "=" * 70)
        logger.info("✓ PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 70)

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Export DAG to dict"""
        return {
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "execution_order": self.execution_order,
        }

    def to_json(self, out_path: Path) -> Path:
        """Save DAG to JSON file"""
        with open(out_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"DAG saved to: {out_path}")
        return out_path

    def to_svg(self, out_path: Path) -> Path:
        """
        Save DAG as SVG visualization.

        Requires graphviz package.
        """
        try:
            import graphviz
        except ImportError:
            logger.warning("graphviz not available; skipping SVG export")
            return None

        dot = graphviz.Digraph(comment="FoodSpec Pipeline DAG")

        # Add nodes
        for name, node in self.nodes.items():
            color = {
                NodeStatus.SUCCESS: "lightgreen",
                NodeStatus.FAILED: "lightcoral",
                NodeStatus.RUNNING: "lightyellow",
                NodeStatus.PENDING: "lightgray",
                NodeStatus.SKIPPED: "lightgray",
            }.get(node.status, "white")

            dot.node(name, label=f"{name}\n({node.node_type.value})", style="filled", fillcolor=color)

        # Add edges
        for name, node in self.nodes.items():
            for dep in node.inputs:
                dot.edge(dep, name)

        dot.render(str(out_path), format="svg", cleanup=True)
        logger.info(f"DAG visualization saved to: {out_path}.svg")
        return Path(str(out_path) + ".svg")


# ============================================================================
# Standard Pipeline Builder
# ============================================================================

def build_standard_pipeline() -> PipelineDAG:
    """
    Build standard FoodSpec pipeline DAG.

    Returns:
        PipelineDAG with standard stages
    """
    dag = PipelineDAG()

    dag.add_node(
        "preprocess",
        NodeType.PREPROCESS,
        inputs=[],
        outputs=["X_processed"],
    )

    dag.add_node(
        "qc",
        NodeType.QC,
        inputs=["preprocess"],
        outputs=["qc_results"],
    )

    dag.add_node(
        "features",
        NodeType.FEATURES,
        inputs=["qc"],
        outputs=["features_table"],
    )

    dag.add_node(
        "model",
        NodeType.MODEL,
        inputs=["features"],
        outputs=["metrics", "predictions"],
    )

    dag.add_node(
        "trust",
        NodeType.TRUST,
        inputs=["model"],
        outputs=["calibration", "conformal", "abstention"],
    )

    dag.add_node(
        "visualization",
        NodeType.VIZ,
        inputs=["trust"],
        outputs=["figures"],
    )

    dag.add_node(
        "reporting",
        NodeType.REPORT,
        inputs=["visualization"],
        outputs=["report_html", "card_json"],
    )

    dag.validate()
    return dag
