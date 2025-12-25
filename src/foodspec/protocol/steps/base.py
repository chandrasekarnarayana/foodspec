"""
Base Step class for protocol execution.

All protocol steps inherit from this base class and implement the run() method.
"""

from typing import Any, Dict


class Step:
    """Base class for protocol steps."""

    name: str = "base_step"

    def run(self, ctx: Dict[str, Any]):
        """Execute the step with given context.

        Parameters
        ----------
        ctx : Dict[str, Any]
            Execution context containing data, logs, metadata, etc.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError
