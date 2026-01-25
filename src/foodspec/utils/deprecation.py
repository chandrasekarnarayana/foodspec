"""Deprecation utilities for FoodSpec.

This module provides utilities for managing deprecated code
during the v1.x → v2.0.0 transition.
"""

import warnings
from functools import wraps
from typing import Callable


def deprecated(
    reason: str,
    version: str = "2.0.0",
    alternative: str | None = None
) -> Callable:
    """Decorator to mark functions/classes as deprecated.
    
    Parameters
    ----------
    reason : str
        Reason for deprecation
    version : str
        Version when feature will be removed
    alternative : str, optional
        Suggested alternative to use
    
    Examples
    --------
    >>> @deprecated("Use new_function instead", alternative="new_function")
    ... def old_function():
    ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"{func.__name__} is deprecated: {reason}"
            if alternative:
                msg += f" Use {alternative} instead."
            msg += f" Will be removed in v{version}."
            
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        
        # Add deprecation marker to docstring
        if wrapper.__doc__:
            wrapper.__doc__ = (
                f".. deprecated:: 1.1.0\n"
                f"    {reason}\n"
                f"    Will be removed in v{version}.\n\n"
                f"{wrapper.__doc__}"
            )
        
        return wrapper
    return decorator


def warn_deprecated_import(
    old_module: str,
    new_module: str,
    version: str = "2.0.0"
):
    """Issue warning for deprecated module import.
    
    Parameters
    ----------
    old_module : str
        Old module name
    new_module : str
        New module name to use
    version : str
        Version when module will be removed
    """
    warnings.warn(
        f"{old_module} is deprecated and will be removed in v{version}. "
        f"Use {new_module} instead. "
        f"See docs/migration/v1-to-v2.md for migration guide.",
        DeprecationWarning,
        stacklevel=3
    )
