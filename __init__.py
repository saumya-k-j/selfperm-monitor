"""
Self-Permission Monitor
=======================

A chain-of-thought safety monitor that catches manipulation hiding behind
responsible-looking ("self-permission") reasoning -- a blind spot in naive
CoT monitoring that grows with model scale.

Built for the small, open reasoning models deployed where compute is scarce.
"""

from .monitor import (
    NaiveMonitor,
    SelfPermissionMonitor,
    Verdict,
    is_refusal,
    compare,
    groq_backend,
)

__all__ = [
    "NaiveMonitor",
    "SelfPermissionMonitor",
    "Verdict",
    "is_refusal",
    "compare",
    "groq_backend",
]

__version__ = "0.1.0"
