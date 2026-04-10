"""User-aware dynamic runner wrapper."""

from __future__ import annotations

from user_dynamic import run_dynamic_prompt


def run_user_aware_dynamic_prompt(**kwargs):
    """Run the dynamic flow using real user-specific history instead of fake seed history."""
    return run_dynamic_prompt(full_text=False, **kwargs)
