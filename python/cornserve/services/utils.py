"""Utilities for CornServe services."""

from __future__ import annotations


def to_strict_k8s_name(name: str) -> str:
    """Normalize a name to be suitable even for the strictest Kubernetes requirements.

    RFC 1035 Label Names are the most restrictive:
    - contain at most 63 characters
    - contain only lowercase alphanumeric characters or '-'
    - start with an alphabetic character
    - end with an alphanumeric character

    Ref: https://kubernetes.io/docs/concepts/overview/working-with-objects/names
    """
    name = name.lower()
    name = "".join(c if c.isalnum() or c == "-" else "-" for c in name)
    name = name.strip("-")
    if not name:
        raise ValueError("Name cannot be empty after normalization.")
    return name[:63]
