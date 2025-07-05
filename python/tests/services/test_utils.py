from __future__ import annotations

import pytest

from cornserve.services.utils import to_strict_k8s_name


def test_to_strict_k8s_name():
    assert to_strict_k8s_name("valid-name") == "valid-name"
    assert to_strict_k8s_name("ValidName") == "validname"
    assert to_strict_k8s_name("valid-name-123") == "valid-name-123"
    assert to_strict_k8s_name("valid--name") == "valid--name"
    assert to_strict_k8s_name("valid.name") == "valid-name"
    assert to_strict_k8s_name("valid_name") == "valid-name"
    assert to_strict_k8s_name("123invalid-start") == "invalid-start"
    assert to_strict_k8s_name("-leading-dash") == "leading-dash"
    assert to_strict_k8s_name("trailing-dash-") == "trailing-dash"
    assert to_strict_k8s_name(".start-dot-") == "start-dot"
    assert to_strict_k8s_name("a" * 64) == "a" * 63
    assert to_strict_k8s_name("a" * 63) == "a" * 63

    with pytest.raises(ValueError):
        to_strict_k8s_name("")
    with pytest.raises(ValueError):
        to_strict_k8s_name("-")
