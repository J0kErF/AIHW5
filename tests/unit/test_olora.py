"""Numerical tests for the OLoRA orthonormal initializer."""

from __future__ import annotations

import pytest
import torch

from localforge.finetune.olora import orthonormal_


def test_tall_matrix_has_orthonormal_columns() -> None:
    t = torch.zeros(8, 4)
    orthonormal_(t)
    gram = t.t() @ t
    assert torch.allclose(gram, torch.eye(4), atol=1e-5)


def test_wide_matrix_has_orthonormal_rows() -> None:
    t = torch.zeros(4, 8)
    orthonormal_(t)
    gram = t @ t.t()
    assert torch.allclose(gram, torch.eye(4), atol=1e-5)


def test_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        orthonormal_(torch.zeros(5))
