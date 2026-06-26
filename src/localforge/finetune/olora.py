"""OLoRA orthonormal initialization fallback (docs/SPECIFICATION.md §11.4).

OLoRA (Büyükakyüz, 2024) initializes the LoRA adapter matrices to be orthonormal
(via a QR decomposition) so that, unlike vanilla LoRA's zero/Gaussian init, the
adapter starts well-conditioned and converges faster. PEFT added native OLoRA
init, but support varies by version; when it is unavailable we apply this thin
QR-orthonormal initializer ourselves so the OLoRA path is always demonstrable.
"""

from __future__ import annotations

from typing import Any


def orthonormal_(tensor: Any) -> Any:
    """In-place: replace a 2D tensor's columns with an orthonormal set (QR).

    Returns the same tensor for convenience. ``Q`` from a reduced QR has
    orthonormal columns, i.e. ``Qᵀ Q ≈ I``.
    """
    import torch

    if tensor.dim() != 2:
        raise ValueError(f"orthonormal_ expects a 2D tensor, got shape {tuple(tensor.shape)}")
    rows, cols = tensor.shape
    # QR needs at least as many rows as columns for orthonormal *columns*.
    flip = rows < cols
    work = tensor.t() if flip else tensor
    q, _ = torch.linalg.qr(torch.randn_like(work), mode="reduced")
    result = q.t() if flip else q
    with torch.no_grad():
        tensor.copy_(result)
    return tensor


def apply_olora_init(model: Any) -> int:
    """Orthonormalize every LoRA ``lora_A`` weight in a PEFT model.

    Returns the number of adapter matrices initialized. Used as the fallback when
    PEFT's native OLoRA init is unavailable.
    """
    count = 0
    for name, param in model.named_parameters():
        if "lora_A" in name and param.dim() == 2:
            orthonormal_(param)
            count += 1
    return count
