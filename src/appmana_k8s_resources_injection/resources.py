"""torchrun's --nproc-per-node from a container's resources, as the Kubeflow
Trainer torch plugin derives numProcPerNode: ``auto`` when the container asks
for a GPU (any resource whose name contains "gpu", requests first, then
limits); otherwise the CPU count (requests, else limits, rounded up as
Quantity.Value() rounds), at least 1."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Mapping, Union

from .topology import TopologyError

_MULTIPLIERS = {
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
    "Pi": 2**50,
    "Ei": 2**60,
    "n": Fraction(1, 10**9),
    "u": Fraction(1, 10**6),
    "m": Fraction(1, 10**3),
    "": 1,
    "k": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "P": 10**15,
    "E": 10**18,
}
# k8s.io/apimachinery/pkg/api/resource quantity grammar.
_QUANTITY = re.compile(r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))(?:[eE]([+-]?[0-9]+)|(Ki|Mi|Gi|Ti|Pi|Ei|n|u|m|k|M|G|T|P|E))?")


def quantity_value(quantity: Union[str, int, float]) -> int:
    """Quantity.Value(): the quantity rounded up to an integer."""
    if isinstance(quantity, bool):
        raise TopologyError(f"not a quantity: {quantity!r}")
    if isinstance(quantity, int):
        return quantity
    text = str(quantity).strip()
    match = _QUANTITY.fullmatch(text)
    if match is None:
        raise TopologyError(f"not a quantity: {quantity!r}")
    number = Fraction(match.group(1))
    if match.group(2) is not None:
        number *= Fraction(10) ** int(match.group(2))
    else:
        number *= _MULTIPLIERS[match.group(3) or ""]
    return math.ceil(number)


def _gpus(resource_list: Mapping) -> int:
    values = {quantity_value(q) for name, q in (resource_list or {}).items() if "gpu" in name.lower()}
    if len(values) > 1:
        raise TopologyError(f"several GPU resources with different counts: {dict(resource_list)}")
    return values.pop() if values else 0


def processes_per_node_from_resources(resources: Union[Mapping, None]) -> Union[int, str]:
    resources = resources or {}
    requests = resources.get("requests") or {}
    limits = resources.get("limits") or {}
    gpus = _gpus(requests) or _gpus(limits)
    if gpus > 0:
        return "auto"
    cpu = 0
    if "cpu" in requests and quantity_value(requests["cpu"]) != 0:
        cpu = quantity_value(requests["cpu"])
    elif "cpu" in limits:
        cpu = quantity_value(limits["cpu"])
    return max(1, cpu)
