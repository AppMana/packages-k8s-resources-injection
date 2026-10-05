"""The canonical env of a topology.

Node level, for every pod:

* MASTER_ADDR, MASTER_PORT: torch.distributed env:// rendezvous.
* NODE_RANK, NNODES.
* PET_MASTER_ADDR, PET_MASTER_PORT, PET_NODE_RANK, PET_NNODES,
  PET_NPROC_PER_NODE: torchrun reads each ``--x-y`` option from ``PET_X_Y``.
  Kubeflow Trainer's torch plugin sets PET_NNODES, PET_NPROC_PER_NODE and
  PET_NODE_RANK, with PET_MASTER_ADDR/PET_MASTER_PORT equal to
  MASTER_ADDR/MASTER_PORT (kubeflow/trainer PR #1840).

Per process, only when ``processes_per_node`` is exactly 1: RANK = node rank,
WORLD_SIZE = node count, LOCAL_RANK = 0, LOCAL_WORLD_SIZE = 1. With more
processes per node these belong to the launcher (torchrun, DeepSpeed) that
starts each process, so they are never emitted. Nothing emulates Slurm or MPI.

Names that carry the same fact form a group, so injection can keep a
manifest's value for one name and make its partners follow it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from . import keys
from .topology import FieldRef, Topology

ALL_NAMES = (
    keys.MASTER_ADDR,
    keys.MASTER_PORT,
    keys.NODE_RANK,
    keys.NNODES,
    keys.PET_MASTER_ADDR,
    keys.PET_MASTER_PORT,
    keys.PET_NODE_RANK,
    keys.PET_NNODES,
    keys.PET_NPROC_PER_NODE,
    keys.RANK,
    keys.WORLD_SIZE,
    keys.LOCAL_RANK,
    keys.LOCAL_WORLD_SIZE,
)


@dataclass(frozen=True)
class EnvGroup:
    """Names that hold one value."""

    names: tuple[str, ...]
    value: Union[str, FieldRef]


def _single_process(topology: Topology) -> bool:
    return not isinstance(topology.processes_per_node, bool) and topology.processes_per_node == 1


def env_groups(topology: Topology) -> tuple[EnvGroup, ...]:
    single = _single_process(topology)
    rank = topology.node_rank if isinstance(topology.node_rank, FieldRef) else str(topology.node_rank)
    groups = [
        EnvGroup((keys.MASTER_ADDR, keys.PET_MASTER_ADDR), topology.master_addr),
        EnvGroup((keys.MASTER_PORT, keys.PET_MASTER_PORT), str(topology.master_port)),
        EnvGroup((keys.NODE_RANK, keys.PET_NODE_RANK) + ((keys.RANK,) if single else ()), rank),
        EnvGroup((keys.NNODES, keys.PET_NNODES) + ((keys.WORLD_SIZE,) if single else ()), str(topology.num_nodes)),
        EnvGroup(
            (keys.PET_NPROC_PER_NODE,) + ((keys.LOCAL_WORLD_SIZE,) if single else ()),
            str(topology.processes_per_node),
        ),
    ]
    if single:
        groups.append(EnvGroup((keys.LOCAL_RANK,), "0"))
    return tuple(groups)


def canonical_env(topology: Topology) -> dict[str, Union[str, FieldRef]]:
    """Name to value, in ALL_NAMES order."""
    values = {name: group.value for group in env_groups(topology) for name in group.names}
    return {name: values[name] for name in ALL_NAMES if name in values}


def env_var(name: str, value: Union[str, FieldRef]) -> dict:
    """A Kubernetes EnvVar."""
    if isinstance(value, FieldRef):
        return {"name": name, "valueFrom": {"fieldRef": {"fieldPath": value.field_path}}}
    return {"name": name, "value": value}


def env_vars(topology: Topology) -> list[dict]:
    """The canonical env as Kubernetes EnvVars, in ALL_NAMES order."""
    return [env_var(name, value) for name, value in canonical_env(topology).items()]
