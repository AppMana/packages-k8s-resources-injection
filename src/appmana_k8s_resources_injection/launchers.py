"""Launcher command lines for a topology.

torchrun (torch/distributed/run.py): the default static rendezvous takes
--nnodes, --nproc-per-node, --node-rank, --master-addr and --master-port.

DeepSpeed (deepspeed/launcher/runner.py): --num_nodes, --num_gpus,
--node_rank, --master_addr, --master_port; a multi-node launch reads its
node list from --hostfile (``<host> slots=<n>`` lines), and with --no_ssh
each node runs its own launcher with its --node_rank.
"""

from __future__ import annotations

import os
from typing import Union

from .topology import FieldRef, Topology, TopologyError


def _rank(topology: Topology) -> int:
    if isinstance(topology.node_rank, FieldRef):
        raise TopologyError(
            f"node rank is the field {topology.node_rank.field_path!r}; resolve the topology inside the pod"
        )
    return topology.node_rank


def torchrun_argv(topology: Topology, *, executable: str = "torchrun") -> list[str]:
    return [
        executable,
        f"--nnodes={topology.num_nodes}",
        f"--nproc-per-node={topology.processes_per_node}",
        f"--node-rank={_rank(topology)}",
        f"--master-addr={topology.master_addr}",
        f"--master-port={topology.master_port}",
    ]


def deepspeed_argv(
    topology: Topology,
    *,
    num_gpus: int,
    executable: str = "deepspeed",
    hostfile: Union[str, os.PathLike, None] = None,
    no_ssh: bool = False,
) -> list[str]:
    rank = _rank(topology)
    if topology.num_nodes > 1 and hostfile is None:
        raise TopologyError("a multi-node DeepSpeed launch reads its nodes from a hostfile; pass hostfile")
    argv = [
        executable,
        f"--num_nodes={topology.num_nodes}",
        f"--num_gpus={num_gpus}",
        f"--node_rank={rank}",
        f"--master_addr={topology.master_addr}",
        f"--master_port={topology.master_port}",
    ]
    if hostfile is not None:
        argv.append(f"--hostfile={os.fspath(hostfile)}")
    if no_ssh:
        argv.append("--no_ssh")
    return argv


def deepspeed_hostfile(topology: Topology, *, slots: int) -> str:
    if not topology.hosts:
        raise TopologyError("the topology has no hosts list to write a DeepSpeed hostfile from")
    return "".join(f"{host} slots={slots}\n" for host in topology.hosts)
