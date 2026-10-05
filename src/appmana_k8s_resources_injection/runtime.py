"""Resolve the topology inside a pod, from its environment.

Each fact comes from the canonical env when the manifest or an admission
webhook set it (MASTER_ADDR or PET_MASTER_ADDR, MASTER_PORT or
PET_MASTER_PORT, NODE_RANK or PET_NODE_RANK, NNODES or PET_NNODES), and
otherwise from the workload's own env through the same functions the
webhook side uses, so both compute identical values:

* LeaderWorkerSet: LWS_LEADER_ADDRESS, LWS_GROUP_SIZE, LWS_WORKER_INDEX,
  which LWS injects into every container.
* JobSet: env the manifest maps from the pod's JobSet labels with the
  downward API, plus JOB_COMPLETION_INDEX, which the Job controller injects
  into Indexed Job pods. The Job's completions and a non-default subdomain
  are not on the pod; they are literal env values.

  ========================================  =========================================================
  JOBSET_NAME                               fieldRef metadata.labels['jobset.sigs.k8s.io/jobset-name']
  JOBSET_REPLICATEDJOB_NAME                 fieldRef metadata.labels['jobset.sigs.k8s.io/replicatedjob-name']
  JOBSET_REPLICATEDJOB_REPLICAS             fieldRef metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
  JOBSET_JOB_INDEX                          fieldRef metadata.labels['jobset.sigs.k8s.io/job-index']
  JOB_COMPLETION_INDEX                      set by the Job controller
  JOBSET_JOB_COMPLETIONS                    the replicatedJob's template.spec.completions
  JOBSET_SUBDOMAIN (optional)               spec.network.subdomain, when set
  ========================================  =========================================================

A pod with neither canonical nor workload env is a single node.
"""

from __future__ import annotations

import os
from typing import Mapping, Union

from . import keys
from .topology import Topology, TopologyError, jobset_layout, lws_hosts

JOBSET_NAME_ENV = "JOBSET_NAME"
JOBSET_REPLICATED_JOB_NAME_ENV = "JOBSET_REPLICATEDJOB_NAME"
JOBSET_REPLICATED_JOB_REPLICAS_ENV = "JOBSET_REPLICATEDJOB_REPLICAS"
JOBSET_JOB_INDEX_ENV = "JOBSET_JOB_INDEX"
JOBSET_JOB_COMPLETIONS_ENV = "JOBSET_JOB_COMPLETIONS"
JOBSET_SUBDOMAIN_ENV = "JOBSET_SUBDOMAIN"


def _get(env: Mapping[str, str], name: str) -> Union[str, None]:
    value = env.get(name)
    return value if value not in (None, "") else None


def _int(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise TopologyError(f"{name} must be an integer, got {value!r}") from None


def _pair(env: Mapping[str, str], plain: str, pet: str) -> Union[str, None]:
    a, b = _get(env, plain), _get(env, pet)
    if a is not None and b is not None and a != b:
        raise TopologyError(f"{plain}={a!r} and {pet}={b!r} disagree")
    return a if a is not None else b


def _pair_int(env: Mapping[str, str], plain: str, pet: str) -> Union[int, None]:
    value = _pair(env, plain, pet)
    if value is None:
        return None
    if plain == keys.NNODES and ":" in value:
        raise TopologyError(f"{plain}/{pet}={value!r} is an elastic node range; only a fixed node count is supported")
    return _int(value, plain)


def _required(env: Mapping[str, str], name: str) -> str:
    value = _get(env, name)
    if value is None:
        raise TopologyError(f"{name} is not set")
    return value


def _jobset(env: Mapping[str, str], pods_per_job: Union[int, None], num_nodes: Union[int, None]):
    name = _required(env, JOBSET_NAME_ENV)
    replicated_job = _required(env, JOBSET_REPLICATED_JOB_NAME_ENV)
    replicas = _int(_required(env, JOBSET_REPLICATED_JOB_REPLICAS_ENV), JOBSET_REPLICATED_JOB_REPLICAS_ENV)
    job_index = _int(_required(env, JOBSET_JOB_INDEX_ENV), JOBSET_JOB_INDEX_ENV)
    completion_index = _int(_required(env, keys.JOB_COMPLETION_INDEX_ENV), keys.JOB_COMPLETION_INDEX_ENV)
    if pods_per_job is None and _get(env, JOBSET_JOB_COMPLETIONS_ENV) is not None:
        pods_per_job = _int(env[JOBSET_JOB_COMPLETIONS_ENV], JOBSET_JOB_COMPLETIONS_ENV)
    if pods_per_job is None and num_nodes is not None and num_nodes % replicas == 0:
        pods_per_job = num_nodes // replicas
    if pods_per_job is None:
        raise TopologyError(
            f"the Job's completions are unknown: set {JOBSET_JOB_COMPLETIONS_ENV} to the replicatedJob's "
            "template.spec.completions, or NNODES"
        )
    subdomain = _get(env, JOBSET_SUBDOMAIN_ENV) or name
    num, rank, hosts = jobset_layout(
        jobset_name=name,
        replicated_job=replicated_job,
        replicas=replicas,
        pods_per_job=pods_per_job,
        subdomain=subdomain,
        job_index=job_index,
        completion_index=completion_index,
    )
    return num, rank, hosts[0], hosts


def _lws(env: Mapping[str, str]):
    leader_address = _required(env, keys.LWS_LEADER_ADDRESS_ENV)
    size = _int(_required(env, keys.LWS_GROUP_SIZE_ENV), keys.LWS_GROUP_SIZE_ENV)
    worker_index = _int(_required(env, keys.LWS_WORKER_INDEX_ENV), keys.LWS_WORKER_INDEX_ENV)
    return size, worker_index, leader_address, lws_hosts(leader_address=leader_address, size=size)


def _processes_per_node(value: str) -> Union[int, str]:
    return value if value in keys.NPROC_PER_NODE_WORDS else _int(value, keys.PET_NPROC_PER_NODE)


def topology_from_environment(
    env: Union[Mapping[str, str], None] = None,
    *,
    pods_per_job: Union[int, None] = None,
    processes_per_node: Union[int, str, None] = None,
    default_port: int = keys.DEFAULT_MASTER_PORT,
) -> Topology:
    """The topology of this pod. ``pods_per_job`` overrides
    JOBSET_JOB_COMPLETIONS; ``processes_per_node`` overrides
    PET_NPROC_PER_NODE (default ``auto``); ``default_port`` applies when
    neither MASTER_PORT nor PET_MASTER_PORT is set."""
    env = os.environ if env is None else env
    master_addr = _pair(env, keys.MASTER_ADDR, keys.PET_MASTER_ADDR)
    master_port = _pair_int(env, keys.MASTER_PORT, keys.PET_MASTER_PORT)
    node_rank = _pair_int(env, keys.NODE_RANK, keys.PET_NODE_RANK)
    num_nodes = _pair_int(env, keys.NNODES, keys.PET_NNODES)
    if processes_per_node is None:
        nproc = _get(env, keys.PET_NPROC_PER_NODE)
        processes_per_node = _processes_per_node(nproc) if nproc is not None else "auto"

    complete = None not in (master_addr, master_port, node_rank, num_nodes)
    try:
        if _get(env, keys.LWS_LEADER_ADDRESS_ENV) is not None:
            derived = _lws(env)
        elif _get(env, JOBSET_NAME_ENV) is not None:
            derived = _jobset(env, pods_per_job, num_nodes)
        else:
            derived = None
    except TopologyError:
        # With every canonical fact set, the workload env only adds the hosts list.
        if not complete:
            raise
        derived = None

    if derived is None:
        facts = {
            f"{keys.MASTER_ADDR}/{keys.PET_MASTER_ADDR}": master_addr,
            f"{keys.NODE_RANK}/{keys.PET_NODE_RANK}": node_rank,
            f"{keys.NNODES}/{keys.PET_NNODES}": num_nodes,
        }
        if all(value is None for value in facts.values()):
            return Topology(
                num_nodes=1,
                node_rank=0,
                master_addr=keys.LOCAL_MASTER_ADDR,
                master_port=master_port if master_port is not None else default_port,
                processes_per_node=processes_per_node,
            )
        missing = [name for name, value in facts.items() if value is None]
        if missing:
            raise TopologyError(
                f"{', '.join(missing)} not set, and no JobSet or LeaderWorkerSet env to derive them from"
            )
        return Topology(
            num_nodes=num_nodes,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port if master_port is not None else default_port,
            processes_per_node=processes_per_node,
        )

    derived_nodes, derived_rank, derived_addr, derived_hosts = derived
    final_nodes = num_nodes if num_nodes is not None else derived_nodes
    final_addr = master_addr if master_addr is not None else derived_addr
    # Another address for rank 0 leaves the node list valid; another node count does not.
    hosts = derived_hosts if final_nodes == derived_nodes else ()
    return Topology(
        num_nodes=final_nodes,
        node_rank=node_rank if node_rank is not None else derived_rank,
        master_addr=final_addr,
        master_port=master_port if master_port is not None else default_port,
        processes_per_node=processes_per_node,
        hosts=hosts,
    )
