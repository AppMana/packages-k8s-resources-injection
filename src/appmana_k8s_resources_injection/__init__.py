"""Canonical distributed-launch topology and env for JobSet, LeaderWorkerSet
and Indexed Job pods, with fill-only-if-absent injection into Kubernetes
objects and in-pod resolution from the same functions."""

from . import keys
from .env import ALL_NAMES, EnvGroup, canonical_env, env_groups, env_var, env_vars
from .inject import (
    InjectionRecord,
    InjectionResult,
    inject_job,
    inject_jobset,
    inject_leaderworkerset,
    inject_pod,
)
from .launchers import deepspeed_argv, deepspeed_hostfile, torchrun_argv
from .resources import processes_per_node_from_resources, quantity_value
from .runtime import topology_from_environment
from .topology import (
    FieldRef,
    PortRange,
    Topology,
    TopologyError,
    job_pod_topology,
    job_topology,
    jobset_layout,
    jobset_pod_topology,
    jobset_rendezvous_identity,
    jobset_replicated_job_topology,
    leaderworkerset_pod_topology,
    leaderworkerset_topology,
    lws_hosts,
    pod_topology,
    with_rank_ordered_addresses,
)

__all__ = [
    "ALL_NAMES",
    "EnvGroup",
    "FieldRef",
    "InjectionRecord",
    "InjectionResult",
    "PortRange",
    "Topology",
    "TopologyError",
    "canonical_env",
    "deepspeed_argv",
    "deepspeed_hostfile",
    "env_groups",
    "env_var",
    "env_vars",
    "inject_job",
    "inject_jobset",
    "inject_leaderworkerset",
    "inject_pod",
    "job_pod_topology",
    "job_topology",
    "jobset_layout",
    "jobset_pod_topology",
    "jobset_rendezvous_identity",
    "jobset_replicated_job_topology",
    "keys",
    "leaderworkerset_pod_topology",
    "leaderworkerset_topology",
    "lws_hosts",
    "pod_topology",
    "processes_per_node_from_resources",
    "quantity_value",
    "topology_from_environment",
    "torchrun_argv",
    "with_rank_ordered_addresses",
]
