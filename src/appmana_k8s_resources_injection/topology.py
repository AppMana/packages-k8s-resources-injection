"""The node-level topology of one distributed workload, computed from what
the owning controller writes on its pods or templates.

A *node* is one pod. The world of a pod is its own workload unit:

* JobSet: the pod's ReplicatedJob. ``replicas`` Jobs of ``completions``
  Indexed pods each, ranked job-major (``job_index * completions +
  completion_index``). Rank 0 is job 0 pod 0, whose stable name is the
  JobSet pod hostname ``<jobset>-<replicatedJob>-0-0.<subdomain>``: the
  format of JobSet's CoordinatorEndpoint and of Kubeflow Trainer's
  PET_MASTER_ADDR (``<trainjob>-node-0-0.<trainjob>``).
* LeaderWorkerSet: one group. ``size`` pods, rank = worker index, rank 0 is
  the leader at ``LWS_LEADER_ADDRESS`` (``<lws>-<group>.<subdomain>.<namespace>``).
* Indexed Job: ``completions`` pods, rank = completion index, rank 0 at
  ``<job>-0.<subdomain>`` when the pod template sets a subdomain.

Nothing here talks to the API server. Values a pod does not carry (a Job's
``spec.completions``, a LeaderWorkerSet's uid) are arguments.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Mapping, Sequence, Union

from . import keys


class TopologyError(ValueError):
    """The inputs do not determine the topology exactly."""


@dataclass(frozen=True)
class FieldRef:
    """A downward-API field path, for values that differ between the pods of
    one template (injected as ``valueFrom.fieldRef``)."""

    field_path: str


@dataclass(frozen=True)
class PortRange:
    """Ports ``start`` .. ``start + size - 1``. A workload's port is
    ``start + int.from_bytes(sha256(identity)[:8], "big") % size``: every pod of
    one workload computes the same port with no coordination, and distinct
    workloads land on distinct ports except with probability 1/size per pair.

    For hostNetwork pods the rendezvous port is a node port, shared with every
    other workload whose rank 0 lands on the same node. A pure function of the
    workload's identity into a finite range cannot be collision-free; that
    needs the set of ports already in use on the node, which is state."""

    start: int
    size: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.size < 1 or self.start + self.size - 1 > 65535:
            raise ValueError(f"port range {self.start}+{self.size} is not within 1-65535")

    def port_for(self, identity: str) -> int:
        digest = hashlib.sha256(identity.encode()).digest()
        return self.start + int.from_bytes(digest[:8], "big") % self.size


Port = Union[int, PortRange]


def _validate_processes_per_node(value: object) -> None:
    if isinstance(value, bool):
        raise ValueError(f"processes_per_node must be an integer or one of {sorted(keys.NPROC_PER_NODE_WORDS)}")
    if isinstance(value, int):
        if value < 1:
            raise ValueError("processes_per_node must be at least 1")
        return
    if value not in keys.NPROC_PER_NODE_WORDS:
        raise ValueError(
            f"processes_per_node must be an integer or one of {sorted(keys.NPROC_PER_NODE_WORDS)}, got {value!r}"
        )


@dataclass(frozen=True)
class Topology:
    """Node-level launch topology.

    ``processes_per_node`` is torchrun's ``--nproc-per-node``: an integer, or
    ``auto``/``cpu``/``gpu``/``xpu``. Per-process env (RANK, WORLD_SIZE,
    LOCAL_RANK, LOCAL_WORLD_SIZE) is derivable only when it is exactly 1.

    ``hosts`` lists every node's address in rank order when known.
    ``rendezvous_identity`` names the workload a derived port came from.
    """

    num_nodes: int
    node_rank: Union[int, FieldRef]
    master_addr: str
    master_port: int
    processes_per_node: Union[int, str] = "auto"
    hosts: tuple[str, ...] = ()
    rendezvous_identity: Union[str, None] = None

    def __post_init__(self) -> None:
        if isinstance(self.num_nodes, bool) or not isinstance(self.num_nodes, int) or self.num_nodes < 1:
            raise ValueError(f"num_nodes must be a positive integer, got {self.num_nodes!r}")
        if isinstance(self.node_rank, FieldRef):
            pass
        elif isinstance(self.node_rank, bool) or not isinstance(self.node_rank, int):
            raise ValueError(f"node_rank must be an integer or FieldRef, got {self.node_rank!r}")
        elif not 0 <= self.node_rank < self.num_nodes:
            raise ValueError(f"node_rank {self.node_rank} is outside {self.num_nodes} nodes")
        if not self.master_addr:
            raise ValueError("master_addr is empty")
        if isinstance(self.master_port, bool) or not isinstance(self.master_port, int) or not 1 <= self.master_port <= 65535:
            raise ValueError(f"master_port must be 1-65535, got {self.master_port!r}")
        _validate_processes_per_node(self.processes_per_node)
        object.__setattr__(self, "hosts", tuple(self.hosts))
        if self.hosts and len(self.hosts) != self.num_nodes:
            raise ValueError(f"{len(self.hosts)} hosts for {self.num_nodes} nodes")


def with_rank_ordered_addresses(topology: Topology, addresses: Sequence[str]) -> Topology:
    """Replace the DNS names with addresses (for example node IPs of
    hostNetwork pods) the caller knows to be in rank order.

    An admission list is not necessarily in rank order: Kueue's TAS ungater
    (pkg/controller/tas/topology_ungater.go) assigns pods to the admitted
    domains by rank only when every pod carries its index label and running
    pods match their rank's domain, and otherwise greedily."""
    addresses = tuple(addresses)
    if len(addresses) != topology.num_nodes:
        raise TopologyError(f"{len(addresses)} addresses for {topology.num_nodes} nodes")
    return replace(topology, master_addr=addresses[0], hosts=addresses)


# Metadata access


def _value(metadata: Mapping, key: str) -> Union[str, None]:
    for field in ("labels", "annotations"):
        value = (metadata.get(field) or {}).get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _index(value: Union[str, None], what: str) -> Union[int, None]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        raise TopologyError(f"{what} must be an integer, got {value!r}") from None
    if parsed < 0:
        raise TopologyError(f"{what} must not be negative, got {parsed}")
    return parsed


def _required(metadata: Mapping, key: str) -> str:
    value = _value(metadata, key)
    if value is None:
        raise TopologyError(f"pod has no {key} label or annotation")
    return value


def _required_index(metadata: Mapping, key: str) -> int:
    return _index(_required(metadata, key), key)


def _resolve_port(port: Port, identity: Union[str, None], needs: str) -> int:
    if isinstance(port, PortRange):
        if identity is None:
            raise TopologyError(f"a port range needs {needs}")
        return port.port_for(identity)
    if isinstance(port, bool) or not isinstance(port, int):
        raise TypeError(f"port must be an int or PortRange, got {port!r}")
    return port


def _check_hostname(hostname: str) -> None:
    # The Job controller sets spec.hostname = <job-name>-<completion-index>
    # without truncation; JobSet's validating webhook rejects JobSets whose
    # generated Job or pod names exceed a 63-character DNS-1035 label.
    if len(hostname) > 63:
        raise TopologyError(
            f"hostname {hostname!r} is {len(hostname)} characters; a DNS label holds at most 63"
        )


def _positive(value: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TopologyError(f"{what} must be a positive integer, got {value!r}")
    return value


def _local(*, port: int, processes_per_node: Union[int, str], identity: Union[str, None]) -> Topology:
    """A world of one pod: rank 0 is the pod itself, at torchrun's default
    --master-addr."""
    return Topology(
        num_nodes=1,
        node_rank=0,
        master_addr=keys.LOCAL_MASTER_ADDR,
        master_port=port,
        processes_per_node=processes_per_node,
        rendezvous_identity=identity,
    )


# JobSet


def jobset_layout(
    *,
    jobset_name: str,
    replicated_job: str,
    replicas: int,
    pods_per_job: int,
    subdomain: str,
    job_index: int = 0,
    completion_index: int = 0,
) -> tuple[int, int, tuple[str, ...]]:
    """(num_nodes, node_rank, rank-ordered hosts) of one ReplicatedJob."""
    _positive(replicas, "replicas")
    _positive(pods_per_job, "pods_per_job (the Job's spec.completions)")
    if not 0 <= job_index < replicas:
        raise TopologyError(f"job index {job_index} is outside replicas={replicas}")
    if not 0 <= completion_index < pods_per_job:
        raise TopologyError(f"completion index {completion_index} is outside pods_per_job={pods_per_job}")
    _check_hostname(f"{jobset_name}-{replicated_job}-{replicas - 1}-{pods_per_job - 1}")
    hosts = tuple(
        f"{jobset_name}-{replicated_job}-{job}-{pod}.{subdomain}"
        for job in range(replicas)
        for pod in range(pods_per_job)
    )
    return replicas * pods_per_job, job_index * pods_per_job + completion_index, hosts


def jobset_rendezvous_identity(metadata: Mapping) -> Union[str, None]:
    """``<jobset-uid>/<replicatedJob>/<restart-attempt>``: one rendezvous per
    ReplicatedJob per JobSet restart, from labels the JobSet controller writes
    on the pod template (present at pod CREATE whatever the webhook order)."""
    uid = _value(metadata, keys.JOBSET_UID)
    replicated_job = _value(metadata, keys.JOBSET_REPLICATED_JOB_NAME)
    restart = _value(metadata, keys.JOBSET_RESTART_ATTEMPT)
    if uid is None or replicated_job is None or restart is None:
        return None
    return f"{uid}/{replicated_job}/{restart}"


def jobset_pod_topology(
    metadata: Mapping,
    spec: Mapping,
    *,
    pods_per_job: int,
    port: Port = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
) -> Topology:
    """Topology of a JobSet pod. ``pods_per_job`` is the Job's
    ``spec.completions``; pods do not carry it."""
    jobset_name = _required(metadata, keys.JOBSET_NAME)
    replicated_job = _required(metadata, keys.JOBSET_REPLICATED_JOB_NAME)
    replicas = _required_index(metadata, keys.JOBSET_REPLICATED_JOB_REPLICAS)
    job_index = _required_index(metadata, keys.JOBSET_JOB_INDEX)
    identity = jobset_rendezvous_identity(metadata)
    port_needs = "the JobSet uid, replicatedjob-name and restart-attempt labels"
    completion_index = _index(_value(metadata, keys.JOB_COMPLETION_INDEX), keys.JOB_COMPLETION_INDEX)
    subdomain = (spec or {}).get("subdomain")
    if replicas * pods_per_job == 1 and (completion_index is None or not subdomain):
        return _local(port=_resolve_port(port, identity, port_needs), processes_per_node=processes_per_node, identity=identity)
    if completion_index is None:
        raise TopologyError(f"pod has no {keys.JOB_COMPLETION_INDEX}: its Job is not in Indexed completion mode")
    if not subdomain:
        raise TopologyError(
            "pod has no spec.subdomain: the JobSet disables DNS hostnames, so rank 0 has no stable name"
        )
    num_nodes, node_rank, hosts = jobset_layout(
        jobset_name=jobset_name,
        replicated_job=replicated_job,
        replicas=replicas,
        pods_per_job=pods_per_job,
        subdomain=subdomain,
        job_index=job_index,
        completion_index=completion_index,
    )
    return Topology(
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=hosts[0],
        master_port=_resolve_port(port, identity, port_needs),
        processes_per_node=processes_per_node,
        hosts=hosts,
        rendezvous_identity=identity,
    )


def jobset_replicated_job_topology(
    jobset: Mapping,
    replicated_job: str,
    *,
    port: int = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
) -> Topology:
    """Template-level topology of one ReplicatedJob of a JobSet object.

    The node rank differs per pod, so it is a downward-API field: the
    ``jobset.sigs.k8s.io/job-index`` annotation when each Job has one pod, or
    the ``batch.kubernetes.io/job-completion-index`` annotation (Kubeflow
    Trainer's PET_NODE_RANK field path) when there is one Job. Jobs of several
    pods each, several times over, have no single-field rank."""
    metadata = jobset.get("metadata") or {}
    spec = jobset.get("spec") or {}
    name = metadata.get("name")
    if not name:
        raise TopologyError("JobSet has no metadata.name (generateName only), so its pods have no stable name yet")
    network = spec.get("network") or {}
    if network.get("enableDNSHostnames") is False:
        raise TopologyError("JobSet sets spec.network.enableDNSHostnames=false: no pod DNS names")
    subdomain = network.get("subdomain") or name
    template = next((r for r in spec.get("replicatedJobs") or [] if r.get("name") == replicated_job), None)
    if template is None:
        raise TopologyError(f"JobSet {name} has no replicatedJob {replicated_job!r}")
    replicas = template.get("replicas", 1)
    job_spec = (template.get("template") or {}).get("spec") or {}
    if job_spec.get("completionMode") != "Indexed":
        raise TopologyError(f"replicatedJob {replicated_job!r} is not Indexed: its pods have no hostnames")
    completions = job_spec.get("completions")
    if completions is None:
        raise TopologyError(f"replicatedJob {replicated_job!r} has no spec.completions")
    num_nodes, _, hosts = jobset_layout(
        jobset_name=name,
        replicated_job=replicated_job,
        replicas=replicas,
        pods_per_job=completions,
        subdomain=subdomain,
    )
    if num_nodes == 1:
        node_rank: Union[int, FieldRef] = 0
    elif completions == 1:
        node_rank = FieldRef(f"metadata.annotations['{keys.JOBSET_JOB_INDEX}']")
    elif replicas == 1:
        node_rank = FieldRef(f"metadata.annotations['{keys.JOB_COMPLETION_INDEX}']")
    else:
        raise TopologyError(
            f"replicatedJob {replicated_job!r} has replicas={replicas} Jobs of completions={completions} pods: "
            "the node rank job_index*completions+completion_index is no single field; inject at pod admission"
        )
    return Topology(
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=hosts[0],
        master_port=_resolve_port(port, None, "a pod's workload identity; template injection takes a fixed port"),
        processes_per_node=processes_per_node,
        hosts=hosts,
    )


# LeaderWorkerSet

# LWS statefulset_utils.GetParentNameAndOrdinal.
_STATEFUL_POD = re.compile(r"(.*)-([0-9]+)$")


def _parent_and_ordinal(name: str) -> tuple[str, int]:
    match = _STATEFUL_POD.match(name)
    if match is None or int(match.group(2)) > 2**31 - 1:
        raise TopologyError(f"pod name {name!r} has no StatefulSet ordinal")
    return match.group(1), int(match.group(2))


def lws_hosts(*, leader_address: str, size: int) -> tuple[str, ...]:
    """Rank-ordered addresses of one LWS group: the leader, then worker pods
    ``<leader-pod>-<w>`` in the same subdomain (the worker StatefulSet's
    serviceName is the LWS name, or the leader pod name under
    UniquePerReplica, which is also the leader's subdomain)."""
    leader, separator, domain = leader_address.partition(".")
    if not separator:
        raise TopologyError(f"leader address {leader_address!r} has no subdomain")
    return (leader_address,) + tuple(f"{leader}-{worker}.{domain}" for worker in range(1, size))


def leaderworkerset_pod_topology(
    metadata: Mapping,
    spec: Mapping,
    *,
    lws_uid: Union[str, None] = None,
    port: Port = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
) -> Topology:
    """Topology of a LeaderWorkerSet pod, with or without the labels LWS's own
    pod webhook adds (group-index on leaders, worker-index on workers): absent
    ones come from the StatefulSet pod name, as LWS derives them.

    The derived port's identity is ``<lws_uid>/<group-index>`` (one rendezvous
    per group); pods do not carry the LeaderWorkerSet uid."""
    lws_name = _required(metadata, keys.LWS_NAME)
    pod_name = metadata.get("name")
    if not pod_name:
        raise TopologyError("LeaderWorkerSet pod has no metadata.name")
    namespace = metadata.get("namespace")
    if not namespace:
        raise TopologyError("LeaderWorkerSet pod has no metadata.namespace")
    size = _index(_value(metadata, keys.LWS_SIZE), keys.LWS_SIZE)
    if size is None:
        raise TopologyError(f"pod has no {keys.LWS_SIZE} annotation")
    _positive(size, keys.LWS_SIZE)
    worker_index = _index((metadata.get("labels") or {}).get(keys.LWS_WORKER_INDEX), keys.LWS_WORKER_INDEX)
    if worker_index is None:
        worker_index = _parent_and_ordinal(pod_name)[1]
    leader = worker_index == 0
    group_index = _index((metadata.get("labels") or {}).get(keys.LWS_GROUP_INDEX), keys.LWS_GROUP_INDEX)
    if group_index is None:
        parent, ordinal = _parent_and_ordinal(pod_name)
        group_index = ordinal if leader else _parent_and_ordinal(parent)[1]
    if worker_index >= size:
        raise TopologyError(f"worker index {worker_index} is outside group size {size}")
    unique = (metadata.get("annotations") or {}).get(keys.LWS_SUBDOMAIN_POLICY) == keys.LWS_SUBDOMAIN_UNIQUE_PER_REPLICA
    if leader and unique:
        subdomain = pod_name
    else:
        subdomain = (spec or {}).get("subdomain")
        if not subdomain:
            raise TopologyError("LeaderWorkerSet pod has no spec.subdomain")
    leader_address = f"{lws_name}-{group_index}.{subdomain}.{namespace}"
    hosts = lws_hosts(leader_address=leader_address, size=size)
    identity = f"{lws_uid}/{group_index}" if lws_uid else None
    return Topology(
        num_nodes=size,
        node_rank=worker_index,
        master_addr=leader_address,
        master_port=_resolve_port(port, identity, "the LeaderWorkerSet uid (lws_uid)"),
        processes_per_node=processes_per_node,
        hosts=hosts,
        rendezvous_identity=identity,
    )


def leaderworkerset_topology(
    lws: Mapping,
    *,
    port: int = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
) -> Topology:
    """Template-level topology of a LeaderWorkerSet object: rank 0 is
    ``$(LWS_LEADER_ADDRESS)``, which LWS prepends to every container's env so
    the reference expands, and the node rank is the worker-index label."""
    template = ((lws.get("spec") or {}).get("leaderWorkerTemplate")) or {}
    size = _positive(template.get("size", 1), "leaderWorkerTemplate.size")
    return Topology(
        num_nodes=size,
        node_rank=FieldRef(f"metadata.labels['{keys.LWS_WORKER_INDEX}']") if size > 1 else 0,
        master_addr=f"$({keys.LWS_LEADER_ADDRESS_ENV})",
        master_port=_resolve_port(port, None, "a pod's workload identity; template injection takes a fixed port"),
        processes_per_node=processes_per_node,
    )


# Indexed Job


def job_pod_topology(
    metadata: Mapping,
    spec: Mapping,
    *,
    completions: int,
    port: Port = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
) -> Topology:
    """Topology of a pod of a plain Indexed Job. Pods have hostnames
    ``<job>-<index>``; they resolve when ``spec.subdomain`` names a headless
    Service selecting them. The derived port's identity is the Job uid."""
    job_name = _required(metadata, keys.JOB_NAME)
    _positive(completions, "completions")
    identity = _value(metadata, keys.JOB_CONTROLLER_UID)
    port_needs = f"the {keys.JOB_CONTROLLER_UID} label"
    completion_index = _index(_value(metadata, keys.JOB_COMPLETION_INDEX), keys.JOB_COMPLETION_INDEX)
    subdomain = (spec or {}).get("subdomain")
    if completions == 1 and (completion_index is None or not subdomain):
        return _local(port=_resolve_port(port, identity, port_needs), processes_per_node=processes_per_node, identity=identity)
    if completion_index is None:
        raise TopologyError(f"pod has no {keys.JOB_COMPLETION_INDEX}: its Job is not in Indexed completion mode")
    if not subdomain:
        raise TopologyError("Job pod has no spec.subdomain, so rank 0 has no DNS name")
    if completion_index >= completions:
        raise TopologyError(f"completion index {completion_index} is outside completions={completions}")
    _check_hostname(f"{job_name}-{completions - 1}")
    hosts = tuple(f"{job_name}-{index}.{subdomain}" for index in range(completions))
    return Topology(
        num_nodes=completions,
        node_rank=completion_index,
        master_addr=hosts[0],
        master_port=_resolve_port(port, identity, port_needs),
        processes_per_node=processes_per_node,
        hosts=hosts,
        rendezvous_identity=identity,
    )


def job_topology(
    job: Mapping,
    *,
    port: int = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
) -> Topology:
    """Template-level topology of an Indexed Job object."""
    name = (job.get("metadata") or {}).get("name")
    if not name:
        raise TopologyError("Job has no metadata.name, so its pods have no stable name yet")
    spec = job.get("spec") or {}
    if spec.get("completionMode") != "Indexed":
        raise TopologyError(f"Job {name} is not Indexed: its pods have no hostnames")
    completions = _positive(spec.get("completions"), "Job spec.completions")
    subdomain = ((spec.get("template") or {}).get("spec") or {}).get("subdomain")
    if not subdomain:
        raise TopologyError(f"Job {name} pod template has no spec.subdomain, so rank 0 has no DNS name")
    _check_hostname(f"{name}-{completions - 1}")
    hosts = tuple(f"{name}-{index}.{subdomain}" for index in range(completions))
    return Topology(
        num_nodes=completions,
        node_rank=FieldRef(f"metadata.annotations['{keys.JOB_COMPLETION_INDEX}']") if completions > 1 else 0,
        master_addr=hosts[0],
        master_port=_resolve_port(port, None, "a pod's workload identity; template injection takes a fixed port"),
        processes_per_node=processes_per_node,
        hosts=hosts,
    )


# Dispatch


def pod_topology(
    metadata: Mapping,
    spec: Mapping,
    *,
    pods_per_job: Union[int, None] = None,
    lws_uid: Union[str, None] = None,
    port: Port = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
) -> Union[Topology, None]:
    """Topology of a pod owned by a JobSet, LeaderWorkerSet or Indexed Job, or
    None for a pod owned by none of them. ``pods_per_job`` is the owning Job's
    ``spec.completions`` (JobSet and plain Job pods)."""
    if _value(metadata, keys.JOBSET_NAME) is not None:
        if pods_per_job is None:
            raise TopologyError("JobSet pod: pods_per_job (the Job's spec.completions) is required")
        return jobset_pod_topology(
            metadata, spec, pods_per_job=pods_per_job, port=port, processes_per_node=processes_per_node
        )
    if _value(metadata, keys.LWS_NAME) is not None:
        return leaderworkerset_pod_topology(
            metadata, spec, lws_uid=lws_uid, port=port, processes_per_node=processes_per_node
        )
    if _value(metadata, keys.JOB_NAME) is not None:
        if pods_per_job is None:
            raise TopologyError("Job pod: pods_per_job (the Job's spec.completions) is required")
        return job_pod_topology(
            metadata, spec, completions=pods_per_job, port=port, processes_per_node=processes_per_node
        )
    return None
