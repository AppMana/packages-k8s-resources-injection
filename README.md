# appmana-k8s-resources-injection

Computes the node-level distributed-launch topology of a Kubernetes workload
(node count, node rank, rank 0's address, rendezvous port) and maps it to:

- the canonical env: `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, `NNODES` and
  torchrun's `PET_*` set;
- torchrun and DeepSpeed launcher command lines;
- Kubernetes objects. The env is filled into the containers of a Pod, or into
  the pod templates of a Job, JobSet or LeaderWorkerSet. A name the manifest
  already defines is never overridden.

It supports JobSet (one pod per Job, or Indexed Jobs of several pods),
LeaderWorkerSet, and plain Indexed Jobs. The same functions run in an
admission webhook (from pod labels) and inside the pod (from its env), so
both sides compute identical values. The package is pure Python with no
dependencies.

## Install

Python 3.10 or newer. Install the wheel from a GitHub release:

```shell
uv pip install "appmana-k8s-resources-injection @ https://github.com/AppMana/packages-k8s-resources-injection/releases/download/v0.1.0/appmana_k8s_resources_injection-0.1.0-py3-none-any.whl"
```

## Env contract

| Name | Value | Set when | Source of the convention |
|---|---|---|---|
| `MASTER_ADDR` | rank 0's address | always | torch.distributed `env://` |
| `MASTER_PORT` | rendezvous port | always | torch.distributed `env://`; default 29500 |
| `NODE_RANK` | this pod's node rank | always | |
| `NNODES` | node count | always | |
| `PET_MASTER_ADDR` | = `MASTER_ADDR` | always | torchrun `--master-addr`; Kubeflow Trainer |
| `PET_MASTER_PORT` | = `MASTER_PORT` | always | torchrun `--master-port`; Kubeflow Trainer |
| `PET_NODE_RANK` | = `NODE_RANK` | always | torchrun `--node-rank`; Kubeflow Trainer |
| `PET_NNODES` | = `NNODES` | always | torchrun `--nnodes`; Kubeflow Trainer |
| `PET_NPROC_PER_NODE` | processes per node, or `auto` | always | torchrun `--nproc-per-node`; Kubeflow Trainer |
| `RANK` | = `NODE_RANK` | one process per node | torch.distributed `env://` |
| `WORLD_SIZE` | = `NNODES` | one process per node | torch.distributed `env://` |
| `LOCAL_RANK` | `0` | one process per node | torch.distributed `env://` |
| `LOCAL_WORLD_SIZE` | `1` | one process per node | torch.distributed `env://` |

torchrun reads every `--x-y` option from `PET_X_Y`. Kubeflow Trainer's torch
plugin sets the `PET_*` names (kubeflow/trainer PR #1840). With more than one
process per node, `RANK`, `WORLD_SIZE`, `LOCAL_RANK` and `LOCAL_WORLD_SIZE`
belong to the launcher, so they are never set. No Slurm or MPI variables are
set.

Topology per workload:

| Workload | Nodes | Node rank | Rank 0 address |
|---|---|---|---|
| JobSet | the pod's ReplicatedJob: `replicas × completions` | `job-index × completions + completion-index` | `<jobset>-<replicatedJob>-0-0.<subdomain>` |
| LeaderWorkerSet | one group: `size` | worker index | `<lws>-<group>.<subdomain>.<namespace>` (`LWS_LEADER_ADDRESS`) |
| Indexed Job | `completions` | completion index | `<job>-0.<subdomain>` |

Rank 0's JobSet address uses the format of JobSet's coordinator endpoint and
of Kubeflow Trainer's `PET_MASTER_ADDR`. A JobSet whose generated pod names
would exceed 63 characters is rejected, as JobSet's own webhook rejects it.

### Rendezvous port

The port is `MASTER_PORT` when set, otherwise the `port` argument (default
29500). Pods on hostNetwork share their node's ports, so give them a
`PortRange`. Each workload then gets
`start + int.from_bytes(sha256(identity)[:8], "big") % size`, where the
identity is:

| Workload | Identity |
|---|---|
| JobSet | `<jobset-uid>/<replicatedJob>/<restart-attempt>` |
| LeaderWorkerSet | `<lws-uid>/<group-index>` |
| Indexed Job | `<job-uid>` |

All pods of one workload agree on the port. Two workloads collide with
probability 1/`size` per pair, which matters only when both rank 0 pods share
a node. A JobSet restart moves the port.

## In an admission webhook

The pod labels carry everything except two values. A JobSet or Job pod
needs its Job's `spec.completions`, and a LeaderWorkerSet port needs the
LeaderWorkerSet's uid. Both come from labels that the owning controller
writes, so they are present at pod CREATE whatever the webhook order.

```python
import logging

import kopf
from kubernetes import client

from appmana_k8s_resources_injection import PortRange, TopologyError, inject_pod, keys, pod_topology

PORTS = PortRange(start=20000, size=6000)  # a range nothing on the nodes binds


def pods_per_job(meta):
    job = (meta.get("labels") or {}).get(keys.JOB_NAME)
    if job is None:
        return None
    return client.BatchV1Api().read_namespaced_job(job, meta["namespace"]).spec.completions


def lws_uid(meta):
    name = (meta.get("labels") or {}).get(keys.LWS_NAME)
    if name is None:
        return None
    lws = client.CustomObjectsApi().get_namespaced_custom_object(
        "leaderworkerset.x-k8s.io", "v1", meta["namespace"], "leaderworkersets", name
    )
    return lws["metadata"]["uid"]


@kopf.on.mutate("v1", "pods", id="distributed-env")
def distributed_env(body, meta, spec, patch, operation, **_):
    if operation != "CREATE":
        return
    try:
        topology = pod_topology(meta, spec, pods_per_job=pods_per_job(meta), lws_uid=lws_uid(meta), port=PORTS)
    except TopologyError as error:
        logging.warning("not injecting distributed env: %s", error)
        return
    if topology is None:
        return
    result = inject_pod(body, topology)
    for field in ("containers", "initContainers"):
        if field in result.object["spec"]:
            patch.spec[field] = result.object["spec"][field]
    for record in result.records:
        logging.info("%s %s %s (%s)", record.container, record.name, record.action, record.detail)
```

- `inject_pod(..., names=("MASTER_PORT",))` fills only the names given;
  `containers=(...)` limits the containers.
- `result.patch` is the same change as an RFC 6902 JSON patch.
- `result.records` lists every decision: `injected`, `kept` (the manifest
  defines it in `env` or possibly via `envFrom`), `linked` (a partner such as
  `MASTER_ADDR` is defined, so `PET_MASTER_ADDR` is set to `$(MASTER_ADDR)`),
  or `skipped` (a partner may come from `envFrom`).
- `with_rank_ordered_addresses(topology, ips)` swaps the DNS names for
  addresses. Use it only with a list you know is in rank order.

### Into workload templates

To inject before the pods exist, for example in a webhook on the workload
kind or when rendering manifests, pass the workload object. Per-pod values
become downward-API fields:

```python
from appmana_k8s_resources_injection import inject_job, inject_jobset, inject_leaderworkerset

result = inject_jobset(jobset)              # every replicatedJob; replicated_jobs=("trainer",) to choose
result = inject_leaderworkerset(lws)        # leaderTemplate and workerTemplate
result = inject_job(job)                    # Indexed Job with spec.template.spec.subdomain
```

A JobSet node rank must be expressible as a single field. With one pod per
Job it is the `jobset.sigs.k8s.io/job-index` annotation. With one Job it is
the `batch.kubernetes.io/job-completion-index` annotation. A ReplicatedJob
with several multi-pod Jobs raises `TopologyError`; inject it at pod
admission instead.

For a LeaderWorkerSet, `MASTER_ADDR` is `$(LWS_LEADER_ADDRESS)` and the rank
is the `worker-index` label. Template injection takes a fixed `port`.

## In the pod

```python
import os

from appmana_k8s_resources_injection import topology_from_environment, torchrun_argv

topology = topology_from_environment(processes_per_node=8)
os.execvp("torchrun", torchrun_argv(topology) + ["train.py"])
```

```python
import os
from pathlib import Path

from appmana_k8s_resources_injection import deepspeed_argv, deepspeed_hostfile, topology_from_environment

gpus = 1
topology = topology_from_environment(processes_per_node=gpus)
hostfile = None
if topology.num_nodes > 1:
    hostfile = Path("/tmp/hostfile")
    hostfile.write_text(deepspeed_hostfile(topology, slots=gpus))
argv = deepspeed_argv(topology, num_gpus=gpus, hostfile=hostfile, no_ssh=topology.num_nodes > 1)
os.execvp(argv[0], argv + ["--module", "my_trainer", "--deepspeed"])
```

`topology_from_environment()` takes each fact from the canonical env when it
is set: by the manifest, or by a webhook using this package. Otherwise it
derives the fact from the workload env:

- **LeaderWorkerSet:** `LWS_LEADER_ADDRESS`, `LWS_GROUP_SIZE` and
  `LWS_WORKER_INDEX`, which LWS sets itself.
- **JobSet:** the env below.

A pod with neither is a single node at `127.0.0.1`.

```yaml
env:
  - name: JOBSET_NAME
    valueFrom: {fieldRef: {fieldPath: "metadata.labels['jobset.sigs.k8s.io/jobset-name']"}}
  - name: JOBSET_REPLICATEDJOB_NAME
    valueFrom: {fieldRef: {fieldPath: "metadata.labels['jobset.sigs.k8s.io/replicatedjob-name']"}}
  - name: JOBSET_REPLICATEDJOB_REPLICAS
    valueFrom: {fieldRef: {fieldPath: "metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']"}}
  - name: JOBSET_JOB_INDEX
    valueFrom: {fieldRef: {fieldPath: "metadata.labels['jobset.sigs.k8s.io/job-index']"}}
  - name: JOBSET_JOB_COMPLETIONS   # the replicatedJob's template.spec.completions
    value: "1"
  # - name: JOBSET_SUBDOMAIN       # only when spec.network.subdomain is set
  #   value: my-subdomain
```

`JOB_COMPLETION_INDEX` is set by the Job controller.
`topology_from_environment(pods_per_job=...)` overrides
`JOBSET_JOB_COMPLETIONS`. When `NNODES` is set, the completions follow from
it.
