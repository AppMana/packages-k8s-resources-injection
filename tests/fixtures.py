"""Pod metadata and specs as the owning controllers create them.

Each builder returns what a mutating admission webhook sees at pod CREATE:
labels and annotations the workload controller wrote into the pod template,
plus what the Job or StatefulSet controller adds to the pod it constructs.
Labels that another controller's own pod webhook adds afterwards (LWS's
group-index on leaders and worker-index on workers) are optional, because
webhook ordering across MutatingWebhookConfigurations is not guaranteed.
"""

from __future__ import annotations

JOBSET_UID_A = "848873cb-4b8a-4fd3-87b0-c14a00ee5fdd"
JOBSET_UID_B = "0f6c5f0e-2a8e-4c43-9a51-7d2d3c1b9e10"
LWS_UID = "5a3b9c1d-8e7f-4a6b-9c0d-1e2f3a4b5c6d"
JOB_UID = "e92e0cf5-d940-43b5-8fe6-b6af6e6a17fe"


def jobset_pod(
    *,
    jobset: str = "qwen-image",
    replicated_job: str = "trainer",
    uid: str = JOBSET_UID_A,
    replicas: int = 2,
    job_index: int = 0,
    completion_index: int = 0,
    global_replicas: int | None = None,
    restart_attempt: int = 0,
    subdomain: str | None = None,
    namespace: str = "training",
) -> tuple[dict, dict]:
    """JobSet v0.11.1 labelAndAnnotateObject() writes these keys as both
    labels and annotations on the Job's pod template; the Job controller
    adds the completion index (annotation, and label with PodIndexLabel),
    job-name and controller-uid, and sets hostname and subdomain."""
    job_name = f"{jobset}-{replicated_job}-{job_index}"
    keys = {
        "jobset.sigs.k8s.io/jobset-name": jobset,
        "jobset.sigs.k8s.io/jobset-uid": uid,
        "jobset.sigs.k8s.io/replicatedjob-name": replicated_job,
        "jobset.sigs.k8s.io/replicatedjob-replicas": str(replicas),
        "jobset.sigs.k8s.io/global-replicas": str(global_replicas if global_replicas is not None else replicas),
        "jobset.sigs.k8s.io/job-index": str(job_index),
        "jobset.sigs.k8s.io/job-global-index": str(job_index),
        "jobset.sigs.k8s.io/restart-attempt": str(restart_attempt),
    }
    labels = dict(keys)
    labels.update(
        {
            "batch.kubernetes.io/job-completion-index": str(completion_index),
            "batch.kubernetes.io/controller-uid": f"{job_name}-uid",
            "batch.kubernetes.io/job-name": job_name,
        }
    )
    annotations = dict(keys)
    annotations["batch.kubernetes.io/job-completion-index"] = str(completion_index)
    metadata = {
        "generateName": f"{job_name}-{completion_index}-",
        "namespace": namespace,
        "labels": labels,
        "annotations": annotations,
    }
    spec = {
        "hostname": f"{job_name}-{completion_index}",
        "subdomain": subdomain if subdomain is not None else jobset,
        "containers": [{"name": "trainer", "image": "example.invalid/trainer"}],
    }
    return metadata, spec


def lws_pod(
    *,
    name: str = "vllm",
    group_index: int = 0,
    worker_index: int = 0,
    size: int = 4,
    namespace: str = "inference",
    after_lws_webhook: bool = False,
    unique_per_replica: bool = False,
) -> tuple[dict, dict]:
    """LWS v0.8.0 pods.

    Leader (worker_index 0): created by StatefulSet <name> with ordinal
    group_index, so its name is <name>-<group>; the leader template carries
    name and worker-index="0" labels and the size annotation (and the
    subdomainPolicy annotation under UniquePerReplica). LWS's pod webhook
    adds group-index from the name ordinal.

    Worker: created by StatefulSet <name>-<group> with ordinals from 1, so
    its name is <name>-<group>-<worker>; the worker template carries
    group-index, name and group-key labels and size and leader-name
    annotations. LWS's pod webhook adds worker-index from the name ordinal.

    The StatefulSet controller sets spec.subdomain to the StatefulSet
    serviceName: <name> for the leader StatefulSet, and for workers <name>
    under SubdomainShared or the leader pod name under UniquePerReplica.
    """
    leader_name = f"{name}-{group_index}"
    labels = {"leaderworkerset.sigs.k8s.io/name": name}
    annotations = {"leaderworkerset.sigs.k8s.io/size": str(size)}
    if worker_index == 0:
        pod_name = leader_name
        labels["leaderworkerset.sigs.k8s.io/worker-index"] = "0"
        if unique_per_replica:
            annotations["leaderworkerset.sigs.k8s.io/subdomainPolicy"] = "UniquePerReplica"
        if after_lws_webhook:
            labels["leaderworkerset.sigs.k8s.io/group-index"] = str(group_index)
        subdomain = leader_name if (after_lws_webhook and unique_per_replica) else name
    else:
        pod_name = f"{leader_name}-{worker_index}"
        labels["leaderworkerset.sigs.k8s.io/group-index"] = str(group_index)
        labels["leaderworkerset.sigs.k8s.io/group-key"] = "0f1e2d3c"
        annotations["leaderworkerset.sigs.k8s.io/leader-name"] = leader_name
        if after_lws_webhook:
            labels["leaderworkerset.sigs.k8s.io/worker-index"] = str(worker_index)
        subdomain = leader_name if unique_per_replica else name
    metadata = {
        "name": pod_name,
        "namespace": namespace,
        "labels": labels,
        "annotations": annotations,
    }
    spec = {
        "hostname": pod_name,
        "subdomain": subdomain,
        "containers": [{"name": "vllm", "image": "example.invalid/vllm"}],
    }
    return metadata, spec


def job_pod(
    *,
    job: str = "sweep",
    uid: str = JOB_UID,
    completion_index: int = 0,
    subdomain: str | None = "sweep-svc",
    namespace: str = "default",
) -> tuple[dict, dict]:
    """A pod of a plain Indexed Job (kube-controller-manager job_controller)."""
    metadata = {
        "generateName": f"{job}-{completion_index}-",
        "namespace": namespace,
        "labels": {
            "batch.kubernetes.io/controller-uid": uid,
            "batch.kubernetes.io/job-name": job,
            "batch.kubernetes.io/job-completion-index": str(completion_index),
            "controller-uid": uid,
            "job-name": job,
        },
        "annotations": {"batch.kubernetes.io/job-completion-index": str(completion_index)},
    }
    spec = {
        "hostname": f"{job}-{completion_index}",
        "containers": [{"name": "main", "image": "example.invalid/main"}],
    }
    if subdomain is not None:
        spec["subdomain"] = subdomain
    return metadata, spec


def env_of(container: dict) -> dict:
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in container.get("env") or []}
