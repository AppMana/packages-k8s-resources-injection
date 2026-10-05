import hashlib

import pytest

from appmana_k8s_resources_injection import (
    PortRange,
    Topology,
    TopologyError,
    jobset_pod_topology,
    job_pod_topology,
    leaderworkerset_pod_topology,
    pod_topology,
    with_rank_ordered_addresses,
)

from fixtures import JOB_UID, JOBSET_UID_A, JOBSET_UID_B, LWS_UID, jobset_pod, job_pod, lws_pod

RANGE = PortRange(start=20000, size=6000)


def range_port(identity: str) -> int:
    digest = hashlib.sha256(identity.encode()).digest()
    return 20000 + int.from_bytes(digest[:8], "big") % 6000


# JobSet


def test_expanded_jobset_rank_is_job_index():
    for job_index in range(4):
        meta, spec = jobset_pod(replicas=4, job_index=job_index)
        topology = jobset_pod_topology(meta, spec, pods_per_job=1)
        assert topology.num_nodes == 4
        assert topology.node_rank == job_index
        assert topology.master_addr == "qwen-image-trainer-0-0.qwen-image"
        assert topology.master_port == 29500


def test_indexed_jobset_rank_is_completion_index():
    for completion_index in range(4):
        meta, spec = jobset_pod(replicas=1, completion_index=completion_index)
        topology = jobset_pod_topology(meta, spec, pods_per_job=4)
        assert topology.num_nodes == 4
        assert topology.node_rank == completion_index
        assert topology.master_addr == "qwen-image-trainer-0-0.qwen-image"


def test_indexed_jobset_hosts_enumerate_completion_indices():
    meta, spec = jobset_pod(replicas=1, completion_index=2)
    topology = jobset_pod_topology(meta, spec, pods_per_job=3)
    assert topology.hosts == (
        "qwen-image-trainer-0-0.qwen-image",
        "qwen-image-trainer-0-1.qwen-image",
        "qwen-image-trainer-0-2.qwen-image",
    )


def test_jobs_of_several_pods_are_job_major():
    meta, spec = jobset_pod(replicas=2, job_index=1, completion_index=1)
    topology = jobset_pod_topology(meta, spec, pods_per_job=3)
    assert topology.num_nodes == 6
    assert topology.node_rank == 4
    assert topology.hosts[topology.node_rank] == "qwen-image-trainer-1-1.qwen-image"


def test_rank_zero_name_is_the_jobset_coordinator_endpoint_format():
    # JobSet CoordinatorEndpoint: <jobset>-<replicatedJob>-<jobIndex>-<podIndex>.<subdomain>;
    # Kubeflow Trainer's PET_MASTER_ADDR is the same name for job 0 pod 0.
    meta, spec = jobset_pod(jobset="probe", replicated_job="rank", subdomain="custom-net", replicas=2, job_index=1)
    topology = jobset_pod_topology(meta, spec, pods_per_job=1)
    assert topology.master_addr == "probe-rank-0-0.custom-net"
    assert topology.hosts == ("probe-rank-0-0.custom-net", "probe-rank-1-0.custom-net")


def test_world_is_the_pods_own_replicated_job():
    meta, spec = jobset_pod(replicas=2, global_replicas=3, job_index=1)
    topology = jobset_pod_topology(meta, spec, pods_per_job=1)
    assert topology.num_nodes == 2


def test_hostname_longer_than_a_dns_label_is_rejected():
    # The Job controller sets hostname <job-name>-<index> untruncated, and JobSet's
    # validating webhook rejects JobSets whose generated names exceed 63 characters.
    # <jobset>-trainer-11-0 with a 51-character JobSet name is 64 characters; the
    # longest name decides even when this pod's own name would fit.
    jobset_name = "j" * 51
    meta, spec = jobset_pod(jobset=jobset_name, replicated_job="trainer", replicas=12, job_index=0)
    with pytest.raises(TopologyError, match="63"):
        jobset_pod_topology(meta, spec, pods_per_job=1)


def test_hostname_of_exactly_63_characters_is_accepted():
    # <jobset>-trainer-1-0 with a 51-character JobSet name is 63 characters.
    jobset_name = "j" * 51
    meta, spec = jobset_pod(jobset=jobset_name, replicated_job="trainer", replicas=2, job_index=1)
    topology = jobset_pod_topology(meta, spec, pods_per_job=1)
    assert len(topology.hosts[-1].split(".")[0]) == 63


def test_completion_index_outside_pods_per_job_is_rejected():
    meta, spec = jobset_pod(replicas=1, completion_index=3)
    with pytest.raises(TopologyError, match="pods_per_job"):
        jobset_pod_topology(meta, spec, pods_per_job=2)


def test_jobset_without_dns_hostnames_has_no_rank_zero_address():
    meta, spec = jobset_pod(replicas=2)
    spec.pop("subdomain")
    with pytest.raises(TopologyError, match="subdomain"):
        jobset_pod_topology(meta, spec, pods_per_job=1)


def test_jobset_keys_are_read_from_annotations_when_labels_lack_them():
    meta, spec = jobset_pod(replicas=2, job_index=1)
    meta["labels"] = {}
    topology = jobset_pod_topology(meta, spec, pods_per_job=1)
    assert topology.node_rank == 1


# JobSet ports


def test_all_pods_of_one_jobset_get_the_same_port():
    expanded = {
        jobset_pod_topology(*jobset_pod(replicas=4, job_index=k), pods_per_job=1, port=RANGE).master_port
        for k in range(4)
    }
    indexed = {
        jobset_pod_topology(*jobset_pod(replicas=1, completion_index=k), pods_per_job=4, port=RANGE).master_port
        for k in range(4)
    }
    assert expanded == {range_port(f"{JOBSET_UID_A}/trainer/0")}
    assert indexed == expanded


def test_two_jobsets_get_different_ports():
    a = jobset_pod_topology(*jobset_pod(uid=JOBSET_UID_A), pods_per_job=1, port=RANGE)
    b = jobset_pod_topology(*jobset_pod(uid=JOBSET_UID_B), pods_per_job=1, port=RANGE)
    assert a.master_port == range_port(f"{JOBSET_UID_A}/trainer/0")
    assert b.master_port == range_port(f"{JOBSET_UID_B}/trainer/0")
    assert a.master_port != b.master_port


def test_a_jobset_restart_moves_the_port():
    first = jobset_pod_topology(*jobset_pod(restart_attempt=0), pods_per_job=1, port=RANGE)
    second = jobset_pod_topology(*jobset_pod(restart_attempt=1), pods_per_job=1, port=RANGE)
    assert second.master_port == range_port(f"{JOBSET_UID_A}/trainer/1")
    assert first.master_port != second.master_port


def test_rendezvous_identity_is_recorded():
    topology = jobset_pod_topology(*jobset_pod(), pods_per_job=1)
    assert topology.rendezvous_identity == f"{JOBSET_UID_A}/trainer/0"


def test_fixed_port_is_used_verbatim():
    topology = jobset_pod_topology(*jobset_pod(), pods_per_job=1, port=29511)
    assert topology.master_port == 29511


def test_port_range_bounds():
    for identity in ("a", "b", JOBSET_UID_A, LWS_UID, JOB_UID):
        assert 20000 <= RANGE.port_for(identity) <= 25999
    with pytest.raises(ValueError):
        PortRange(start=65000, size=1000)
    with pytest.raises(ValueError):
        PortRange(start=0, size=10)


# LeaderWorkerSet


@pytest.mark.parametrize("after_lws_webhook", [False, True])
def test_lws_leader_and_workers(after_lws_webhook):
    leader = leaderworkerset_pod_topology(
        *lws_pod(group_index=1, worker_index=0, after_lws_webhook=after_lws_webhook)
    )
    worker = leaderworkerset_pod_topology(
        *lws_pod(group_index=1, worker_index=3, after_lws_webhook=after_lws_webhook)
    )
    # LWS AddLWSVariables: LWS_LEADER_ADDRESS = <lws>-<group>.<subdomain>.<namespace>
    assert leader.master_addr == worker.master_addr == "vllm-1.vllm.inference"
    assert (leader.num_nodes, leader.node_rank) == (4, 0)
    assert (worker.num_nodes, worker.node_rank) == (4, 3)
    assert worker.hosts == (
        "vllm-1.vllm.inference",
        "vllm-1-1.vllm.inference",
        "vllm-1-2.vllm.inference",
        "vllm-1-3.vllm.inference",
    )
    assert leader.hosts == worker.hosts


@pytest.mark.parametrize("after_lws_webhook", [False, True])
def test_lws_unique_per_replica_subdomain(after_lws_webhook):
    leader = leaderworkerset_pod_topology(
        *lws_pod(group_index=2, worker_index=0, unique_per_replica=True, after_lws_webhook=after_lws_webhook)
    )
    worker = leaderworkerset_pod_topology(
        *lws_pod(group_index=2, worker_index=1, unique_per_replica=True, after_lws_webhook=after_lws_webhook)
    )
    assert leader.master_addr == worker.master_addr == "vllm-2.vllm-2.inference"
    assert worker.hosts[1] == "vllm-2-1.vllm-2.inference"


def test_lws_group_shares_one_port_and_groups_differ():
    leader = leaderworkerset_pod_topology(*lws_pod(group_index=0, worker_index=0), lws_uid=LWS_UID, port=RANGE)
    worker = leaderworkerset_pod_topology(*lws_pod(group_index=0, worker_index=3), lws_uid=LWS_UID, port=RANGE)
    other = leaderworkerset_pod_topology(*lws_pod(group_index=1, worker_index=0), lws_uid=LWS_UID, port=RANGE)
    assert leader.master_port == worker.master_port == range_port(f"{LWS_UID}/0")
    assert other.master_port == range_port(f"{LWS_UID}/1")
    assert other.master_port != leader.master_port


def test_lws_port_range_needs_the_lws_uid():
    with pytest.raises(TopologyError, match="uid"):
        leaderworkerset_pod_topology(*lws_pod(), port=RANGE)


def test_lws_without_size_annotation_is_rejected():
    meta, spec = lws_pod()
    meta["annotations"] = {}
    with pytest.raises(TopologyError, match="size"):
        leaderworkerset_pod_topology(meta, spec)


# Plain indexed Job


def test_plain_indexed_job():
    topology = job_pod_topology(*job_pod(completion_index=2), completions=3, port=RANGE)
    assert topology.num_nodes == 3
    assert topology.node_rank == 2
    assert topology.master_addr == "sweep-0.sweep-svc"
    assert topology.hosts == ("sweep-0.sweep-svc", "sweep-1.sweep-svc", "sweep-2.sweep-svc")
    assert topology.master_port == range_port(JOB_UID)


def test_plain_job_without_subdomain_has_no_rank_zero_address():
    with pytest.raises(TopologyError, match="subdomain"):
        job_pod_topology(*job_pod(subdomain=None), completions=2)


# Dispatch


def test_pod_topology_dispatches_by_owner():
    assert pod_topology(*jobset_pod(replicas=2, job_index=1), pods_per_job=1).node_rank == 1
    assert pod_topology(*lws_pod(worker_index=2)).node_rank == 2
    assert pod_topology(*job_pod(completion_index=1), pods_per_job=2).node_rank == 1


def test_pod_without_an_owning_workload_has_no_topology():
    assert pod_topology({"namespace": "x", "labels": {"app": "web"}}, {"containers": []}) is None


# Rank-ordered address lists


def test_rank_ordered_addresses_replace_dns_names():
    topology = jobset_pod_topology(*jobset_pod(replicas=2, job_index=1), pods_per_job=1)
    by_ip = with_rank_ordered_addresses(topology, ["10.0.0.4", "10.0.0.5"])
    assert by_ip.master_addr == "10.0.0.4"
    assert by_ip.hosts == ("10.0.0.4", "10.0.0.5")
    assert by_ip.node_rank == 1
    with pytest.raises(TopologyError):
        with_rank_ordered_addresses(topology, ["10.0.0.4"])


# Validation


def test_topology_validates_its_fields():
    with pytest.raises(ValueError):
        Topology(num_nodes=2, node_rank=2, master_addr="a", master_port=29500)
    with pytest.raises(ValueError):
        Topology(num_nodes=1, node_rank=0, master_addr="a", master_port=70000)
    with pytest.raises(ValueError):
        Topology(num_nodes=1, node_rank=0, master_addr="a", master_port=1, processes_per_node="many")
