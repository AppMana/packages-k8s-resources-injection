import dataclasses

import pytest

from appmana_k8s_resources_injection import (
    PortRange,
    TopologyError,
    canonical_env,
    inject_pod,
    jobset_pod_topology,
    leaderworkerset_pod_topology,
    topology_from_environment,
)

from fixtures import env_of, jobset_pod, lws_pod

# Downward-API env of a JobSet trainer pod, plus the two values the
# downward API cannot expose: the Job's completions and, optionally, a
# non-default subdomain.


def jobset_env(*, job_index=0, completion_index=0, replicas=2, completions=1, name="qwen-image", rjob="trainer"):
    return {
        "JOBSET_NAME": name,
        "JOBSET_REPLICATEDJOB_NAME": rjob,
        "JOBSET_REPLICATEDJOB_REPLICAS": str(replicas),
        "JOBSET_JOB_INDEX": str(job_index),
        "JOB_COMPLETION_INDEX": str(completion_index),
        "JOBSET_JOB_COMPLETIONS": str(completions),
    }


def test_expanded_jobset_env():
    topology = topology_from_environment(jobset_env(job_index=1))
    assert topology.num_nodes == 2
    assert topology.node_rank == 1
    assert topology.master_addr == "qwen-image-trainer-0-0.qwen-image"
    assert topology.master_port == 29500
    assert topology.hosts == ("qwen-image-trainer-0-0.qwen-image", "qwen-image-trainer-1-0.qwen-image")


def test_indexed_jobset_env_ranks_by_completion_index():
    # One Job of four pods: JOBSET_JOB_INDEX (and the global index) is 0 on
    # every pod; the completion index is the rank, and every host differs in
    # its last component.
    topology = topology_from_environment(jobset_env(replicas=1, completions=4, completion_index=2))
    assert topology.num_nodes == 4
    assert topology.node_rank == 2
    assert topology.hosts == tuple(f"qwen-image-trainer-0-{i}.qwen-image" for i in range(4))


def test_completions_come_from_the_argument_when_not_in_env():
    env = jobset_env(replicas=1, completion_index=3)
    del env["JOBSET_JOB_COMPLETIONS"]
    assert topology_from_environment(env, pods_per_job=4).node_rank == 3


def test_missing_completions_is_an_error_not_a_guess():
    env = jobset_env(replicas=1, completion_index=0)
    del env["JOBSET_JOB_COMPLETIONS"]
    with pytest.raises(TopologyError, match="JOBSET_JOB_COMPLETIONS"):
        topology_from_environment(env)


def test_completions_follow_from_a_canonical_node_count():
    env = jobset_env(replicas=2, completion_index=1, job_index=1)
    del env["JOBSET_JOB_COMPLETIONS"]
    env["NNODES"] = "4"
    topology = topology_from_environment(env)
    assert topology.node_rank == 3
    assert len(topology.hosts) == 4


def test_global_replicas_alone_is_not_the_node_count():
    env = jobset_env(job_index=1)
    del env["JOBSET_REPLICATEDJOB_REPLICAS"]
    env["JOBSET_GLOBAL_REPLICAS"] = "2"
    with pytest.raises(TopologyError, match="JOBSET_REPLICATEDJOB_REPLICAS"):
        topology_from_environment(env)


def test_subdomain_env():
    env = jobset_env() | {"JOBSET_SUBDOMAIN": "net"}
    assert topology_from_environment(env).master_addr == "qwen-image-trainer-0-0.net"


def test_launcher_and_webhook_agree_for_jobset_pods():
    for job_index, completion_index, replicas, completions in ((1, 0, 2, 1), (0, 2, 1, 3), (1, 1, 2, 2)):
        meta, spec = jobset_pod(
            replicas=replicas, job_index=job_index, completion_index=completion_index
        )
        webhook = jobset_pod_topology(meta, spec, pods_per_job=completions)
        launcher = topology_from_environment(
            jobset_env(job_index=job_index, completion_index=completion_index, replicas=replicas, completions=completions)
        )
        assert launcher == dataclasses.replace(webhook, rendezvous_identity=None)


def test_launcher_reads_what_the_webhook_injected():
    meta, spec = jobset_pod(replicas=2, job_index=1)
    webhook = jobset_pod_topology(meta, spec, pods_per_job=1, port=PortRange(20000, 6000))
    container = inject_pod({"metadata": meta, "spec": spec}, webhook).object["spec"]["containers"][0]
    launcher = topology_from_environment(env_of(container) | jobset_env(job_index=1))
    assert (launcher.master_addr, launcher.master_port, launcher.node_rank, launcher.num_nodes) == (
        webhook.master_addr,
        webhook.master_port,
        webhook.node_rank,
        webhook.num_nodes,
    )
    assert launcher.hosts == webhook.hosts


def test_canonical_env_alone_is_enough():
    env = {"MASTER_ADDR": "10.0.0.4", "MASTER_PORT": "23456", "NODE_RANK": "1", "NNODES": "2"}
    topology = topology_from_environment(env)
    assert (topology.master_addr, topology.master_port, topology.node_rank, topology.num_nodes) == (
        "10.0.0.4",
        23456,
        1,
        2,
    )
    assert topology.hosts == ()


def test_pet_names_are_read_too():
    env = {"PET_MASTER_ADDR": "10.0.0.4", "PET_MASTER_PORT": "23456", "PET_NODE_RANK": "0", "PET_NNODES": "1"}
    assert topology_from_environment(env).master_addr == "10.0.0.4"


def test_disagreeing_plain_and_pet_values_are_an_error():
    env = jobset_env() | {"MASTER_PORT": "23456", "PET_MASTER_PORT": "23457"}
    with pytest.raises(TopologyError, match="MASTER_PORT"):
        topology_from_environment(env)


def test_elastic_node_range_is_rejected():
    with pytest.raises(TopologyError, match="elastic"):
        topology_from_environment(jobset_env() | {"PET_NNODES": "1:4"})


def test_manifest_values_win_over_derived_ones():
    topology = topology_from_environment(jobset_env(job_index=1) | {"MASTER_ADDR": "10.0.0.4"})
    assert topology.master_addr == "10.0.0.4"
    assert topology.hosts == ()


def test_lws_env():
    meta, spec = lws_pod(group_index=1, worker_index=2)
    env = {"LWS_LEADER_ADDRESS": "vllm-1.vllm.inference", "LWS_GROUP_SIZE": "4", "LWS_WORKER_INDEX": "2"}
    topology = topology_from_environment(env)
    webhook = leaderworkerset_pod_topology(meta, spec)
    assert (topology.master_addr, topology.node_rank, topology.num_nodes, topology.hosts) == (
        webhook.master_addr,
        webhook.node_rank,
        webhook.num_nodes,
        webhook.hosts,
    )


def test_no_workload_is_a_single_node():
    topology = topology_from_environment({})
    assert (topology.num_nodes, topology.node_rank, topology.master_addr, topology.master_port) == (
        1,
        0,
        "127.0.0.1",
        29500,
    )


def test_partial_canonical_env_without_a_workload_is_an_error():
    with pytest.raises(TopologyError, match="NNODES"):
        topology_from_environment({"MASTER_ADDR": "10.0.0.4"})


def test_processes_per_node():
    assert topology_from_environment(jobset_env(), processes_per_node=2).processes_per_node == 2
    assert topology_from_environment(jobset_env() | {"PET_NPROC_PER_NODE": "4"}).processes_per_node == 4
    assert topology_from_environment(jobset_env()).processes_per_node == "auto"
    assert "RANK" not in canonical_env(topology_from_environment(jobset_env(), processes_per_node=2))
