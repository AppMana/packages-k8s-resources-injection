import pytest

from appmana_k8s_resources_injection import (
    FieldRef,
    Topology,
    TopologyError,
    canonical_env,
    deepspeed_argv,
    deepspeed_hostfile,
    env_vars,
    processes_per_node_from_resources,
    torchrun_argv,
)


def two_node(processes_per_node="auto"):
    return Topology(
        num_nodes=2,
        node_rank=1,
        master_addr="qwen-image-trainer-0-0.qwen-image",
        master_port=29500,
        processes_per_node=processes_per_node,
        hosts=("qwen-image-trainer-0-0.qwen-image", "qwen-image-trainer-1-0.qwen-image"),
    )


def test_node_level_env_and_pet_mirror():
    env = canonical_env(two_node())
    assert env == {
        "MASTER_ADDR": "qwen-image-trainer-0-0.qwen-image",
        "MASTER_PORT": "29500",
        "NODE_RANK": "1",
        "NNODES": "2",
        "PET_MASTER_ADDR": "qwen-image-trainer-0-0.qwen-image",
        "PET_MASTER_PORT": "29500",
        "PET_NODE_RANK": "1",
        "PET_NNODES": "2",
        "PET_NPROC_PER_NODE": "auto",
    }


def test_multi_process_pods_never_get_per_process_env():
    for processes in ("auto", "gpu", 2, 8):
        env = canonical_env(two_node(processes))
        assert not {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"} & set(env)


def test_single_process_pods_get_exact_per_process_env():
    env = canonical_env(two_node(1))
    assert env["RANK"] == "1"
    assert env["WORLD_SIZE"] == "2"
    assert env["LOCAL_RANK"] == "0"
    assert env["LOCAL_WORLD_SIZE"] == "1"
    assert env["PET_NPROC_PER_NODE"] == "1"


def test_no_slurm_or_mpi_env():
    env = canonical_env(two_node(1))
    assert not [name for name in env if name.startswith(("SLURM_", "OMPI_", "PMI_", "MV2_", "MPI_"))]


def test_field_ref_rank_becomes_value_from():
    topology = Topology(
        num_nodes=4,
        node_rank=FieldRef("metadata.annotations['batch.kubernetes.io/job-completion-index']"),
        master_addr="a-node-0-0.a",
        master_port=29500,
    )
    entries = {e["name"]: e for e in env_vars(topology)}
    assert entries["PET_NODE_RANK"] == {
        "name": "PET_NODE_RANK",
        "valueFrom": {"fieldRef": {"fieldPath": "metadata.annotations['batch.kubernetes.io/job-completion-index']"}},
    }
    assert entries["NNODES"] == {"name": "NNODES", "value": "4"}


def test_torchrun_argv_is_static_rendezvous():
    assert torchrun_argv(two_node(4)) == [
        "torchrun",
        "--nnodes=2",
        "--nproc-per-node=4",
        "--node-rank=1",
        "--master-addr=qwen-image-trainer-0-0.qwen-image",
        "--master-port=29500",
    ]


def test_deepspeed_argv_for_a_no_ssh_multi_node_launch():
    assert deepspeed_argv(
        two_node(1), num_gpus=1, hostfile="/tmp/hostfile", no_ssh=True
    ) == [
        "deepspeed",
        "--num_nodes=2",
        "--num_gpus=1",
        "--node_rank=1",
        "--master_addr=qwen-image-trainer-0-0.qwen-image",
        "--master_port=29500",
        "--hostfile=/tmp/hostfile",
        "--no_ssh",
    ]


def test_deepspeed_multi_node_needs_a_hostfile():
    with pytest.raises(TopologyError, match="hostfile"):
        deepspeed_argv(two_node(1), num_gpus=1, hostfile=None, no_ssh=True)


def test_deepspeed_hostfile():
    assert deepspeed_hostfile(two_node(), slots=2) == (
        "qwen-image-trainer-0-0.qwen-image slots=2\nqwen-image-trainer-1-0.qwen-image slots=2\n"
    )
    without_hosts = Topology(num_nodes=2, node_rank=0, master_addr="a", master_port=1)
    with pytest.raises(TopologyError, match="hosts"):
        deepspeed_hostfile(without_hosts, slots=1)


def test_launchers_need_a_concrete_rank():
    topology = Topology(num_nodes=2, node_rank=FieldRef("x"), master_addr="a", master_port=1)
    with pytest.raises(TopologyError):
        torchrun_argv(topology)


# Kubeflow Trainer torch plugin: numProcPerNode "auto" unless no GPU is requested,
# then the CPU count (requests, else limits, rounded up), at least 1.


@pytest.mark.parametrize(
    "resources,expected",
    [
        ({"limits": {"nvidia.com/gpu": "2"}}, "auto"),
        ({"requests": {"nvidia.com/gpu": 1}}, "auto"),
        ({"requests": {"cpu": "4"}}, 4),
        ({"requests": {"cpu": "500m"}}, 1),
        ({"requests": {"cpu": "1500m"}}, 2),
        ({"limits": {"cpu": "3"}}, 3),
        ({"requests": {"cpu": "0"}, "limits": {"cpu": "2"}}, 2),
        ({}, 1),
        (None, 1),
    ],
)
def test_processes_per_node_from_resources(resources, expected):
    assert processes_per_node_from_resources(resources) == expected
