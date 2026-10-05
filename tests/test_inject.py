import copy

import jsonpatch
import pytest

from appmana_k8s_resources_injection import (
    PortRange,
    Topology,
    TopologyError,
    inject_job,
    inject_jobset,
    inject_leaderworkerset,
    inject_pod,
    pod_topology,
)

from fixtures import JOB_UID, JOBSET_UID_A, JOBSET_UID_B, LWS_UID, env_of, jobset_pod, job_pod, lws_pod

RANGE = PortRange(start=20000, size=6000)
PORT_ONLY = ("MASTER_PORT",)


def pod(meta, spec):
    return {"apiVersion": "v1", "kind": "Pod", "metadata": meta, "spec": spec}


def with_containers(spec, *, containers=1, init=0):
    spec["containers"] = [{"name": f"c{i}", "image": "x"} for i in range(containers)]
    if init:
        spec["initContainers"] = [{"name": f"i{i}", "image": "x"} for i in range(init)]
    return spec


def topology_of(meta, spec, **kwargs):
    kwargs.setdefault("pods_per_job", 1)
    return pod_topology(meta, spec, port=RANGE, lws_uid=LWS_UID, **kwargs)


def master_port_of(meta, spec, **kwargs):
    result = inject_pod(pod(meta, spec), topology_of(meta, spec, **kwargs), names=PORT_ONLY)
    return env_of(result.object["spec"]["containers"][0])["MASTER_PORT"]


def assert_patch_reproduces(result, original):
    assert jsonpatch.apply_patch(original, result.patch) == result.object


# Fill-only-if-absent, per container and per name


def test_injected_when_absent_in_every_container_and_init_container():
    meta, spec = jobset_pod()
    with_containers(spec, containers=2, init=1)
    original = pod(meta, spec)
    result = inject_pod(original, topology_of(meta, spec))
    expected = str(RANGE.port_for(f"{JOBSET_UID_A}/trainer/0"))
    for field in ("containers", "initContainers"):
        for container in result.object["spec"][field]:
            env = env_of(container)
            assert env["MASTER_PORT"] == env["PET_MASTER_PORT"] == expected
            assert env["MASTER_ADDR"] == "qwen-image-trainer-0-0.qwen-image"
    assert_patch_reproduces(result, original)
    assert "env" not in original["spec"]["containers"][0]


def test_stable_across_pods_of_one_expanded_jobset():
    ports = {master_port_of(*jobset_pod(replicas=4, job_index=k)) for k in range(4)}
    assert ports == {str(RANGE.port_for(f"{JOBSET_UID_A}/trainer/0"))}


def test_stable_across_pods_of_one_indexed_jobset():
    ports = {master_port_of(*jobset_pod(replicas=1, completion_index=k), pods_per_job=4) for k in range(4)}
    assert ports == {str(RANGE.port_for(f"{JOBSET_UID_A}/trainer/0"))}


def test_two_workloads_get_different_ports():
    a = master_port_of(*jobset_pod(uid=JOBSET_UID_A))
    b = master_port_of(*jobset_pod(uid=JOBSET_UID_B))
    assert a != b


def test_lws_group_shares_one_port_and_groups_differ():
    leader = master_port_of(*lws_pod(group_index=0, worker_index=0))
    worker = master_port_of(*lws_pod(group_index=0, worker_index=3))
    other = master_port_of(*lws_pod(group_index=1, worker_index=0))
    assert leader == worker == str(RANGE.port_for(f"{LWS_UID}/0"))
    assert other == str(RANGE.port_for(f"{LWS_UID}/1"))


def test_plain_job_pod_uses_job_uid():
    assert master_port_of(*job_pod(completion_index=1), pods_per_job=2) == str(RANGE.port_for(JOB_UID))


def test_not_injected_without_an_owning_workload():
    assert topology_of({"namespace": "x", "labels": {}}, {"containers": [{"name": "c"}]}) is None


def test_manifest_value_is_kept():
    meta, spec = jobset_pod()
    spec["containers"][0]["env"] = [{"name": "MASTER_PORT", "value": "29511"}]
    result = inject_pod(pod(meta, spec), topology_of(meta, spec), names=PORT_ONLY)
    assert result.object["spec"]["containers"][0]["env"] == [{"name": "MASTER_PORT", "value": "29511"}]
    assert result.patch == []
    [record] = result.records
    assert (record.name, record.action) == ("MASTER_PORT", "kept")


def test_manifest_value_from_field_ref_is_kept():
    meta, spec = jobset_pod()
    ref = {"name": "MASTER_PORT", "valueFrom": {"fieldRef": {"fieldPath": "metadata.annotations['port']"}}}
    spec["containers"][0]["env"] = [ref]
    result = inject_pod(pod(meta, spec), topology_of(meta, spec), names=PORT_ONLY)
    assert result.object["spec"]["containers"][0]["env"] == [ref]


def test_partial_definition_fills_only_the_containers_missing_it():
    meta, spec = jobset_pod()
    with_containers(spec, containers=2)
    spec["containers"][0]["env"] = [{"name": "MASTER_PORT", "value": "29511"}]
    original = pod(meta, spec)
    result = inject_pod(original, topology_of(meta, spec), names=PORT_ONLY)
    assert env_of(result.object["spec"]["containers"][0])["MASTER_PORT"] == "29511"
    assert env_of(result.object["spec"]["containers"][1])["MASTER_PORT"] == str(
        RANGE.port_for(f"{JOBSET_UID_A}/trainer/0")
    )
    assert [r.container for r in result.injected] == ["spec.containers[c1]"]
    assert [r.container for r in result.kept] == ["spec.containers[c0]"]
    assert_patch_reproduces(result, original)


@pytest.mark.parametrize(
    "source",
    [{"secretRef": {"name": "s"}}, {"configMapRef": {"name": "c"}, "prefix": "MASTER_"}],
)
def test_env_from_that_can_define_it_is_not_shadowed(source):
    # An explicit env entry takes precedence over envFrom, so injecting one would
    # override whatever the ConfigMap or Secret provides.
    meta, spec = jobset_pod()
    spec["containers"][0]["envFrom"] = [source]
    result = inject_pod(pod(meta, spec), topology_of(meta, spec), names=PORT_ONLY)
    assert "MASTER_PORT" not in env_of(result.object["spec"]["containers"][0])
    assert result.records[0].detail == "envFrom"


def test_env_from_whose_prefix_cannot_define_it_does_not_block():
    meta, spec = jobset_pod()
    spec["containers"][0]["envFrom"] = [{"secretRef": {"name": "hf-token"}, "prefix": "HF_"}]
    result = inject_pod(pod(meta, spec), topology_of(meta, spec), names=PORT_ONLY)
    assert env_of(result.object["spec"]["containers"][0])["MASTER_PORT"] == str(
        RANGE.port_for(f"{JOBSET_UID_A}/trainer/0")
    )


def test_fully_specified_manifest_passes_through_unchanged():
    meta, spec = jobset_pod()
    with_containers(spec, containers=2, init=1)
    full = [
        {"name": "MASTER_ADDR", "value": "10.0.0.4"},
        {"name": "MASTER_PORT", "value": "29511"},
        {"name": "NODE_RANK", "value": "0"},
        {"name": "NNODES", "value": "2"},
        {"name": "PET_MASTER_ADDR", "value": "10.0.0.4"},
        {"name": "PET_MASTER_PORT", "value": "29511"},
        {"name": "PET_NODE_RANK", "value": "0"},
        {"name": "PET_NNODES", "value": "2"},
        {"name": "PET_NPROC_PER_NODE", "value": "8"},
        {"name": "OTHER", "value": "x"},
    ]
    for field in ("containers", "initContainers"):
        for container in spec[field]:
            container["env"] = copy.deepcopy(full)
    original = pod(meta, spec)
    result = inject_pod(original, topology_of(meta, spec))
    assert result.object == original
    assert result.patch == []
    assert result.injected == ()


# Each fact has one value per container


def test_a_defined_partner_is_followed_by_reference():
    meta, spec = jobset_pod()
    spec["containers"][0]["env"] = [{"name": "MASTER_ADDR", "value": "10.0.0.4"}]
    result = inject_pod(pod(meta, spec), topology_of(meta, spec))
    env = result.object["spec"]["containers"][0]["env"]
    # Kubernetes expands $(VAR) from variables defined earlier in the list.
    assert env[0] == {"name": "MASTER_ADDR", "value": "10.0.0.4"}
    assert {"name": "PET_MASTER_ADDR", "value": "$(MASTER_ADDR)"} in env
    linked = [r for r in result.records if r.name == "PET_MASTER_ADDR"]
    assert linked[0].action == "linked"


def test_a_partner_that_env_from_may_define_blocks_the_fact():
    meta, spec = jobset_pod()
    spec["containers"][0]["envFrom"] = [{"configMapRef": {"name": "rdzv"}, "prefix": "PET_"}]
    result = inject_pod(pod(meta, spec), topology_of(meta, spec))
    env = env_of(result.object["spec"]["containers"][0])
    assert "MASTER_ADDR" not in env and "PET_MASTER_ADDR" not in env
    skipped = {r.name: r for r in result.records if r.action == "skipped"}
    assert skipped["MASTER_ADDR"].detail == "envFrom may define PET_MASTER_ADDR"


def test_single_process_rank_follows_a_manifest_node_rank():
    meta, spec = jobset_pod(replicas=2, job_index=1)
    spec["containers"][0]["env"] = [{"name": "NODE_RANK", "value": "1"}]
    topology = pod_topology(meta, spec, pods_per_job=1, processes_per_node=1)
    env = env_of(inject_pod(pod(meta, spec), topology).object["spec"]["containers"][0])
    assert env["RANK"] == "$(NODE_RANK)"
    assert env["WORLD_SIZE"] == "2"
    assert env["LOCAL_RANK"] == "0"


def test_container_filter():
    meta, spec = jobset_pod()
    with_containers(spec, containers=2, init=1)
    result = inject_pod(pod(meta, spec), topology_of(meta, spec), containers=("c1",))
    assert "env" not in result.object["spec"]["containers"][0]
    assert "env" not in result.object["spec"]["initContainers"][0]
    assert "MASTER_ADDR" in env_of(result.object["spec"]["containers"][1])


def test_unknown_names_are_rejected():
    meta, spec = jobset_pod()
    with pytest.raises(ValueError, match="RANKK"):
        inject_pod(pod(meta, spec), topology_of(meta, spec), names=("RANKK",))


def test_input_object_is_not_mutated():
    meta, spec = jobset_pod()
    original = pod(meta, spec)
    snapshot = copy.deepcopy(original)
    inject_pod(original, topology_of(meta, spec))
    assert original == snapshot


# Workload templates


def jobset(replicas, completions, *, name="qwen-image", subdomain=None):
    obj = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {"name": name, "namespace": "training"},
        "spec": {
            "replicatedJobs": [
                {
                    "name": "trainer",
                    "replicas": replicas,
                    "template": {
                        "spec": {
                            "completionMode": "Indexed",
                            "completions": completions,
                            "parallelism": completions,
                            "template": {"spec": {"containers": [{"name": "trainer", "image": "x"}]}},
                        }
                    },
                }
            ]
        },
    }
    if subdomain:
        obj["spec"]["network"] = {"subdomain": subdomain}
    return obj


def template_env(result, path=("spec", "replicatedJobs", 0, "template", "spec", "template", "spec")):
    node = result.object
    for key in path:
        node = node[key]
    return {e["name"]: e for e in node["containers"][0]["env"]}


def test_expanded_jobset_template_rank_is_the_job_index_annotation():
    original = jobset(4, 1)
    result = inject_jobset(original)
    env = template_env(result)
    assert env["NODE_RANK"]["valueFrom"] == {
        "fieldRef": {"fieldPath": "metadata.annotations['jobset.sigs.k8s.io/job-index']"}
    }
    assert env["NNODES"]["value"] == "4"
    assert env["PET_MASTER_ADDR"]["value"] == "qwen-image-trainer-0-0.qwen-image"
    assert env["PET_MASTER_PORT"]["value"] == "29500"
    assert_patch_reproduces(result, original)


def test_indexed_jobset_template_rank_is_the_completion_index_annotation():
    # The Kubeflow Trainer torch plugin's PET_NODE_RANK field path.
    env = template_env(inject_jobset(jobset(1, 4, subdomain="net")))
    assert env["PET_NODE_RANK"]["valueFrom"] == {
        "fieldRef": {"fieldPath": "metadata.annotations['batch.kubernetes.io/job-completion-index']"}
    }
    assert env["MASTER_ADDR"]["value"] == "qwen-image-trainer-0-0.net"


def test_jobset_template_with_several_multi_pod_jobs_has_no_single_field_rank():
    with pytest.raises(TopologyError, match="replicas"):
        inject_jobset(jobset(2, 2))


def test_jobset_template_without_dns_hostnames_is_rejected():
    obj = jobset(2, 1)
    obj["spec"]["network"] = {"enableDNSHostnames": False}
    with pytest.raises(TopologyError, match="DNS"):
        inject_jobset(obj)


def test_generate_name_jobset_has_no_stable_address():
    obj = jobset(2, 1)
    obj["metadata"] = {"generateName": "qwen-", "namespace": "training"}
    with pytest.raises(TopologyError, match="name"):
        inject_jobset(obj)


def lws(size=4, leader=True):
    obj = {
        "apiVersion": "leaderworkerset.x-k8s.io/v1",
        "kind": "LeaderWorkerSet",
        "metadata": {"name": "vllm", "namespace": "inference"},
        "spec": {
            "replicas": 2,
            "leaderWorkerTemplate": {
                "size": size,
                "workerTemplate": {"spec": {"containers": [{"name": "vllm", "image": "x"}]}},
            },
        },
    }
    if leader:
        obj["spec"]["leaderWorkerTemplate"]["leaderTemplate"] = {
            "spec": {"containers": [{"name": "vllm", "image": "x"}]}
        }
    return obj


@pytest.mark.parametrize("template", ["leaderTemplate", "workerTemplate"])
def test_lws_templates_follow_lws_env(template):
    original = lws()
    result = inject_leaderworkerset(original)
    env = template_env(result, ("spec", "leaderWorkerTemplate", template, "spec"))
    # LWS prepends LWS_LEADER_ADDRESS to every container's env, so the reference expands.
    assert env["MASTER_ADDR"]["value"] == "$(LWS_LEADER_ADDRESS)"
    assert env["NNODES"]["value"] == "4"
    assert env["NODE_RANK"]["valueFrom"] == {
        "fieldRef": {"fieldPath": "metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']"}
    }
    assert_patch_reproduces(result, original)


def test_lws_without_leader_template_injects_the_worker_template():
    result = inject_leaderworkerset(lws(leader=False))
    assert "leaderTemplate" not in result.object["spec"]["leaderWorkerTemplate"]
    assert template_env(result, ("spec", "leaderWorkerTemplate", "workerTemplate", "spec"))["NNODES"]["value"] == "4"


def test_indexed_job_template():
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "sweep", "namespace": "default"},
        "spec": {
            "completionMode": "Indexed",
            "completions": 3,
            "template": {"spec": {"subdomain": "sweep-svc", "containers": [{"name": "main", "image": "x"}]}},
        },
    }
    env = template_env(inject_job(job), ("spec", "template", "spec"))
    assert env["MASTER_ADDR"]["value"] == "sweep-0.sweep-svc"
    assert env["NNODES"]["value"] == "3"
    assert env["NODE_RANK"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.annotations['batch.kubernetes.io/job-completion-index']"
    )


def test_inject_pod_accepts_an_explicit_topology():
    meta, spec = jobset_pod()
    topology = Topology(num_nodes=1, node_rank=0, master_addr="127.0.0.1", master_port=29500)
    env = env_of(inject_pod(pod(meta, spec), topology).object["spec"]["containers"][0])
    assert env["NNODES"] == "1"
