"""Names defined by the upstream projects this library reads and mirrors.

Sources (read at these revisions):

* JobSet v0.11.1 (kubernetes-sigs/jobset ecaf56ab):
  api/jobset/v1alpha2/jobset_types.go, pkg/constants/constants.go,
  pkg/controllers/jobset_controller.go (labelAndAnnotateObject, GetSubdomain,
  CoordinatorEndpoint), pkg/util/placement/placement.go (GenJobName),
  pkg/webhooks/jobset_webhook.go (generated-name length validation).
* LeaderWorkerSet v0.8.0 (kubernetes-sigs/lws fc19b9ae):
  api/leaderworkerset/v1/leaderworkerset_types.go,
  pkg/utils/pod/pod_utils.go (AddLWSVariables),
  pkg/webhooks/pod_webhook.go, pkg/utils/statefulset/statefulset_utils.go.
* Kubernetes Job controller (kubernetes/kubernetes pkg/controller/job):
  job_controller.go sets ``spec.hostname = <job-name>-<completion-index>``
  and the completion index annotation and label; indexed_job_utils.go.
* Kubeflow Trainer (kubeflow/trainer 6dc131ce): pkg/constants/constants.go
  and pkg/runtime/framework/plugins/torch/torch.go; the PET_* contract
  comes from kubeflow/trainer PR #1840 ("Set correct ENV for PytorchJob to
  support torchrun").
* PyTorch torch/distributed/argparse_util.py: every torchrun option ``--x-y``
  defaults to the environment variable ``PET_X_Y``.
"""

# JobSet labels and annotations; the controller writes each as both on the
# Job's pod template, so they are on the pod when the Job controller creates it.
JOBSET_NAME = "jobset.sigs.k8s.io/jobset-name"
JOBSET_UID = "jobset.sigs.k8s.io/jobset-uid"
JOBSET_REPLICATED_JOB_NAME = "jobset.sigs.k8s.io/replicatedjob-name"
JOBSET_REPLICATED_JOB_REPLICAS = "jobset.sigs.k8s.io/replicatedjob-replicas"
JOBSET_GLOBAL_REPLICAS = "jobset.sigs.k8s.io/global-replicas"
JOBSET_JOB_INDEX = "jobset.sigs.k8s.io/job-index"
JOBSET_JOB_GLOBAL_INDEX = "jobset.sigs.k8s.io/job-global-index"
JOBSET_RESTART_ATTEMPT = "jobset.sigs.k8s.io/restart-attempt"

# Kubernetes Job (batch/v1) pod labels and annotations.
JOB_COMPLETION_INDEX = "batch.kubernetes.io/job-completion-index"
JOB_NAME = "batch.kubernetes.io/job-name"
JOB_CONTROLLER_UID = "batch.kubernetes.io/controller-uid"
# Env var the Job controller adds to every container of an Indexed Job pod.
JOB_COMPLETION_INDEX_ENV = "JOB_COMPLETION_INDEX"

# LeaderWorkerSet labels and annotations.
LWS_NAME = "leaderworkerset.sigs.k8s.io/name"
LWS_GROUP_INDEX = "leaderworkerset.sigs.k8s.io/group-index"
LWS_WORKER_INDEX = "leaderworkerset.sigs.k8s.io/worker-index"
LWS_SIZE = "leaderworkerset.sigs.k8s.io/size"
LWS_SUBDOMAIN_POLICY = "leaderworkerset.sigs.k8s.io/subdomainPolicy"
LWS_SUBDOMAIN_UNIQUE_PER_REPLICA = "UniquePerReplica"
# Env vars LWS's pod webhook prepends to every container.
LWS_LEADER_ADDRESS_ENV = "LWS_LEADER_ADDRESS"
LWS_GROUP_SIZE_ENV = "LWS_GROUP_SIZE"
LWS_WORKER_INDEX_ENV = "LWS_WORKER_INDEX"

# torch.distributed env:// rendezvous and torchrun.
MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = "MASTER_PORT"
NODE_RANK = "NODE_RANK"
NNODES = "NNODES"
PET_MASTER_ADDR = "PET_MASTER_ADDR"
PET_MASTER_PORT = "PET_MASTER_PORT"
PET_NODE_RANK = "PET_NODE_RANK"
PET_NNODES = "PET_NNODES"
PET_NPROC_PER_NODE = "PET_NPROC_PER_NODE"
# Per-process; a launcher (torchrun, DeepSpeed) sets these for each process it starts.
RANK = "RANK"
WORLD_SIZE = "WORLD_SIZE"
LOCAL_RANK = "LOCAL_RANK"
LOCAL_WORLD_SIZE = "LOCAL_WORLD_SIZE"

# torch's default rendezvous port (torch/distributed/run.py --master-port) and
# the Kubeflow Trainer ContainerTrainerPort.
DEFAULT_MASTER_PORT = 29500
# torchrun --master-addr default.
LOCAL_MASTER_ADDR = "127.0.0.1"
# torchrun --nproc-per-node accepts an integer or one of these.
NPROC_PER_NODE_WORDS = frozenset({"auto", "cpu", "gpu", "xpu"})
