"""Fill a Kubernetes object's containers with the canonical env, never
overriding what the manifest defines.

Per container, per name:

* defined in ``env`` (``value`` or ``valueFrom``): kept.
* possibly defined through ``envFrom``: kept. An explicit ``env`` entry
  takes precedence over ``envFrom``, so injecting one would override a
  ConfigMap or Secret value that cannot be read at admission; a source
  counts when its ``prefix`` (empty by default) begins the name.
* a partner name of the same fact (MASTER_ADDR and PET_MASTER_ADDR, ...) is
  defined in ``env``: linked as ``$(<partner>)``. The kubelet expands
  ``$(VAR)`` from variables defined earlier in the list (and ``envFrom``),
  and injected entries are appended, so the container sees one value.
* a partner may come from ``envFrom``: skipped, since the value it would
  have to match is unknown.
* otherwise: injected.

The result holds the new object (the input is not modified), an RFC 6902
JSON patch that turns the input into it, and one record per decision.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Collection, Iterable, Mapping, Sequence, Union

from .env import ALL_NAMES, EnvGroup, env_groups, env_var
from .topology import (
    FieldRef,
    Topology,
    TopologyError,
    job_topology,
    jobset_replicated_job_topology,
    leaderworkerset_topology,
)
from . import keys


@dataclass(frozen=True)
class InjectionRecord:
    container: str
    name: str
    action: str  # "injected", "linked", "kept" or "skipped"
    detail: str


@dataclass(frozen=True)
class InjectionResult:
    object: dict
    patch: list
    records: tuple[InjectionRecord, ...]

    @property
    def injected(self) -> tuple[InjectionRecord, ...]:
        """Entries added (``injected`` and ``linked``)."""
        return tuple(r for r in self.records if r.action in ("injected", "linked"))

    @property
    def kept(self) -> tuple[InjectionRecord, ...]:
        return tuple(r for r in self.records if r.action == "kept")

    @property
    def skipped(self) -> tuple[InjectionRecord, ...]:
        return tuple(r for r in self.records if r.action == "skipped")


def _selected(names: Union[Collection[str], None]) -> frozenset[str]:
    if names is None:
        return frozenset(ALL_NAMES)
    unknown = sorted(set(names) - set(ALL_NAMES))
    if unknown:
        raise ValueError(f"not canonical env names: {', '.join(unknown)}; choose from {', '.join(ALL_NAMES)}")
    return frozenset(names)


def _fill_pod_spec(
    spec: dict,
    groups: Sequence[EnvGroup],
    *,
    pointer: str,
    label: str,
    selected: frozenset[str],
    containers: Union[Collection[str], None],
    records: list,
    patch: list,
) -> None:
    for field in ("containers", "initContainers"):
        for index, container in enumerate(spec.get(field) or []):
            if containers is not None and container.get("name") not in containers:
                continue
            where = f"{label}.{field}[{container.get('name')}]"
            explicit = {entry.get("name") for entry in container.get("env") or []}
            sources = container.get("envFrom") or []

            def from_env_from(name: str) -> bool:
                return any(name.startswith(source.get("prefix") or "") for source in sources)

            additions = []
            for group in groups:
                defined = [n for n in group.names if n in explicit]
                possible = [n for n in group.names if n not in explicit and from_env_from(n)]
                for name in group.names:
                    if name not in selected:
                        continue
                    if name in explicit:
                        records.append(InjectionRecord(where, name, "kept", "env"))
                    elif from_env_from(name):
                        records.append(InjectionRecord(where, name, "kept", "envFrom"))
                    elif defined:
                        reference = f"$({defined[0]})"
                        additions.append({"name": name, "value": reference})
                        records.append(InjectionRecord(where, name, "linked", reference))
                    elif possible:
                        records.append(InjectionRecord(where, name, "skipped", f"envFrom may define {possible[0]}"))
                    else:
                        additions.append(env_var(name, group.value))
                        detail = (
                            f"fieldRef {group.value.field_path}" if isinstance(group.value, FieldRef) else group.value
                        )
                        records.append(InjectionRecord(where, name, "injected", detail))
            if not additions:
                continue
            path = f"{pointer}/{field}/{index}/env"
            if container.get("env") is None:
                container["env"] = additions
                patch.append({"op": "add", "path": path, "value": copy.deepcopy(additions)})
            else:
                container["env"].extend(additions)
                patch.extend({"op": "add", "path": f"{path}/-", "value": copy.deepcopy(a)} for a in additions)


def _pointer(path: Iterable[Union[str, int]]) -> str:
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def _label(path: Iterable[Union[str, int]]) -> str:
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else part)
    return out


def _inject(
    obj: Mapping,
    locations: Sequence[tuple[tuple[Union[str, int], ...], Topology]],
    *,
    names: Union[Collection[str], None],
    containers: Union[Collection[str], None],
) -> InjectionResult:
    selected = _selected(names)
    out = copy.deepcopy(dict(obj))
    records: list = []
    patch: list = []
    for path, topology in locations:
        spec = out
        for part in path:
            spec = spec[part]
        _fill_pod_spec(
            spec,
            env_groups(topology),
            pointer=_pointer(path),
            label=_label(path),
            selected=selected,
            containers=containers,
            records=records,
            patch=patch,
        )
    return InjectionResult(object=out, patch=patch, records=tuple(records))


def inject_pod(
    pod: Mapping,
    topology: Topology,
    *,
    names: Union[Collection[str], None] = None,
    containers: Union[Collection[str], None] = None,
) -> InjectionResult:
    """Fill a Pod (``spec.containers`` and ``spec.initContainers``). ``names``
    limits the canonical names considered; ``containers`` limits the
    container names."""
    return _inject(pod, [(("spec",), topology)], names=names, containers=containers)


def inject_jobset(
    jobset: Mapping,
    *,
    port: int = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
    replicated_jobs: Union[Collection[str], None] = None,
    names: Union[Collection[str], None] = None,
    containers: Union[Collection[str], None] = None,
) -> InjectionResult:
    """Fill the pod template of each ReplicatedJob (or those named), each with
    its own world (see jobset_replicated_job_topology)."""
    locations = []
    for index, replicated_job in enumerate((jobset.get("spec") or {}).get("replicatedJobs") or []):
        name = replicated_job.get("name")
        if replicated_jobs is not None and name not in replicated_jobs:
            continue
        topology = jobset_replicated_job_topology(jobset, name, port=port, processes_per_node=processes_per_node)
        locations.append((("spec", "replicatedJobs", index, "template", "spec", "template", "spec"), topology))
    return _inject(jobset, locations, names=names, containers=containers)


def inject_leaderworkerset(
    lws: Mapping,
    *,
    port: int = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
    names: Union[Collection[str], None] = None,
    containers: Union[Collection[str], None] = None,
) -> InjectionResult:
    """Fill the leader template (when present) and the worker template."""
    topology = leaderworkerset_topology(lws, port=port, processes_per_node=processes_per_node)
    template = (lws.get("spec") or {}).get("leaderWorkerTemplate") or {}
    locations = [
        (("spec", "leaderWorkerTemplate", key, "spec"), topology)
        for key in ("leaderTemplate", "workerTemplate")
        if (template.get(key) or {}).get("spec") is not None
    ]
    if not locations:
        raise TopologyError("LeaderWorkerSet has no leaderTemplate or workerTemplate spec")
    return _inject(lws, locations, names=names, containers=containers)


def inject_job(
    job: Mapping,
    *,
    port: int = keys.DEFAULT_MASTER_PORT,
    processes_per_node: Union[int, str] = "auto",
    names: Union[Collection[str], None] = None,
    containers: Union[Collection[str], None] = None,
) -> InjectionResult:
    """Fill an Indexed Job's pod template."""
    topology = job_topology(job, port=port, processes_per_node=processes_per_node)
    return _inject(job, [(("spec", "template", "spec"), topology)], names=names, containers=containers)
