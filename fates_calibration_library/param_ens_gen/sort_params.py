from __future__ import annotations

from collections import defaultdict, deque
 
from .parameter import Parameter, ScaleFromRootParameter

def sort_params(params: list[Parameter]) -> list[Parameter]:
    """Return params reordered so every root_param is written before its dependents.
    
    Only ScaleFromRootParameters introduce ordering constraints. All other
    parameter types are treated as having no dependencies.
    
     If a root_param is not present in the parameter list (i.e. it is not
    being calibrated), no ordering constraint is added for it — the dataset
    value is assumed to be whatever was last written.

    Args:
        params (list[Parameter]): Unsorted parameter list, as constructed from the input 
        spreadsheet

    Raises:
        ValueError:  If a cycle is detected among ScaleFromRootParameter dependencies
        (e.g. A scales from B and B scales from A).

    Returns:
        list[Parameter]: Topologically sorted parameter list.
    """
    name_to_param = {p.spec.name: p for p in params}
 
    # Build adjacency list and in-degree count.
    # Edge: root_param to dependent (root must come first)
    dependents: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {p.spec.name: 0 for p in params}
 
    for param in params:
        if isinstance(param, ScaleFromRootParameter):
            root = param.spec.root_param
            if root in name_to_param:
                dependents[root].append(param.spec.name)
                in_degree[param.spec.name] += 1
 
    # Kahn's algorithm — seed with all zero-in-degree nodes in original order
    queue: deque[str] = deque(
        p.spec.name for p in params if in_degree[p.spec.name] == 0
    )
    sorted_names: list[str] = []
 
    while queue:
        name = queue.popleft()
        sorted_names.append(name)
        for dependent in dependents[name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
 
    if len(sorted_names) != len(params):
        cycle_members = [
            p.spec.name for p in params if p.spec.name not in sorted_names
        ]
        raise ValueError(
            f"Cycle detected among ScaleFromRootParameter dependencies: "
            f"{cycle_members}. Each root_param must not depend on its own "
            "dependents."
        )
 
    return [name_to_param[name] for name in sorted_names]
