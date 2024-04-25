from collections import deque, defaultdict

def topological_sort(indicator_configs):
    # Maps each indicator by its name for quick access
    config_map = {config['name']: config for config in indicator_configs}

    # Dependency map and reverse dependency map
    dependency_map = defaultdict(set)
    reverse_dependency_map = defaultdict(set)

    # Build the dependency and reverse dependency maps
    for config in indicator_configs:
        for dependency in config.get('depends_on', []):
            dependency_map[dependency].add(config['name'])
            reverse_dependency_map[config['name']].add(dependency)

    # Queue for processing - start with nodes having no dependencies
    queue = deque([name for name in config_map if not reverse_dependency_map[name]])
    sorted_configs = []

    while queue:
        current_name = queue.popleft()
        current_config = config_map[current_name]
        sorted_configs.append(current_config)

        # Process each node that depends on the current node
        for dependent in dependency_map[current_name]:
            reverse_dependency_map[dependent].remove(current_name)
            if not reverse_dependency_map[dependent]:  # No more dependencies
                queue.append(dependent)

    # Verify that all indicators have been sorted (no cyclic dependencies)
    if len(sorted_configs) != len(indicator_configs):
        raise ValueError("A cycle was detected in the dependencies, or there are missing dependencies.")

    return sorted_configs
