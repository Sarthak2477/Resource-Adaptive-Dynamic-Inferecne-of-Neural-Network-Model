def build_cost_table(model, device, configs, input_shape=(1,3,32,32), n_reps=30):
    """configs: list of (width, bit_width) or (width, bit_width, depth) tuples."""
    table = []
    for cfg in configs:
        latency = profile_config(model, *cfg, device, input_shape, n_reps)  # from earlier profiling harness
        acc = eval_lookup.get(cfg)  # from your existing evaluate() results
        table.append({'config': cfg, 'latency_ms': latency['latency_ms'],
                       'mem_mb': latency['peak_mem_mb'], 'acc': acc})
    return table

def pareto_frontier(table):
    frontier = []
    for r in table:
        dominated = any(o['latency_ms'] <= r['latency_ms'] and o['acc'] >= r['acc'] and o is not r
                         for o in table)
        if not dominated:
            frontier.append(r)
    return sorted(frontier, key=lambda r: r['latency_ms'])