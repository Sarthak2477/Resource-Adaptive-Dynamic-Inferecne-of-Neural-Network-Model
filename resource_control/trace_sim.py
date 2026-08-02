def run_controller_against_trace(controller, trace, latency_lookup, acc_lookup):
    results = []
    for true_budget in trace:
        config = controller.select(resource_state=None, latency_budget_ms=true_budget)
        true_latency = latency_lookup[config]
        results.append({
            'config': config, 'acc': acc_lookup[config],
            'latency': true_latency, 'violated': true_latency > true_budget,
        })
    return results