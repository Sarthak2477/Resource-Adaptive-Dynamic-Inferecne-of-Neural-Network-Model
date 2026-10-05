import random


def class_interleaved_indices(targets, num_classes, start_offset=0, stop_offset=None, seed=0):
    """Return a seeded, class-interleaved dataset slice with equal class counts."""
    if start_offset < 0 or (stop_offset is not None and stop_offset <= start_offset):
        raise ValueError("Invalid per-class index range")
    indices_by_class = {class_index: [] for class_index in range(num_classes)}
    for dataset_index, class_index in enumerate(targets):
        class_index = int(class_index)
        if class_index not in indices_by_class:
            raise ValueError(f"Unexpected class index {class_index}")
        indices_by_class[class_index].append(dataset_index)
    rng = random.Random(seed)
    for indices in indices_by_class.values():
        rng.shuffle(indices)

    end_offset = stop_offset
    if end_offset is None:
        end_offset = max((len(indices) for indices in indices_by_class.values()), default=0)
    if any(len(indices) < end_offset for indices in indices_by_class.values()):
        raise ValueError("At least one class does not contain the requested per-class range")
    return [
        indices_by_class[class_index][offset]
        for offset in range(start_offset, end_offset)
        for class_index in range(num_classes)
    ]