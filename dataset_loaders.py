from datasets import load_dataset


def load_jbb_behaviors(max_examples=None, split="harmful") -> list[dict]:
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", name="behaviors", split=split)
    behaviors = []
    for item in dataset:
        behavior = {
            "goal": item.get("Goal", ""),
            "target": item.get("Target", ""),
            "category": item.get("Category", ""),
        }
        if behavior["goal"]:
            behaviors.append(behavior)
            if max_examples and len(behaviors) >= max_examples:
                break
    return behaviors


def load_advbench_behaviors(max_examples=None) -> list[dict]:
    dataset = load_dataset("walledai/AdvBench", split="train")
    behaviors = []
    for item in dataset:
        behavior = {
            "goal": item.get("prompt", "") or item.get("goal", ""),
            "target": item.get("target", ""),
            "category": item.get("category", ""),
        }
        if behavior["goal"]:
            behaviors.append(behavior)
            if max_examples and len(behaviors) >= max_examples:
                break
    return behaviors


def load_harmbench_behaviors(max_examples=None) -> list[dict]:
    dataset = load_dataset("walledai/HarmBench", "standard", split="train")
    behaviors = []
    for item in dataset:
        behavior = {
            "goal": item.get("prompt", "") or item.get("text", ""),
            "target": item.get("target", ""),
            "category": item.get("category", "") or item.get("semantic_category", ""),
        }
        if behavior["goal"]:
            behaviors.append(behavior)
            if max_examples and len(behaviors) >= max_examples:
                break
    return behaviors


def load_behaviors(dataset: str, max_examples=None) -> list[dict]:
    """Dispatch to the correct loader by name."""
    dataset = dataset.lower()
    if dataset == "harmbench":
        print("Loading HarmBench dataset...")
        return load_harmbench_behaviors(max_examples=max_examples)
    elif dataset == "advbench":
        print("Loading AdvBench dataset...")
        return load_advbench_behaviors(max_examples=max_examples)
    else:
        print("Loading JailbreakBench dataset...")
        return load_jbb_behaviors(max_examples=max_examples)
