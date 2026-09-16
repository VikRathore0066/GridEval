from src.run_tsfm_sweep import run_context_sweep_for_dataset


def run(config: dict) -> dict:
    dataset = config["dataset"]
    model_id = config.get("model_id", "amazon/chronos-t5-small")
    device = config.get("device", None)
    return run_context_sweep_for_dataset(dataset, model_id=model_id, device=device)
