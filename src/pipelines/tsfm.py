from src.run_tsfm_sweep import run_context_sweep_for_dataset


def run(config: dict) -> dict:
    return run_context_sweep_for_dataset(config)
