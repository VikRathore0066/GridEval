from src.baselines import run_baselines_for_dataset


def run(config: dict) -> dict:
    dataset = config["dataset"]
    return run_baselines_for_dataset(dataset)
