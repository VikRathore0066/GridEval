from src.baselines import run_baselines_for_dataset


def run(config: dict) -> dict:
    dataset = config["dataset"]
    include_weather = bool(config.get("features", {}).get("include_weather", False))
    return run_baselines_for_dataset(dataset, include_weather=include_weather)
