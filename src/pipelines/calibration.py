from src.run_calibration import run_calibration_suite


def run(config: dict) -> dict:
    dataset = config["dataset"]
    device = config.get("device", None)
    return run_calibration_suite(dataset, device=device)
