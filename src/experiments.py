from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentResult:
    name: str
    config: dict
    metrics: dict
    runtime_seconds: float
    timestamp: str
    status: str

    def save(self) -> Path:
        out_path = RESULTS_DIR / f"{self.name}_run.json"
        with open(out_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return out_path


def execute_experiment(name: str, config: dict, pipeline_fn) -> ExperimentResult:
    """Execute a pipeline function with timing and metric logging."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting experiment: {name}")
    start_time = time.time()

    try:
        metrics = pipeline_fn(config)
        status = "SUCCESS"
    except Exception as e:
        traceback.print_exc()
        metrics = {"error": str(e)}
        status = "FAILED"
        print(f"Experiment failed: {e}")

    duration = time.time() - start_time
    result = ExperimentResult(
        name=name,
        config=config,
        metrics=metrics,
        runtime_seconds=round(duration, 2),
        timestamp=datetime.now().isoformat(),
        status=status,
    )
    saved_path = result.save()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {name} in {duration:.1f}s (status: {status}) -> {saved_path}")
    return result
