import numpy as np
import torch
from chronos import ChronosPipeline


def load_chronos_pipeline(model_id: str = "amazon/chronos-t5-small", device: str = None) -> ChronosPipeline:
    """Load pretrained Chronos pipeline from local cache or Hugging Face."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Chronos model '{model_id}' on {device.upper()}...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
        local_files_only=True,
    )


def forecast_chronos_window(
    pipeline: ChronosPipeline,
    context_series: np.ndarray,
    prediction_length: int = 24,
    num_samples: int = 20,
) -> dict:
    """Generate probabilistic forecast samples for a single context window."""
    context_tensor = torch.tensor(context_series, dtype=torch.float32)
    with torch.no_grad():
        forecast = pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

    samples = forecast[0].cpu().numpy()  # shape: [num_samples, prediction_length]
    return {
        "median": np.median(samples, axis=0),
        "samples": samples,
        "q05": np.percentile(samples, 5, axis=0),
        "q95": np.percentile(samples, 95, axis=0),
        "q10": np.percentile(samples, 10, axis=0),
        "q90": np.percentile(samples, 90, axis=0),
    }


def rolling_tsfm_evaluation(
    pipeline: ChronosPipeline,
    history_values: np.ndarray,
    test_values: np.ndarray,
    context_len: int = 512,
    horizon: int = 24,
    step: int = 24,
    num_samples: int = 20,
    batch_size: int = 16,
) -> dict:
    """Run rolling-window evaluation over test sequence in batches."""
    full_stream = np.concatenate([history_values, test_values])
    test_start_idx = len(history_values)
    total_len = len(full_stream)

    window_contexts = []
    window_slices = []
    curr_idx = test_start_idx

    while curr_idx < total_len:
        curr_horizon = min(horizon, total_len - curr_idx)
        if curr_horizon <= 0:
            break
        context_start = max(0, curr_idx - context_len)
        context_window = full_stream[context_start:curr_idx]
        eval_slice = min(step, curr_horizon)

        window_contexts.append(context_window)
        window_slices.append((curr_idx, eval_slice, curr_horizon))
        curr_idx += eval_slice

    preds_median = []
    actuals = []
    lower_90 = []
    upper_90 = []
    all_sample_batches = []

    total_batches = (len(window_contexts) + batch_size - 1) // batch_size
    for b_idx, b_start in enumerate(range(0, len(window_contexts), batch_size)):
        b_end = min(b_start + batch_size, len(window_contexts))
        batch_ctxs = window_contexts[b_start:b_end]
        batch_sl = window_slices[b_start:b_end]

        max_ctx = max(len(c) for c in batch_ctxs)
        padded_batch = np.zeros((len(batch_ctxs), max_ctx), dtype=np.float32)
        for i, c in enumerate(batch_ctxs):
            padded_batch[i, -len(c):] = c

        context_tensor = torch.tensor(padded_batch, dtype=torch.float32)
        max_horizon = max(s[2] for s in batch_sl)

        with torch.no_grad():
            forecast = pipeline.predict(
                context_tensor,
                prediction_length=max_horizon,
                num_samples=num_samples,
            )

        batch_samples = forecast.cpu().numpy()
        print(f"      Batch {b_idx + 1}/{total_batches} processed...", end="\r", flush=True)

        for i, (c_idx, e_slice, _) in enumerate(batch_sl):
            s_i = batch_samples[i, :, :e_slice]
            med_i = np.median(s_i, axis=0)
            q05_i = np.percentile(s_i, 5, axis=0)
            q95_i = np.percentile(s_i, 95, axis=0)

            preds_median.extend(med_i)
            actuals.extend(full_stream[c_idx:c_idx + e_slice])
            lower_90.extend(q05_i)
            upper_90.extend(q95_i)
            all_sample_batches.append(s_i)

    return {
        "y_pred": np.array(preds_median),
        "y_true": np.array(actuals),
        "q05": np.array(lower_90),
        "q95": np.array(upper_90),
        "samples": np.concatenate(all_sample_batches, axis=1),
    }
