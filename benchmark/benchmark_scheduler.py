"""GPU-aware batch experiment scheduler.

Runs multiple experiments in synchronized batches within a total GPU
budget.  Each batch goes through four phases with barriers between them:

1. **Register + scale + setup** — deploy apps, prepare inputs, test request
2. **Dispatch** — all experiments fire requests concurrently (barrier)
3. **Collect** — OTel traces and metrics gathered after dispatch barrier
4. **Teardown + unregister** — clean up, release GPUs for next batch

This eliminates sync-operation interference in latency measurements
by ensuring all experiments in a batch start and stop dispatching at the
same wall-clock time.

Usage::

    from benchmark_scheduler import schedule_all
    from cornserve_utils import clear_all

    await clear_all()
    # Single pass:
    await schedule_all(experiments, total_gpus=8)
    # Multiple sequential passes:
    await schedule_all([pass1_experiments, pass2_experiments], total_gpus=8)
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
import traceback as tb
from typing import Any

from benchmark_cornserve import BenchmarkSession
from app_utils import app_to_expected_tasks
from benchmark_datasets import ControlledDiT, ControlledImageChat, DecodeLLMDataset, DecodeServeGenDataset, ServeGenDataset
from cornserve_utils import (
    check_apps,
    get_gpu_count,
    register_app,
    scale,
    unregister_app,
)
from schema import (
    DecodeLLMApp,
    DummyAudioGeriApp,
    DummyEricApp,
    DummyImageGeriApp,
    DummyLLMApp,
    EPDMLLMApp,
    Experiment,
    GroupedMixedMLLMApp,
    MixedMLLMApp,
    MLLMRouterApp,
    ModServeApp,
    MonolithicLLMApp,
    OmniMLLMApp,
    OmniApp,
    OmniFlexApp,
    OmniRouterApp,
    PrefillLLMApp,
    QwenImageDisaggApp,
    QwenImageTextGeriApp,
    TimeSharingApp,
)

# Serialize all snapshot mutations (scale / release) to avoid races.
_snapshot_lock = asyncio.Lock()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warn_config_mismatch(experiment: Experiment) -> None:
    """Log a warning if the stored result.json config differs from the expected config."""
    path = experiment.to_path()
    try:
        with path.open("r") as f:
            stored = json.load(f)
    except Exception:
        return  # cannot read — nothing to compare
    stored_app = stored.get("config", {}).get("app")
    if stored_app is None:
        return
    expected_app = experiment.model_dump().get("app")
    if expected_app is None:
        return
    # Normalize for comparison: JSON roundtrip converts int dict keys to
    # strings (e.g. type_routing_weights {1: ...} → {"1": ...}).
    # Re-serialize both sides through JSON to get consistent string keys.
    stored_normalized = json.loads(json.dumps(stored_app))
    expected_normalized = json.loads(json.dumps(expected_app))
    if stored_normalized != expected_normalized:
        logger.warning(
            "Config mismatch for %s: stored app config differs from expected. "
            "Rerun with --no-skip-existing to regenerate.\n"
            "  stored:   %s\n"
            "  expected: %s",
            path,
            stored_app,
            expected_app,
        )


def _task_names(experiment: Experiment) -> frozenset[str]:
    """Return the set of unit-task names for *experiment*.

    Uses ``make_name()`` so that tasks which map to the same Cornserve
    task-manager name are treated as conflicts.  Two experiments conflict
    if they share ANY task name, because Cornserve merges replicas of the
    same task into one pool and round-robins across them.
    """
    tasks = app_to_expected_tasks(experiment.app)
    return frozenset(getattr(task, "make_name")() for task in tasks)


def _experiment_label(app: Any) -> str:
    """Build a short readable config label for *app*."""
    model = app.model_id.split("/")[-1]
    if isinstance(app, DummyEricApp):
        return f"Eric {app.modality} {model} bs={app.max_batch_size} tp={app.tp_size}"
    elif isinstance(app, DummyLLMApp):
        return f"LLM {model} tp={app.tp_size} bs={app.max_num_seqs}"
    elif isinstance(app, MonolithicLLMApp):
        return f"MonoLLM {model} tp={app.tp_size} bs={app.max_num_seqs}"
    elif isinstance(app, OmniMLLMApp):
        return f"OmniMLLM {model} tp={app.tp_size} bs={app.max_num_seqs} {app._enc_flags_str()}"
    elif isinstance(app, OmniApp):
        enc = "fission" if app.encoder_fission else "mono"
        ao = (
            f"tk{app.talker_num_replicas}bs{app.talker_max_num_seqs}_"
            f"ag{app.audio_geri_num_replicas}bs{app.audio_geri_max_batch_size}"
            if app.vocoder_fission
            else f"tv{app.talker_vocoder_num_replicas}bs{app.talker_max_num_seqs}"
        )
        return (
            f"Omni {model} {enc} "
            f"llm={app.llm_num_replicas}x[tp{app.llm_tp_size}bs{app.llm_max_num_seqs}] {ao}"
        )
    elif isinstance(app, OmniFlexApp):
        group_strs = []
        for gi, group in enumerate(app.groups):
            mods = "+".join(sorted(group.offloaded_modalities)) or "mono"
            t_strs = [f"tp{t.llm_tp_size}bs{t.llm_max_num_seqs}" for t in group.thinkers]
            group_strs.append(f"g{gi}({mods})[{','.join(t_strs)}]")
        return f"OmniFlex {model} {' '.join(group_strs)}"
    elif isinstance(app, OmniRouterApp):
        route_strs = []
        for i, route in enumerate(app.routes):
            w = app.routing_weights[i]
            rt = route.route_type.replace("omni_", "")
            route_strs.append(
                f"w={w:.2f} {rt} tp={route.llm_tp_size} bs={route.llm_max_num_seqs}"
            )
        return f"OmniRouter {model} [{'; '.join(route_strs)}]"
    elif isinstance(app, PrefillLLMApp):
        return f"PrefillLLM {model} tp={app.tp_size} bs={app.max_num_seqs}"
    elif isinstance(app, DecodeLLMApp):
        re_str = " re1" if app.receive_embeddings else ""
        return f"DecodeLLM {model} tp={app.tp_size} bs={app.max_num_seqs}{re_str}"
    elif isinstance(app, DummyImageGeriApp):
        return f"Geri image {model} sp={app.sp_size} bs={app.max_batch_size}"
    elif isinstance(app, QwenImageDisaggApp):
        return (
            f"QwenImageDisagg {model} enc={app.num_encoders}x"
            f"tp{app.encoder_tp_size}/bs{app.encoder_max_num_seqs} "
            f"geri={app.num_generators}xsp{app.generator_sp_size}"
        )
    elif isinstance(app, QwenImageTextGeriApp):
        return f"Geri image-text {model} sp={app.sp_size} bs={app.max_batch_size}"
    elif isinstance(app, DummyAudioGeriApp):
        return f"Geri audio {model} bs={app.max_batch_size}"
    elif isinstance(app, ModServeApp):
        return (
            f"ModServe {model} erics={app.num_erics} llms={app.num_llms} "
            f"tp={app.llm_tp_size} bs={app.llm_max_num_seqs}"
        )
    elif isinstance(app, TimeSharingApp):
        return (
            f"TimeSharing {model} erics={app.num_erics} llms={app.num_llms} "
            f"fission={app.encoder_fission_prob} tp={app.llm_tp_size} bs={app.llm_max_num_seqs}"
        )
    elif isinstance(app, MixedMLLMApp):
        route_summaries = [
            (
                f"w={weight:.2f} "
                f"llms={route.llm_num_replicas} "
                f"tp={route.llm_tp_size} "
                f"bs={route.llm_max_num_seqs}"
            )
            for route, weight in zip(app.routes, app.routing_weights, strict=True)
        ]
        return (
            f"MixedMLLM {model} erics={app.num_erics} "
            f"routes={len(app.routes)} [{' ; '.join(route_summaries)}]"
        )
    elif isinstance(app, GroupedMixedMLLMApp):
        group_summaries = []
        for g_idx, (group, group_weight) in enumerate(
            zip(app.groups, app.group_routing_weights, strict=True)
        ):
            route_summaries = [
                (
                    f"w={weight:.2f} "
                    f"llms={route.llm_num_replicas} "
                    f"tp={route.llm_tp_size} "
                    f"bs={route.llm_max_num_seqs}"
                )
                for route, weight in zip(
                    group.routes, group.routing_weights, strict=True
                )
            ]
            group_summaries.append(
                (
                    f"g{g_idx}:gw={group_weight:.2f} "
                    f"erics={group.num_erics} "
                    f"routes={len(group.routes)} [{' ; '.join(route_summaries)}]"
                )
            )
        return f"GroupedMixedMLLM {model} groups={len(app.groups)} [{' || '.join(group_summaries)}]"
    elif isinstance(app, MLLMRouterApp):
        route_summaries = []
        for route, weight in zip(app.routes, app.routing_weights, strict=True):
            if route.is_time_sharing():
                route_summaries.append(
                    (
                        f"w={weight:.2f} "
                        f"ts={route.encoder_fission_prob or 0.0:.2f} "
                        f"erics={route.eric_num_replicas} "
                        f"llms={route.llm_num_replicas} "
                        f"tp={route.llm_tp_size} "
                        f"bs={route.llm_max_num_seqs}"
                    )
                )
            else:
                route_summaries.append(
                    (
                        f"w={weight:.2f} "
                        f"re={int(route.encoder_fission)} "
                        f"erics={route.eric_num_replicas} "
                        f"llms={route.llm_num_replicas} "
                        f"tp={route.llm_tp_size} "
                        f"bs={route.llm_max_num_seqs}"
                    )
                )
        return f"MLLMRouter {model} routes={len(app.routes)} [{' ; '.join(route_summaries)}]"
    elif isinstance(app, EPDMLLMApp):
        return (
            f"EPD {model} e={app.num_eric_replicas}xbs{app.eric_max_batch_size} "
            f"p={app.num_prefill_replicas}xtp{app.prefill_tp_size}bs{app.prefill_max_num_seqs} "
            f"d={app.num_decode_replicas}xtp{app.decode_tp_size}bs{app.decode_max_num_seqs}"
        )
    else:
        return type(app).__name__


def _dry_run_line(exp: Experiment, idx: int, gpu_count: int) -> str:
    """Build a compact one-line description for dry-run output."""
    ds = exp.dataset
    if isinstance(ds, ControlledImageChat):
        res = f"{ds.image_width}x{ds.image_height}"
    elif isinstance(ds, ControlledDiT) and len(ds.resolution_pool) == 1:
        h, w = ds.resolution_pool[0]
        res = f"{h}x{w}"
    else:
        res = "--"
    return (
        f"  #{idx}  {_experiment_label(exp.app)}  |  "
        f"{_dataset_label(ds)}  {res}  {exp.request_rate:.1f} req/s  gpu={gpu_count}"
    )


def _dataset_label(dataset: Any) -> str:
    """Build a short readable label for the dataset."""
    if isinstance(dataset, ControlledImageChat):
        return f"in={dataset.input_len} out={dataset.output_len} img={dataset.image_probability}"
    elif isinstance(dataset, ServeGenDataset):
        return f"img={dataset.image_prob} dur={dataset.duration}"
    elif isinstance(dataset, DecodeLLMDataset):
        return f"prefix={dataset.prefix_len} out={dataset.output_len}"
    elif isinstance(dataset, DecodeServeGenDataset):
        return f"img={dataset.image_prob} prefix={dataset.prefix_len}"
    elif isinstance(dataset, ControlledDiT):
        return f"n={dataset.num_requests} steps={dataset.num_inference_steps}"
    else:
        return type(dataset).__name__


def _print_experiment_summary(experiment: Experiment, data: dict[str, Any]) -> None:
    """Print per-app-type result summary after a single experiment."""
    app = experiment.app
    app_label = type(app).__name__

    print(f"\n{'=' * 60}")
    print(f"Results: {app_label} ({experiment.to_path()})")
    print(f"{'=' * 60}")
    print(f"Throughput:       {data['throughput']:.2f} req/s")
    print(f"Duration:         {data['benchmark_duration']:.2f}s")
    total = data["num_successful"] + data["num_failed"]
    print(f"Successful:       {data['num_successful']}/{total}")
    if data["num_failed"] > 0:
        print(f"Failed:           {data['num_failed']}")
    if data.get("pod_failure"):
        print(f"*** POD FAILURE:  {data.get('dead_pods', [])} — results unreliable ***")

    if isinstance(
        app,
        (
            DummyLLMApp,
            MonolithicLLMApp,
            ModServeApp,
            TimeSharingApp,
            MixedMLLMApp,
            GroupedMixedMLLMApp,
            MLLMRouterApp,
        ),
    ):
        output_tput = data.get("output_throughput")
        if output_tput is not None:
            print(f"Output tput:      {output_tput:.2f} tok/s")
        ss = data.get("steady_state")
        if ss:
            print(
                f"Steady state:     {ss['duration_s']:.1f}s, "
                f"{ss['tokens_generated']} tokens, "
                f"{ss['requests_completed']} reqs, "
                f"{ss['token_throughput']:.1f} tok/s, "
                f"{ss['request_throughput']:.2f} req/s"
            )
        for label, key in [
            ("TTFT (ms)", "ttft_ms"),
            ("TTFT-noQ (ms)", "ttft_no_queue_ms"),
            ("ITL  (ms)", "itl_ms"),
            ("E2EL (ms)", "e2el_ms"),
            ("E2EL-noQ (ms)", "e2el_no_queue_ms"),
        ]:
            stats = data.get(key, {})
            if stats and stats.get("mean", 0) > 0:
                print(
                    f"{label:18s}mean={stats['mean']:.1f}  "
                    f"median={stats['median']:.1f}  p99={stats['p99']:.1f}"
                )
    else:
        for label, key in [
            ("E2EL (ms)", "e2el_ms"),
            ("E2EL-noQ (ms)", "e2el_no_queue_ms"),
            ("Engine queuing (ms)", "engine_queuing_ms"),
        ]:
            stats = data.get(key, {})
            if stats and stats.get("mean", 0) > 0:
                print(
                    f"{label:22s}mean={stats['mean']:.1f}  "
                    f"median={stats['median']:.1f}  p99={stats['p99']:.1f}"
                )
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------


async def run_single(experiment: Experiment) -> dict[str, Any]:
    """Run one experiment end-to-end (register, scale, benchmark).

    Uses ``experiment.request_rate`` as the request rate for the benchmark.
    Always unregisters the app on exit — even if the benchmark or an earlier
    step fails — so that orphaned task managers don't block future runs.

    This is a convenience entry point for running a single experiment
    outside of batch scheduling.  The batch scheduler
    (``schedule_all``) uses ``BenchmarkSession`` directly.
    """
    app = experiment.app
    app_label = type(app).__name__
    app_id: str | None = None

    try:
        # 1. Register + Scale (under lock to avoid concurrent deployment races)
        async with _snapshot_lock:
            print(f"[{app_label}] Registering app...")
            app_id = await register_app(app, verify=True)
            print(f"[{app_label}] Registered: {app_id}")

            print(f"[{app_label}] Scaling...")
            await scale(experiment)

        # 2. Wait for readiness
        await _wait_for_ready(app_id, app_label)

        # 3. Benchmark via BenchmarkSession
        session = BenchmarkSession(
            experiment=experiment,
            app_id=app_id,
            request_rate=experiment.request_rate,
        )
        await session.sample_inputs()
        await session.setup()
        try:
            await session.dispatch()
            data = await session.collect()
        finally:
            await session.teardown()

        # 4. Summary
        _print_experiment_summary(experiment, data)
        return data
    finally:
        # Always unregister (under lock so concurrent scale() never sees stale owners).
        if app_id is not None:
            async with _snapshot_lock:
                print(f"[{app_label}] Unregistering app {app_id}...")
                try:
                    await unregister_app(app_id)
                except Exception as e:
                    print(f"[{app_label}] Warning: failed to unregister {app_id}: {e}")


async def _wait_for_ready(app_id: str, label: str = "") -> None:
    """Poll ``check_apps`` until *app_id* is ready (up to 600 s)."""
    tag = f"[{label}] " if label else ""
    print(f"{tag}Waiting for app to become ready...")
    for attempt in range(600):
        try:
            await check_apps([app_id])
            print(f"{tag}App is ready")
            return
        except ValueError:
            if attempt % 10 == 0:
                print(f"{tag}  Still waiting... ({attempt}s)")
            await asyncio.sleep(1)
    raise RuntimeError(f"App {app_id} did not become ready within 600s")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


async def schedule_all(
    experiments: list[Experiment] | list[list[Experiment]],
    total_gpus: int,
    skip_existing: bool = True,
    dry_run: bool = False,
    dry_run_with_batch: bool = False,
) -> dict[int, dict[str, Any]] | list[dict[int, dict[str, Any]]]:
    """Schedule experiments, optionally in sequential passes.

    *experiments* may be a flat list (single pass) or a list of lists
    (multiple sequential passes).  Each pass is batch-scheduled
    independently via :func:`schedule_pass`.

    Returns:
        A single result dict when given ``list[Experiment]``, or a list
        of result dicts (one per pass) when given ``list[list[Experiment]]``.
    """
    # Flat list — single pass, backward-compatible.
    if experiments and not isinstance(experiments[0], list):
        return await schedule_pass(
            experiments,  # type: ignore[arg-type]
            total_gpus=total_gpus,
            skip_existing=skip_existing,
            dry_run=dry_run,
            dry_run_with_batch=dry_run_with_batch,
        )

    # List of lists — run each pass sequentially.
    passes: list[list[Experiment]] = experiments  # type: ignore[assignment]
    all_results: list[dict[int, dict[str, Any]]] = []
    for pass_idx, pass_experiments in enumerate(passes):
        if not pass_experiments:
            all_results.append({})
            continue
        print(f"\n{'*' * 70}")
        print(f"PASS {pass_idx + 1}/{len(passes)}: {len(pass_experiments)} experiment(s)")
        print(f"{'*' * 70}")
        result = await schedule_pass(
            pass_experiments,
            total_gpus=total_gpus,
            skip_existing=skip_existing,
            dry_run=dry_run,
            dry_run_with_batch=dry_run_with_batch,
        )
        all_results.append(result)
    return all_results


async def schedule_pass(
    experiments: list[Experiment],
    total_gpus: int,
    skip_existing: bool = True,
    dry_run: bool = False,
    dry_run_with_batch: bool = False,
) -> dict[int, dict[str, Any]]:
    """Schedule and run *experiments* in synchronized batches within a GPU budget.

    Experiments are grouped into batches that fit the GPU budget with no
    task-key conflicts.  Within each batch, all experiments go through
    four phases with barriers between them so that request dispatching
    starts and ends at the same wall-clock time.

    Returns a dict mapping experiment index to its result data (or error info).
    """
    # Pre-compute metadata per experiment.
    gpu_counts: list[int] = []
    task_name_sets: list[frozenset[str]] = []
    for exp in experiments:
        gpu_counts.append(get_gpu_count(exp))
        task_name_sets.append(_task_names(exp))

    # Sort indices by gpu_count descending (launch big experiments first).
    order = sorted(range(len(experiments)), key=lambda i: gpu_counts[i], reverse=True)

    # Filter out already-completed experiments if requested.
    # Failed experiments (<80% success) are NOT skipped.
    pending: list[int] = []
    skipped: list[int] = []
    for i in order:
        if skip_existing and experiments[i].exists():
            # Check if the result is actually a success
            result_path = experiments[i].to_path()
            is_good = True
            try:
                import json as _json
                with open(result_path) as _f:
                    _data = _json.load(_f)
                n_total = len(_data.get("results", []))
                n_success = _data.get("num_successful", 0)
                if n_total == 0 or (n_total > 0 and n_success / n_total < 0.8):
                    is_good = False
                    if not dry_run:
                        failed_path = result_path.with_suffix(".failed.json")
                        if failed_path.exists():
                            failed_path.unlink()
                        result_path.rename(failed_path)
                    print(f"  Previous run failed ({n_success}/{n_total} success), will rerun: {result_path.parent.name}")
            except Exception:
                pass
            if is_good:
                _warn_config_mismatch(experiments[i])
                skipped.append(i)
            else:
                pending.append(i)
        else:
            pending.append(i)

    if skipped:
        print(f"Skipping {len(skipped)} experiments with existing results")

    # Dry-run: print the schedule plan and summary table for existing results.
    if dry_run:
        print(f"\n{'=' * 70}")
        print("DRY RUN — Schedule Plan")
        print(f"{'=' * 70}")
        print(f"Total GPUs: {total_gpus}")
        print(f"Experiments to run: {len(pending)}")
        print(f"Experiments skipped (existing): {len(skipped)}")
        print()
        for rank, i in enumerate(pending, 1):
            print(f"  {rank:3d}. " + _dry_run_line(experiments[i], i, gpu_counts[i]).lstrip())
        print(f"{'=' * 70}")

        if dry_run_with_batch:
            print()
            print("DRY RUN — Batch Assignment")
            print(f"{'=' * 70}")
            sim_pending = list(pending)
            sim_batch_num = 0
            while sim_pending:
                sim_batch_num += 1
                sim_batch: list[int] = []
                sim_batch_gpus = 0
                sim_batch_task_names: set[str] = set()
                sim_still_pending: list[int] = []
                for i in sim_pending:
                    gc = gpu_counts[i]
                    names = task_name_sets[i]
                    if sim_batch_gpus + gc <= total_gpus and names.isdisjoint(sim_batch_task_names):
                        sim_batch.append(i)
                        sim_batch_gpus += gc
                        sim_batch_task_names.update(names)
                    else:
                        sim_still_pending.append(i)
                sim_pending = sim_still_pending
                if not sim_batch:
                    print(
                        f"\nWARNING: {len(sim_pending)} experiments cannot be scheduled "
                        f"(insufficient GPUs or unresolvable conflicts):"
                    )
                    for i in sim_pending:
                        print(
                            f"  - #{i} [{type(experiments[i].app).__name__}] "
                            f"gpus={gpu_counts[i]} tasks={','.join(sorted(task_name_sets[i]))[:60]}"
                        )
                    break
                print(f"\n{'#' * 70}")
                print(
                    f"BATCH {sim_batch_num}: {len(sim_batch)} experiment(s), "
                    f"{sim_batch_gpus}/{total_gpus} GPUs"
                )
                for i in sim_batch:
                    print(_dry_run_line(experiments[i], i, gpu_counts[i]))
                print(f"{'#' * 70}")
            print(f"{'=' * 70}")

        # Load results for existing experiments and print summary table.
        results: dict[int, dict[str, Any]] = {}
        for i in skipped:
            try:
                results[i] = experiments[i].load()
            except Exception as e:
                print(f"Warning: failed to load results for experiment #{i}: {e}")
        print_summary(experiments, results, gpu_counts, skipped=[], wall_duration=0.0)
        return results

    # --- Batch scheduling loop ---
    results: dict[int, dict[str, Any]] = {}
    wall_start = time.perf_counter()
    batch_num = 0

    while pending:
        batch_num += 1
        # --- Form batch: greedily pick experiments that fit GPU budget ---
        batch: list[int] = []
        batch_gpus = 0
        batch_task_names: set[str] = set()
        still_pending: list[int] = []

        for i in pending:
            gc = gpu_counts[i]
            names = task_name_sets[i]
            if batch_gpus + gc <= total_gpus and names.isdisjoint(batch_task_names):
                batch.append(i)
                batch_gpus += gc
                batch_task_names.update(names)
            else:
                still_pending.append(i)
        pending = still_pending

        if not batch:
            # Remaining experiments can't fit — too large or permanent conflicts.
            print(
                f"\nWARNING: {len(pending)} experiments cannot be scheduled "
                f"(insufficient GPUs or unresolvable conflicts):"
            )
            for i in pending:
                print(
                    f"  - #{i} [{type(experiments[i].app).__name__}] "
                    f"gpus={gpu_counts[i]} tasks={','.join(sorted(task_name_sets[i]))[:60]}"
                )
            break

        print(f"\n{'#' * 70}")
        print(
            f"BATCH {batch_num}: {len(batch)} experiment(s), "
            f"{batch_gpus}/{total_gpus} GPUs"
        )
        for i in batch:
            exp = experiments[i]
            print(
                f"  #{i} [{type(exp.app).__name__}] "
                f"gpus={gpu_counts[i]} path={exp.to_path()}"
            )
        print(f"{'#' * 70}")

        batch_results = await _run_batch(batch, experiments)
        results.update(batch_results)

    wall_duration = time.perf_counter() - wall_start

    # Load results for skipped (existing) experiments so the summary table
    # shows their data instead of "--".
    for i in skipped:
        try:
            results[i] = experiments[i].load()
            _print_experiment_summary(experiments[i], results[i])
        except Exception as e:
            print(f"Warning: failed to load results for experiment #{i}: {e}")

    print_summary(
        experiments, results, gpu_counts, skipped=[], wall_duration=wall_duration
    )
    return results


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def _run_batch(
    batch: list[int],
    experiments: list[Experiment],
) -> dict[int, dict[str, Any]]:
    """Run a single batch of experiments through all four phases.

    Uses barriers (``asyncio.gather``) between phases so that all
    experiments start and stop dispatching at the same wall-clock time.
    """
    results: dict[int, dict[str, Any]] = {}

    # Maps experiment index -> app_id (for cleanup)
    app_ids: dict[int, str] = {}
    # Maps experiment index -> BenchmarkSession
    sessions: dict[int, BenchmarkSession] = {}

    try:
        # ---- Phase 0: Register + Wait for ready (concurrent) ----
        print("\n--- Phase 0: Register, wait for ready ---")

        async def _register(i: int) -> None:
            exp = experiments[i]
            app_label = type(exp.app).__name__
            print(f"  [#{i} {app_label}] Registering app...")
            app_id = await register_app(exp.app, verify=True)
            app_ids[i] = app_id
            print(f"  [#{i} {app_label}] Registered: {app_id}")

        reg_results = await asyncio.gather(
            *[_register(i) for i in batch],
            return_exceptions=True,
        )
        for i, result in zip(batch, reg_results):
            if isinstance(result, BaseException):
                results[i] = _make_error_result(i, result)
                print(f"  [#{i}] FAILED during registration: {result}")

        # Scale registered experiments sequentially under lock so that
        # concurrent batches don't race on the resource snapshot.
        for i in [j for j in batch if j in app_ids and j not in results]:
            app_label = type(experiments[i].app).__name__
            async with _snapshot_lock:
                try:
                    print(f"  [#{i} {app_label}] Scaling...")
                    await scale(experiments[i])
                except Exception as e:
                    results[i] = _make_error_result(i, e)
                    print(f"  [#{i}] FAILED during scale: {e}")

        # Wait for ready + sample inputs concurrently
        ready_indices = [i for i in batch if i in app_ids and i not in results]
        for i in ready_indices:
            exp = experiments[i]
            sessions[i] = BenchmarkSession(
                experiment=exp,
                app_id=app_ids[i],
                request_rate=exp.request_rate,
            )

        if ready_indices:
            # Run ready-wait concurrently with input sampling. Sample sequentially to
            # avoid concurrent writes to the same dataset cache file when
            # multiple experiments share the same dataset configuration.
            ready_tasks = [
                asyncio.create_task(
                    _wait_for_ready(
                        app_ids[i], f"#{i} {type(experiments[i].app).__name__}"
                    )
                )
                for i in ready_indices
            ]
            sample_results = []
            for i in ready_indices:
                try:
                    await sessions[i].sample_inputs()
                    sample_results.append(None)
                except Exception as e:
                    sample_results.append(e)

            ready_results = await asyncio.gather(
                *ready_tasks,
                return_exceptions=True,
            )

            for i, rr, sr in zip(ready_indices, ready_results, sample_results):
                if isinstance(rr, BaseException):
                    results[i] = _make_error_result(i, rr)
                    del sessions[i]
                    print(f"  [#{i}] FAILED waiting for ready: {rr}")
                elif isinstance(sr, BaseException):
                    results[i] = _make_error_result(i, sr)
                    del sessions[i]
                    print(f"  [#{i}] FAILED during input sampling: {sr}")

        # ---- Phase 1: Setup (concurrent) ----
        print("\n--- Phase 1: Setup (log streaming, port-fwd, test request) ---")
        setup_indices = [i for i in ready_indices if i in sessions and i not in results]

        if setup_indices:
            setup_results = await asyncio.gather(
                *[sessions[i].setup() for i in setup_indices],
                return_exceptions=True,
            )
            for i, result in zip(setup_indices, setup_results):
                if isinstance(result, BaseException):
                    results[i] = _make_error_result(i, result)
                    del sessions[i]
                    print(f"  [#{i}] FAILED during setup: {result}")

        # ==== BARRIER: all setups complete ====

        # ---- Phase 2: Dispatch (concurrent) ----
        print("\n--- Phase 2: Dispatch (all experiments fire concurrently) ---")
        dispatch_indices = list(sessions.keys())
        if dispatch_indices:
            dispatch_results = await asyncio.gather(
                *[sessions[i].dispatch() for i in dispatch_indices],
                return_exceptions=True,
            )
            for i, result in zip(dispatch_indices, dispatch_results):
                if isinstance(result, BaseException):
                    results[i] = _make_error_result(i, result)
                    print(f"  [#{i}] FAILED during dispatch: {result}")

        # ==== BARRIER: all dispatches complete ====

        # ---- Phase 3: Collect (concurrent) ----
        print("\n--- Phase 3: Collect (OTel traces, metrics) ---")
        collect_indices = [i for i in dispatch_indices if i not in results]
        if collect_indices:
            collect_results = await asyncio.gather(
                *[sessions[i].collect() for i in collect_indices],
                return_exceptions=True,
            )
            for i, result in zip(collect_indices, collect_results):
                if isinstance(result, BaseException):
                    # Jaeger unreachable → abort the entire schedule
                    if isinstance(result, RuntimeError) and "Jaeger API is unreachable" in str(result):
                        raise result
                    results[i] = _make_error_result(i, result)
                    print(f"  [#{i}] FAILED during collect: {result}")
                else:
                    results[i] = result
                    _print_experiment_summary(experiments[i], result)

    finally:
        # ---- Phase 4: Teardown + Unregister (always runs) ----
        print("\n--- Phase 4: Teardown + Unregister ---")

        # Teardown all sessions that were created
        if sessions:
            teardown_results = await asyncio.gather(
                *[sessions[i].teardown() for i in sessions],
                return_exceptions=True,
            )
            for i, result in zip(sessions, teardown_results):
                if isinstance(result, BaseException):
                    print(f"  [#{i}] Warning: teardown failed: {result}")

        # Unregister all apps (under lock)
        for i in batch:
            if i in app_ids:
                app_label = type(experiments[i].app).__name__
                async with _snapshot_lock:
                    try:
                        await unregister_app(app_ids[i])
                        print(f"  [#{i} {app_label}] Unregistered {app_ids[i]}")
                    except Exception as e:
                        print(
                            f"  [#{i} {app_label}] Warning: "
                            f"failed to unregister {app_ids[i]}: {e}"
                        )

    return results


def _make_error_result(i: int, exc: BaseException) -> dict[str, Any]:
    """Create an error result dict, saving the full traceback to a temp file."""
    err_file = tempfile.NamedTemporaryFile(
        prefix=f"experiment_{i}_",
        suffix=".err",
        delete=False,
        mode="w",
    )
    tb.print_exception(type(exc), exc, exc.__traceback__, file=err_file)
    err_file.close()
    return {"error": str(exc), "error_file": err_file.name}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    experiments: list[Experiment],
    results: dict[int, dict[str, Any]],
    gpu_counts: list[int],
    skipped: list[int],
    wall_duration: float,
) -> None:
    """Print a final table summarising all experiments."""

    def _fit(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    config_labels = [_experiment_label(exp.app) for exp in experiments]
    dataset_labels = [_dataset_label(exp.dataset) for exp in experiments]

    CW = max(37, min(68, max((len(s) for s in config_labels), default=37)))
    DW = max(24, min(36, max((len(s) for s in dataset_labels), default=24)))

    header = (
        f"{'#':>3s}  {'Config':<{CW}s}  {'Dataset':<{DW}s}  "
        f"{'Resolution':>10s}  {'Rate':>10s}  "
        f"{'GPU':>3s}  {'Status':<10s}  "
        f"{'Requests':>12s}  {'OTel':>12s}  "
        f"{'Throughput':>11s}  {'Success%':>8s}  {'Duration':>8s}  {'RunBS':>8s}"
    )
    WIDTH = len(header)
    print(f"\n{'=' * WIDTH}")
    print("Experiment Results Summary".center(WIDTH))
    print(f"{'=' * WIDTH}")
    print(header)
    print("-" * WIDTH)

    completed = 0
    failed = 0
    skipped_set = set(skipped)

    for i, exp in enumerate(experiments):
        label = _fit(_experiment_label(exp.app), CW)
        ds_label = _fit(_dataset_label(exp.dataset), DW)
        gc = gpu_counts[i]

        if i in skipped_set:
            status = "SKIPPED"
            reqs = "--"
            otel = "--"
            tput = "--"
            succ = "--"
            dur = "--"
            run_bs = "--"
        elif i in results:
            data = results[i]
            if "error" in data and "throughput" not in data:
                status = "FAILED"
                reqs = "--"
                otel = "--"
                tput = "--"
                succ = "--"
                dur = "--"
                run_bs = "--"
                failed += 1
            else:
                status = "POD_FAILURE" if data.get("pod_failure") else "COMPLETED"
                if data.get("pod_failure"):
                    failed += 1
                n_ok = data["num_successful"]
                n_total = n_ok + data["num_failed"]
                reqs = f"{n_ok}/{n_total}"
                otel_total = data.get("otel_metrics", {}).get("total_requests", 0)
                otel = f"{otel_total}/{n_total}"
                tput = f"{data['throughput']:.2f} req/s"
                succ = f"{100 * n_ok / n_total:.0f}%" if n_total > 0 else "--"
                dur = f"{data['benchmark_duration']:.1f}s"
                running_bs = data.get("prometheus_metrics", {}).get(
                    "vllm:num_requests_running"
                )
                if isinstance(running_bs, (float, int)):
                    run_bs = f"{running_bs:.1f}"
                elif isinstance(
                    exp.app,
                    (
                        DummyLLMApp,
                        MonolithicLLMApp,
                        PrefillLLMApp,
                        DecodeLLMApp,
                        ModServeApp,
                        TimeSharingApp,
                        MixedMLLMApp,
                        GroupedMixedMLLMApp,
                        MLLMRouterApp,
                    ),
                ):
                    run_bs = "N/A"
                else:
                    run_bs = "--"
                completed += 1
        else:
            status = "NOT RUN"
            reqs = "--"
            otel = "--"
            tput = "--"
            succ = "--"
            dur = "--"
            run_bs = "--"

        rate = f"{exp.request_rate:.1f} req/s"
        ds = exp.dataset
        if isinstance(ds, ControlledImageChat):
            res = f"{ds.image_width}x{ds.image_height}"
        elif isinstance(ds, ControlledDiT) and len(ds.resolution_pool) == 1:
            h, w = ds.resolution_pool[0]
            res = f"{h}x{w}"
        else:
            res = "--"
        print(
            f"{i:3d}  {label:<{CW}s}  {ds_label:<{DW}s}  "
            f"{res:>10s}  {rate:>10s}  "
            f"{gc:3d}  {status:<10s}  "
            f"{reqs:>12s}  {otel:>12s}  "
            f"{tput:>10s}  {succ:>8s}  {dur:>8s}  {run_bs:>8s}"
        )

    print(f"{'=' * WIDTH}")
    total = len(experiments)
    print(
        f"Total: {total} experiments, "
        f"{completed} completed, {failed} failed, "
        f"{len(skipped)} skipped"
    )
    if wall_duration > 0:
        print(f"Total wall-clock time: {wall_duration:.1f}s")
    print(f"{'=' * WIDTH}")
