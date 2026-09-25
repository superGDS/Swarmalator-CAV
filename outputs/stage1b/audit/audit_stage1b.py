"""Independent, read-only Stage 1B audit.

This script writes only below ``outputs/stage1b/audit``.  It hashes the current
workspace files, compares them with the feedback zip snapshot, reruns the
existing four tests outside the script's scope, and calls ``run_episode``
directly for the configured 90-run matrix.  It deliberately does not invoke
``run_stage1`` or write any Stage 1 outputs/reports.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "outputs" / "stage1b" / "audit"
ZIP_PATH = ROOT / "feedback" / "stage1_feedback.zip"
OLD_METRICS_PATH = ROOT / "outputs" / "stage1" / "metrics.csv"
CONFIG_PATH = ROOT / "configs" / "stage1_config.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_current_evidence() -> list[Path]:
    """Return current evidence files while excluding virtualenv/cache/audit."""

    paths: set[Path] = set()
    for directory in ("outputs/stage1", "docs", "src", "tests", "configs", "reports"):
        base = ROOT / directory
        if base.exists():
            paths.update(p for p in base.rglob("*") if p.is_file())
    for name in (
        "AGENTS.md",
        "README.md",
        "codex_swarmalator_stage1_prompt.md",
        "feedback/stage1_feedback.zip",
        "feedback/stage1_feedback_manifest.txt",
        "references/literature_notes.md",
    ):
        path = ROOT / name
        if path.is_file():
            paths.add(path)
    return sorted(paths, key=rel)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_current_hashes() -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for path in iter_current_evidence():
        item = {
            "path": rel(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        rows.append(item)
        index[item["path"]] = item
    write_csv(AUDIT / "current_sha256.csv", rows, ["path", "size_bytes", "sha256"])
    return index


def compare_feedback_snapshot(current: dict[str, dict[str, Any]]) -> None:
    archive: dict[str, tuple[int, str, bytes]] = {}
    with zipfile.ZipFile(ZIP_PATH) as bundle:
        for info in bundle.infolist():
            data = bundle.read(info.filename)
            archive[info.filename] = (info.file_size, sha256_bytes(data), data)

    rows: list[dict[str, Any]] = []
    diff_chunks: list[str] = []
    names = sorted(set(current) | set(archive))
    for name in names:
        current_item = current.get(name)
        archive_item = archive.get(name)
        if archive_item is None:
            status = "current_only"
            archive_size = ""
            archive_sha = ""
            current_size = current_item["size_bytes"]
            current_sha = current_item["sha256"]
        elif current_item is None:
            status = "archive_only"
            archive_size = archive_item[0]
            archive_sha = archive_item[1]
            current_size = ""
            current_sha = ""
        else:
            archive_size = archive_item[0]
            archive_sha = archive_item[1]
            current_size = current_item["size_bytes"]
            current_sha = current_item["sha256"]
            status = "same" if current_sha == archive_sha else "changed"
            if status == "changed":
                try:
                    old_text = archive_item[2].decode("utf-8").splitlines(keepends=True)
                    new_text = (ROOT / Path(name)).read_text(encoding="utf-8").splitlines(keepends=True)
                except (UnicodeDecodeError, OSError):
                    diff_chunks.append(f"\n===== {name} (binary/hash-only difference) =====\n")
                else:
                    diff_chunks.append(f"\n===== {name} =====\n")
                    diff_chunks.extend(
                        difflib.unified_diff(
                            old_text,
                            new_text,
                            fromfile=f"feedback_zip/{name}",
                            tofile=f"current/{name}",
                        )
                    )
        rows.append(
            {
                "path": name,
                "status": status,
                "current_size_bytes": current_size,
                "archive_size_bytes": archive_size,
                "current_sha256": current_sha,
                "archive_sha256": archive_sha,
            }
        )
    write_csv(
        AUDIT / "feedback_snapshot_comparison.csv",
        rows,
        [
            "path",
            "status",
            "current_size_bytes",
            "archive_size_bytes",
            "current_sha256",
            "archive_sha256",
        ],
    )
    (AUDIT / "feedback_snapshot_diff.log").write_text("".join(diff_chunks), encoding="utf-8")


def scalar_equal(old: str, new: Any) -> tuple[bool, str]:
    old_text = "" if old is None else str(old)
    new_text = "" if new is None else str(new)
    try:
        old_num = float(old_text)
        new_num = float(new_text)
    except (TypeError, ValueError):
        return old_text == new_text, ""
    if math.isnan(old_num) and math.isnan(new_num):
        return True, "0"
    delta = new_num - old_num
    equal = math.isclose(old_num, new_num, rel_tol=1e-9, abs_tol=1e-9)
    return equal, f"{delta:.17g}"


def reproduce_and_compare() -> None:
    # Import only after ROOT is known; this keeps the script runnable from any cwd.
    sys.path.insert(0, str(ROOT))
    from src.swarmalator_cav.simulation import SimConfig, run_episode  # noqa: PLC0415

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = SimConfig(**config.get("simulation", {}))
    methods = config.get("methods", ["A", "B", "C", "D", "E"])
    scenarios = config.get("scenarios", ["ample", "collaborative", "short_window"])
    disturbances = config.get("disturbances", ["none", "prepare"])
    init_variants = config.get("init_variants", [0, 1, 2])

    with OLD_METRICS_PATH.open(newline="", encoding="utf-8") as stream:
        old_rows = {row["run_id"]: row for row in csv.DictReader(stream)}

    current_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    run_errors: list[str] = []
    skipped_field = "compute_time_s"
    compared_fields: set[str] = set()

    for method in methods:
        for scenario in scenarios:
            for disturbance in disturbances:
                for init_variant in init_variants:
                    run_id = f"{method}_{scenario}_{disturbance}_i{init_variant}"
                    try:
                        result = run_episode(method, scenario, disturbance, int(init_variant), cfg=cfg)
                    except Exception as exc:  # pragma: no cover - audit should report an individual failure
                        run_errors.append(f"{run_id}: {type(exc).__name__}: {exc}")
                        continue
                    current_rows.append(dict(result.metrics))
                    old = old_rows.get(run_id)
                    run_mismatches = 0
                    run_compared = 0
                    max_abs_delta = 0.0
                    if old is None:
                        comparison_rows.append(
                            {
                                "run_id": run_id,
                                "field": "<row>",
                                "status": "current_row_without_old_row",
                                "old_value": "",
                                "current_value": json.dumps(result.metrics, ensure_ascii=False, sort_keys=True),
                                "delta": "",
                            }
                        )
                        run_mismatches += 1
                    else:
                        fields = sorted(set(old) | set(result.metrics))
                        for field in fields:
                            if field == skipped_field:
                                continue
                            compared_fields.add(field)
                            run_compared += 1
                            old_value = old.get(field, "")
                            current_value = result.metrics.get(field, "")
                            equal, delta = scalar_equal(old_value, current_value)
                            status = "same" if equal else "changed"
                            if not equal:
                                run_mismatches += 1
                                try:
                                    max_abs_delta = max(max_abs_delta, abs(float(delta)))
                                except ValueError:
                                    pass
                            comparison_rows.append(
                                {
                                    "run_id": run_id,
                                    "field": field,
                                    "status": status,
                                    "old_value": old_value,
                                    "current_value": current_value,
                                    "delta": delta,
                                }
                            )
                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "old_row_present": int(old is not None),
                            "current_row_present": 1,
                            "compared_field_count": run_compared,
                            "mismatch_count": run_mismatches,
                            "max_abs_numeric_delta": f"{max_abs_delta:.17g}",
                        }
                    )

    if current_rows:
        fields = sorted({key for row in current_rows for key in row})
        write_csv(AUDIT / "reproduction_current_metrics.csv", current_rows, fields)
    write_csv(
        AUDIT / "reproduction_comparison.csv",
        comparison_rows,
        ["run_id", "field", "status", "old_value", "current_value", "delta"],
    )
    write_csv(
        AUDIT / "reproduction_comparison_summary.csv",
        summary_rows,
        [
            "run_id",
            "old_row_present",
            "current_row_present",
            "compared_field_count",
            "mismatch_count",
            "max_abs_numeric_delta",
        ],
    )
    changed_count = sum(1 for row in comparison_rows if row["status"] == "changed")
    (AUDIT / "reproduction_comparison.log").write_text(
        "direct_call=src.swarmalator_cav.simulation.run_episode\n"
        f"python={sys.executable}\n"
        f"methods={methods}\nscenarios={scenarios}\ndisturbances={disturbances}\n"
        f"init_variants={init_variants}\nexpected_runs={len(methods)*len(scenarios)*len(disturbances)*len(init_variants)}\n"
        f"current_runs={len(current_rows)}\nold_csv_rows={len(old_rows)}\n"
        f"compared_fields={len(compared_fields)}\nchanged_cells={changed_count}\n"
        f"skipped_field={skipped_field}\nrun_errors={len(run_errors)}\n"
        + ("errors:\n" + "\n".join(run_errors) + "\n" if run_errors else "")
        + "\n"
    )


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    current = write_current_hashes()
    compare_feedback_snapshot(current)
    reproduce_and_compare()
    (AUDIT / "audit_scope.txt").write_text(
        "This audit wrote only outputs/stage1b/audit. It did not invoke run_stage1, overwrite outputs/stage1, or modify source/config/report files.\n"
        f"feedback_zip={ZIP_PATH}\nold_metrics={OLD_METRICS_PATH}\nconfig={CONFIG_PATH}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
