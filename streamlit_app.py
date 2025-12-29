from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import streamlit as st

PLOTS_DIR = Path(__file__).parent / "plots"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf"}
DATA_PATH = Path(__file__).parent / "minibatch_samples_20.json"
HIST_PATH = Path(__file__).parent / "minibatch_score_hist.png"

TEMP_RE = re.compile(
    r"(?:^|[,_\s/-])t(?:emp(?:erature)?)?(?:=)?"
    r"([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)(?=$|[,_\s/-])"
)
NUM_SAMPLES_RE = re.compile(
    r"(?:^|[,_\s/-])n(?:um(?:[_-]?samples)?)?(?:=)?([0-9]+)"
)
PASS_AT_RE = re.compile(r"(?:pass@|pass[_-]?at[_-]?)(\{?[0-9,\s]+\}?)")
EVAL_RE = re.compile(r"(?:^|[,_\s/-])(?:eval(?:uation)?|task)=([A-Za-z0-9._-]+)")


@dataclass(frozen=True)
class PlotRecord:
    path: Path
    eval_name: str
    temperature_label: str
    temperature_val: float | None
    num_samples_label: str
    num_samples_val: int | None
    pass_at_label: str
    pass_at_key: tuple[int, ...]


def parse_temperature(value: str) -> tuple[str, float | None]:
    normalized = value.replace("p", ".") if "p" in value and "." not in value else value
    try:
        num = float(normalized)
    except ValueError:
        return value, None
    return format(num, "g"), num


def normalize_pass_at(raw: str) -> tuple[str, tuple[int, ...]]:
    cleaned = raw.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1]
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not parts:
        return "Unknown", tuple()
    if len(parts) == 1:
        label = f"pass@{parts[0]}"
    else:
        label = f"pass@{{{','.join(parts)}}}"
    key: list[int] = []
    for part in parts:
        try:
            key.append(int(part))
        except ValueError:
            continue
    return label, tuple(key)


def scan_plots() -> list[PlotRecord]:
    if not PLOTS_DIR.exists():
        return []
    records: list[PlotRecord] = []
    for path in sorted(PLOTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        rel_path = path.relative_to(PLOTS_DIR)
        eval_name = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        search_text = path.as_posix()
        if eval_name == "root":
            eval_match = EVAL_RE.search(search_text)
            if eval_match:
                eval_name = eval_match.group(1)

        temp_match = TEMP_RE.search(search_text)
        if temp_match:
            temp_label, temp_val = parse_temperature(temp_match.group(1))
            temperature_label = f"t={temp_label}"
            temperature_val = temp_val
        else:
            temperature_label = "Unknown"
            temperature_val = None

        num_match = NUM_SAMPLES_RE.search(search_text)
        if num_match:
            num_samples_val = int(num_match.group(1))
            num_samples_label = f"num_samples={num_samples_val}"
        else:
            num_samples_label = "Unknown"
            num_samples_val = None

        pass_match = PASS_AT_RE.search(search_text)
        if pass_match:
            pass_at_label, pass_at_key = normalize_pass_at(pass_match.group(1))
        else:
            pass_at_label, pass_at_key = "Unknown", tuple()

        records.append(
            PlotRecord(
                path=path,
                eval_name=eval_name,
                temperature_label=temperature_label,
                temperature_val=temperature_val,
                num_samples_label=num_samples_label,
                num_samples_val=num_samples_val,
                pass_at_label=pass_at_label,
                pass_at_key=pass_at_key,
            )
        )
    return records


def sorted_labels_by_temperature(records: list[PlotRecord]) -> list[str]:
    items = {record.temperature_label: record.temperature_val for record in records}
    return [
        label
        for label, _ in sorted(
            items.items(),
            key=lambda item: (
                item[1] is None,
                item[1] if item[1] is not None else float("inf"),
                item[0],
            ),
        )
    ]


def sorted_labels_by_num_samples(records: list[PlotRecord]) -> list[str]:
    items = {record.num_samples_label: record.num_samples_val for record in records}
    return [
        label
        for label, _ in sorted(
            items.items(),
            key=lambda item: (
                item[1] is None,
                item[1] if item[1] is not None else float("inf"),
                item[0],
            ),
        )
    ]


def sorted_labels_by_pass_at(records: list[PlotRecord]) -> list[str]:
    items = {record.pass_at_label: record.pass_at_key for record in records}
    return [
        label
        for label, _ in sorted(
            items.items(),
            key=lambda item: (len(item[1]) == 0, item[1], item[0]),
        )
    ]


def render_plot(record: PlotRecord) -> None:
    ext = record.path.suffix.lower()
    rel_path = record.path.relative_to(PLOTS_DIR).as_posix()
    st.markdown(f"**{rel_path}**")
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
        st.image(str(record.path), use_column_width=True)
    elif ext == ".pdf":
        st.download_button(
            label=f"Download {record.path.name}",
            data=record.path.read_bytes(),
            file_name=record.path.name,
            mime="application/pdf",
        )
    else:
        st.warning(f"Unsupported file type: {ext}")


@st.cache_data(show_spinner=False)
def load_minibatch_samples(data_path: Path, cache_key: float) -> dict[str, Any]:
    with data_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def sort_bin_labels(labels: Iterable[str]) -> list[str]:
    def parse_range(label: str) -> tuple[float, float]:
        try:
            start, end = label.split("-")
            return float(start), float(end)
        except Exception:
            return float("inf"), float("inf")

    return [label for label in sorted(labels, key=parse_range)]


def render_coding_experiments_tab() -> None:
    st.subheader("Coding Experiments")
    st.write(
        "Put plot images in the `plots/` folder and include `t=...` or `t0p6`, "
        "`num_samples=...` or `n200`, and `pass@...` or `pass_at_...` in the filename "
        "or parent folders so the filters can find them."
    )

    records = scan_plots()
    if not records:
        st.info("No plot files found in `plots/` yet.")
        return

    eval_names = sorted({record.eval_name for record in records})
    selected_eval = st.sidebar.selectbox("Evaluation", eval_names)

    filtered = [record for record in records if record.eval_name == selected_eval]
    if not filtered:
        st.warning("No plots for this evaluation.")
        return

    temp_options = sorted_labels_by_temperature(filtered)
    selected_temp = st.sidebar.selectbox("Temperature", temp_options)
    filtered = [record for record in filtered if record.temperature_label == selected_temp]
    if not filtered:
        st.warning("No plots for this temperature.")
        return

    num_options = sorted_labels_by_num_samples(filtered)
    selected_num = st.sidebar.selectbox("Num samples", num_options)
    filtered = [record for record in filtered if record.num_samples_label == selected_num]
    if not filtered:
        st.warning("No plots for this num_samples value.")
        return

    pass_options = sorted_labels_by_pass_at(filtered)
    selected_pass = st.sidebar.selectbox("pass@", pass_options)
    filtered = [record for record in filtered if record.pass_at_label == selected_pass]

    if not filtered:
        st.warning("No plots for this pass@ value.")
        return

    if len(filtered) > 1:
        file_labels = [record.path.relative_to(PLOTS_DIR).as_posix() for record in filtered]
        selected_file = st.selectbox("Matching plots", file_labels)
        filtered = [
            record
            for record in filtered
            if record.path.relative_to(PLOTS_DIR).as_posix() == selected_file
        ]

    for record in filtered:
        render_plot(record)


def render_api_scoring_tab() -> None:
    st.subheader("API Scoring")
    st.write(
        "Explore minibatch score distributions and inspect saved completions by score bin."
    )

    if HIST_PATH.exists():
        st.image(str(HIST_PATH), caption="Minibatch score distribution", use_column_width=True)
    else:
        st.info("Add `minibatch_score_hist.png` to show the score distribution plot.")

    if not DATA_PATH.exists():
        st.warning("`minibatch_samples_20.json` is missing.")
        return

    try:
        data = load_minibatch_samples(DATA_PATH, DATA_PATH.stat().st_mtime)
    except json.JSONDecodeError as exc:
        st.error(f"Could not parse `minibatch_samples_20.json`: {exc}")
        return

    samples_by_bin: dict[str, list[dict[str, Any]]] = data.get("samples") or {}
    bin_counts: dict[str, int] = data.get("bin_counts_0_5") or {}
    if not samples_by_bin:
        st.info("No samples found in `minibatch_samples_20.json` yet.")
        return

    bins = sort_bin_labels(samples_by_bin.keys())
    selected_bin = st.selectbox("Score bin", bins)
    bin_samples = samples_by_bin.get(selected_bin, [])
    total_bin_count = bin_counts.get(selected_bin, "unknown")
    st.caption(f"{len(bin_samples)} saved samples for {selected_bin} (count in dataset: {total_bin_count})")

    if not bin_samples:
        st.info("No saved completions for this bin.")
        return

    sample_idx = st.number_input(
        "Sample to display",
        min_value=1,
        max_value=len(bin_samples),
        value=1,
        step=1,
        key=f"{selected_bin}_sample_index",
    )
    sample = bin_samples[int(sample_idx) - 1]

    col_score, col_batch, col_minibatch = st.columns(3)
    col_score.metric("Score", f"{sample.get('score', 0):.3f}")
    col_batch.metric("Batch ID", str(sample.get("batch_id", "—")))
    col_minibatch.metric("Minibatch ID", str(sample.get("minibatch_id", "—")))

    st.code(sample.get("text", ""), language="text")


st.set_page_config(page_title="Plot Viewer", layout="wide")

st.title("Plot Viewer")
coding_tab, api_tab = st.tabs(["Coding Experiments", "API Scoring"])

with coding_tab:
    render_coding_experiments_tab()

with api_tab:
    render_api_scoring_tab()
