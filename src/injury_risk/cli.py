"""The project's single entry point.

Every stage of the pipeline used to carry its own ``argparse`` block — five
near-identical parsers, five copies of ``--track`` and ``--seed``, and a README that
had to spell out five different ``python -m ...`` incantations.

Everything now hangs off one command:

    injury-risk --help
    injury-risk data
    injury-risk train --track synthetic
    injury-risk dashboard

This matters beyond convenience: the CI smoke test (#24) and the Docker image (#30)
both need to run the pipeline. With a single entry point they call *this*, instead of
each re-encoding the sequence of steps and drifting apart from it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from injury_risk.config import (
    DEFAULT_SEED,
    N_ATHLETES,
    N_DAYS,
    ROOT,
    SYNTHETIC_DATASET,
    TUNING_N_ITER,
)
from injury_risk.data.datasets import TRACKS

app = typer.Typer(
    help="Athlete injury risk detection — data, models, explainability, dashboard.",
    no_args_is_help=True,
    add_completion=False,
)

TrackArg = str  # "synthetic" | "real" | "both"


def _tracks(track: str) -> list[str]:
    if track == "both":
        return list(TRACKS)
    if track not in TRACKS:
        raise typer.BadParameter(f"unknown track {track!r} (expected one of {TRACKS} or 'both')")
    return [track]


@app.command()
def download(
    datasets: list[str] = typer.Argument(None, help="Dataset keys; all of them if omitted."),
) -> None:
    """Download the raw Kaggle datasets (needs a Kaggle token)."""
    from injury_risk.data.download import DATASETS
    from injury_risk.data.download import download as fetch

    keys = datasets or list(DATASETS)
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise typer.BadParameter(f"unknown key(s) {unknown}. Available: {list(DATASETS)}")
    for key in keys:
        fetch(key, DATASETS[key])


@app.command()
def data(
    athletes: int = N_ATHLETES,
    days: int = N_DAYS,
    seed: int = DEFAULT_SEED,
    output: Path = SYNTHETIC_DATASET,
) -> None:
    """Generate the synthetic dataset (athletes, daily series, injury events)."""
    from injury_risk.config import TARGET_COL
    from injury_risk.data.generate_synthetic import generate

    df = generate(n_athletes=athletes, n_days=days, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    injuries = int(df["injury_onset"].sum())
    modelled = df[~df["is_injured"] & df["horizon_complete"]]
    typer.echo(f"Synthetic dataset: {len(df)} rows, {df['athlete_id'].nunique()} athletes")
    typer.echo(f"Injury events    : {injuries} ({injuries / athletes:.1f} per athlete)")
    typer.echo(f"Days sidelined   : {df['is_injured'].mean():.1%} of all athlete-days")
    typer.echo(f"Target           : {modelled[TARGET_COL].mean():.2%} positive")
    typer.echo(f"Written to       : {output}")


@app.command()
def train(
    track: TrackArg = "both",
    seed: int = DEFAULT_SEED,
    tuned: bool = typer.Option(False, help="Use tuned hyperparameters when available."),
    model: str = typer.Option(None, help="Defaults to the tuned benchmark winner for the track."),
    calibrate: bool = typer.Option(True, help="Calibrate probabilities (isotonic)."),
) -> None:
    """Train, calibrate and threshold the model(s); write the metrics report."""
    from injury_risk.models.candidates import CANDIDATES
    from injury_risk.models.train import train_track

    if model is not None and model not in CANDIDATES:
        raise typer.BadParameter(f"unknown model {model!r} (expected one of {CANDIDATES})")
    for name in _tracks(track):
        train_track(name, seed=seed, tuned=tuned, model=model, calibrate=calibrate)


@app.command()
def tune(
    track: TrackArg = "both",
    n_iter: int = TUNING_N_ITER,
    seed: int = DEFAULT_SEED,
) -> None:
    """Search hyperparameters for every candidate (PR-AUC oriented)."""
    from injury_risk.models.tune import tune_track

    for name in _tracks(track):
        tune_track(name, n_iter=n_iter, seed=seed)


@app.command()
def benchmark(
    track: TrackArg = "both",
    seed: int = DEFAULT_SEED,
    tuned: bool = typer.Option(False, help="Compare the *tuned* versions."),
) -> None:
    """Compare the candidates (LogReg / RandomForest / XGBoost) on one protocol."""
    from injury_risk.models.benchmark import benchmark_track

    for name in _tracks(track):
        benchmark_track(name, seed=seed, tuned=tuned)


@app.command()
def shap(
    track: str = "synthetic",
    index: int = typer.Option(0, help="Row explained by the waterfall plot."),
    n: int = 500,
) -> None:
    """Generate the SHAP plots (global summary + individual waterfall)."""
    from injury_risk.visualization.shap_plots import save_summary_plot, save_waterfall_plot

    save_summary_plot(track, n=n)
    save_waterfall_plot(track, index=index, n=n)


@app.command()
def dashboard(port: int = 8501) -> None:
    """Launch the Streamlit dashboard."""
    app_path = ROOT / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        check=True,
    )


if __name__ == "__main__":
    app()
