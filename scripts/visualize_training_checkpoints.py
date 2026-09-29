#!/usr/bin/env python3
"""
visualize_training_checkpoints.py

Liest Trainingsmetriken direkt aus den Checkpoints (.pt) aus und visualisiert
die finale Trainingsgenauigkeit am Ende jedes Trainingsabschnitts (Stage 1, 2, 3).
Das Design und die Farbgebung entsprechen exakt dem Plot `final_checkpoint_accuracies_by_task`.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml

# Konsistente Farbpalette wie in den Test-Auswertungen
TASK_COLORS = {
    "A": "#2b5c8f",  # Blau (Aquatic Mammals)
    "B": "#d95f02",  # Orange (Electronics)
    "C": "#2ca02c",  # Grün (Vehicles)
}

OUTPUT_DIR = Path("figures/training_summary")


def get_checkpoint_dirs():
    """Lädt Checkpoint-Pfade aus experiment.yaml oder sucht dynamisch."""
    repo_root = Path.cwd()
    config_path = repo_root / "configs" / "experiment.yaml"

    single_dir = None
    seq_dir = None

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            paths = cfg.get("paths", {})
            if "single_checkpoints" in paths:
                single_dir = repo_root / paths["single_checkpoints"]
            if "sequential_checkpoints" in paths:
                seq_dir = repo_root / paths["sequential_checkpoints"]

    candidates_single = [
        single_dir,
        repo_root / "checkpoints" / "single",
        repo_root / "checkpoints" / "verified_initialization" / "single",
    ]
    candidates_seq = [
        seq_dir,
        repo_root / "checkpoints" / "sequential",
        repo_root / "checkpoints" / "verified_initialization" / "sequential",
    ]

    resolved_single = next((p for p in candidates_single if p and p.exists()), None)
    resolved_seq = next((p for p in candidates_seq if p and p.exists()), None)

    return resolved_single, resolved_seq


def inspect_single_checkpoints(single_dir: Path) -> pd.DataFrame:
    """Extrahiert Metriken aus den Single-Task-Referenz-Checkpoints."""
    records = []
    if not single_dir or not single_dir.exists():
        return pd.DataFrame()

    for ckpt_path in sorted(single_dir.glob("task_*.pt")):
        data = torch.load(ckpt_path, map_location="cpu")
        task = data.get("task", ckpt_path.stem.replace("task_", ""))
        metrics = data.get("metrics", {})
        records.append(
            {
                "Model Type": "Single-Task",
                "Task": f"Task {task}",
                "Raw Task": task,
                "Checkpoint": ckpt_path.name,
                "Test Accuracy (%)": metrics.get("test_accuracy", 0.0) * 100.0,
                "Test Loss": metrics.get("test_loss", float("nan")),
            }
        )
    return pd.DataFrame(records)


def inspect_sequential_checkpoints(seq_dir: Path) -> pd.DataFrame:
    """Extrahiert Trainingsmetriken aus den sequentiellen Checkpoints."""
    records = []
    if not seq_dir or not seq_dir.exists():
        return pd.DataFrame()

    for order_dir in sorted([p for p in seq_dir.iterdir() if p.is_dir()]):
        order_sequence = order_dir.name.split("_")
        formatted_order = " → ".join(order_sequence)

        for ckpt_path in sorted(order_dir.glob("*.pt")):
            data = torch.load(ckpt_path, map_location="cpu")
            metrics = data.get("metrics", {})
            history_order = data.get("order", [])
            stage = len(history_order)
            current_task = data.get("task", history_order[-1] if history_order else "N/A")

            records.append(
                {
                    "Order": formatted_order,
                    "Raw Order": order_dir.name,
                    "Stage": stage,
                    "Stage Name": ckpt_path.stem,
                    "Task": f"Task {current_task}",
                    "Raw Task": current_task,
                    "Train Accuracy (%)": metrics.get("train_accuracy", 0.0) * 100.0,
                    "Train Loss": metrics.get("train_loss", float("nan")),
                    "Duration (min)": metrics.get("duration_seconds", 0.0) / 60.0,
                    "Checkpoint": ckpt_path.name,
                }
            )
    return pd.DataFrame(records)


def plot_train_accuracies_by_task(df_seq: pd.DataFrame, output_dir: Path):
    """
    Erstellt ein Balkendiagramm der finalen Trainingsgenauigkeit am Ende jeder Stage,
    formatiert im selben visuellen Layout wie `final_checkpoint_accuracies_by_task.png`.
    """
    if df_seq.empty:
        print("[!] Keine sequentiellen Daten zum Plotten vorhanden.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig, ax = plt.subplots(figsize=(13, 6))

    sns.barplot(
        data=df_seq,
        x="Order",
        y="Train Accuracy (%)",
        hue="Task",
        palette=[TASK_COLORS["A"], TASK_COLORS["B"], TASK_COLORS["C"]],
        ax=ax,
    )

    # Zufallsniveau (Chance Level für 5 Klassen) als Referenz
    ax.axhline(20.0, linestyle="--", color="gray", alpha=0.7, label="Chance Level (20%)")

    ax.set_title(
        "Final Training Accuracy at Each Stage by Learned Task (A, B, C)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Final Train Accuracy (%)", fontsize=11)
    ax.set_xlabel("Sequential Training Order", fontsize=11)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20)
    ax.legend(title="Trained Task", loc="upper right", framealpha=0.9)

    # Exakte Werte über die Balken schreiben
    for p in ax.patches:
        val = p.get_height()
        if val > 1.0:
            ax.annotate(
                f"{val:.1f}%",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="center",
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="semibold",
            )

    plt.tight_layout()
    save_path = output_dir / "sequential_stages_train_accuracy_by_task.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {save_path.resolve()}")


def main():
    print("=" * 80)
    print("EXTRAKTION UND VISUALISIERUNG DER CHECKPOINT-TRAININGSDATEN")
    print("=" * 80)

    single_dir, seq_dir = get_checkpoint_dirs()
    print(f"Gefundenes Verzeichnis (Single):     {single_dir}")
    print(f"Gefundenes Verzeichnis (Sequential): {seq_dir}")

    df_single = inspect_single_checkpoints(single_dir)
    if not df_single.empty:
        print("\n[1] Single-Task Referenzmodelle:")
        print(df_single[["Task", "Test Accuracy (%)", "Test Loss"]].to_string(index=False))

    df_seq = inspect_sequential_checkpoints(seq_dir)
    if not df_seq.empty:
        print("\n[2] Sequentielle Trainings-Checkpoints:")
        cols = ["Order", "Stage", "Task", "Train Accuracy (%)", "Duration (min)"]
        print(df_seq[cols].to_string(index=False))

    print("\n[3] Generiere Präsentationsgrafik...")
    plot_train_accuracies_by_task(df_seq, OUTPUT_DIR)
    print("=" * 80)
    print("Fertig. Grafik abgelegt unter:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()