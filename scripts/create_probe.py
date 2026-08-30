import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sequential_finetuning.config import load_config
from sequential_finetuning.probe import create_fixed_probe


def main():

    config = load_config("configs/experiment.yaml")

    dataset_root = config["dataset"]["root"]

    probe_config = config["representation"]["probe"]

    classes = []

    for task_classes in config["tasks"].values():
        classes.extend(task_classes)

    samples_per_class = probe_config["samples_per_class"]

    num_samples = len(classes) * samples_per_class

    probe_path = Path(config["paths"]["results"]) / "representation" / "probe_set.json"

    probe = create_fixed_probe(
        root=dataset_root,
        classes=classes,
        samples_per_class=probe_config["samples_per_class"],
        seed=probe_config["seed"],
        output_path=probe_path,
    )

    print("=" * 70)
    print("FIXED REPRESENTATION PROBE")
    print("=" * 70)

    print(f"Classes:            {probe['num_classes']}")
    print(f"Samples per class:  {probe['samples_per_class']}")
    print(f"Total probe images: { num_samples}")
    print(f"Seed:               {probe['seed']}")

    print()
    print("Probe classes:")

    for class_name in probe["classes"]:
        print(f"  {class_name}: " f"{len(probe['indices'][class_name])} images")

    print()
    print("Probe saved to:")
    print(probe_path)


if __name__ == "__main__":
    main()
