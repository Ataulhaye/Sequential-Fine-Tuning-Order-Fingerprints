from sequential_finetuning.config import load_config


def main():

    config = load_config("configs/experiment.yaml")

    print("=" * 70)
    print("Experiment Configuration")
    print("=" * 70)

    print(f"Seed: {config['seed']}")

    print(f"Model: " f"{config['model']['architecture']}")

    print(f"Epochs per task: " f"{config['training']['epochs_per_task']}")

    print(f"Batch size: " f"{config['training']['batch_size']}")

    print()
    print("Sequential orders:")

    for order in config["experiment"]["sequential_orders"]:

        print(" → ".join(order))


if __name__ == "__main__":
    main()
