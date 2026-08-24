import torch

from sequential_finetuning.model import (
    ResNet18MultiTask,
)
from sequential_finetuning.seed import (
    set_seed,
)


def get_first_parameter():

    set_seed(42)

    model = ResNet18MultiTask(
        num_classes_per_task=5,
        pretrained=False,
    )

    first_parameter = next(model.parameters())

    return first_parameter.detach().clone()


def main():

    parameter_a = get_first_parameter()

    parameter_b = get_first_parameter()

    identical = torch.equal(
        parameter_a,
        parameter_b,
    )

    print(f"Models initialized identically: {identical}")

    difference = (parameter_a - parameter_b).abs().max().item()

    print(f"Maximum parameter difference: " f"{difference}")


if __name__ == "__main__":
    main()
