import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18MultiTask(nn.Module):
    """
    ResNet18 with a shared backbone and one classifier head
    for each task.

    The backbone produces a 512-dimensional representation.

    Each task has its own 5-class classification head.
    """

    def __init__(
        self,
        num_classes_per_task: int = 5,
        pretrained: bool = False,
    ):
        super().__init__()

        # ----------------------------------------------------
        # ResNet18 backbone
        # ----------------------------------------------------

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet18(weights=weights)

        # Adapt ImageNet ResNet18 for CIFAR-sized 32x32 images.
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        backbone.maxpool = nn.Identity()

        # Save the original classification dimension.
        feature_dim = backbone.fc.in_features

        # Remove the original ImageNet classifier.
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.feature_dim = feature_dim

        # ----------------------------------------------------
        # Task-specific classification heads
        # ----------------------------------------------------

        self.heads = nn.ModuleDict(
            {
                "A": nn.Linear(
                    feature_dim,
                    num_classes_per_task,
                ),
                "B": nn.Linear(
                    feature_dim,
                    num_classes_per_task,
                ),
                "C": nn.Linear(
                    feature_dim,
                    num_classes_per_task,
                ),
            }
        )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    def extract_features(self, x):
        """
        Return the shared ResNet18 representation.
        """

        return self.backbone(x)

    # --------------------------------------------------------
    # Task-specific forward pass
    # --------------------------------------------------------

    def forward(self, x, task):
        """
        Forward pass for a particular task.

        Parameters
        ----------
        x:
            Input images.

        task:
            Task identifier: "A", "B", or "C".
        """

        if task not in self.heads:
            raise ValueError(
                f"Unknown task '{task}'. " f"Available tasks: {list(self.heads.keys())}"
            )

        features = self.extract_features(x)

        logits = self.heads[task](features)

        return logits

    # --------------------------------------------------------
    # Representation + logits
    # --------------------------------------------------------

    def forward_with_features(self, x, task):
        """
        Return both representation and classification logits.

        Useful later for CKA and feature-drift analysis.
        """

        if task not in self.heads:
            raise ValueError(
                f"Unknown task '{task}'. " f"Available tasks: {list(self.heads.keys())}"
            )

        features = self.extract_features(x)

        logits = self.heads[task](features)

        return features, logits

    def backbone_parameters(self):
        """
        Return parameters belonging only to the shared backbone.
        """

        return self.backbone.parameters()
