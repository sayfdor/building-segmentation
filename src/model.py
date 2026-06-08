import torch.nn as nn
import segmentation_models_pytorch as smp


class BuildingSegmentationModel(nn.Module):
    def __init__(self, architecture="Segformer", encoder_name="mit_b1", encoder_weights="imagenet", classes=1):
        super().__init__()

        if not hasattr(smp, architecture):
            raise ValueError(f"Architecture '{architecture}' not found in smp.")

        arch_class = getattr(smp, architecture)
        self.seg_model = arch_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=None
        )

    def forward(self, x):
        return self.seg_model(x)

    def freeze_encoder(self):
        encoder = getattr(self.seg_model, "encoder", None)
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        encoder = getattr(self.seg_model, "encoder", None)
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = True
