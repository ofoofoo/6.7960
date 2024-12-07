# should prolly make this more generalizable, or maybe not if we just stick with this guy for this project
from transformers import ViTForImageClassification
import hydra

def create_model(num_encoder_layers_frozen = 0, num_classes=101):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        label2id={i:i for i in range(num_classes)},
        id2label={i:i for i in range(num_classes)},
        ignore_mismatched_sizes = True)

    for i in range(num_encoder_layers_frozen):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = False
    return model
