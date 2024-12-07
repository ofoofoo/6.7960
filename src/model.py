# should prolly make this more generalizable, or maybe not if we just stick with this guy for this project
from transformers import ViTForImageClassification

def create_model(num_encoder_layers_unfrozen = 0):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        label2id={i:i for i in range(101)},
        id2label={i:i for i in range(101)},
        ignore_mismatched_sizes = True)

    for param in model.vit.parameters():
        param.requires_grad = False
    for i in range(num_encoder_layers_unfrozen):
        for param in model.vit.encoder.layer[-i-1].parameters():
            param.requires_grad = True
    return model
