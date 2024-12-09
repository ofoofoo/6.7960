# should prolly make this more generalizable, or maybe not if we just stick with this guy for this project
from transformers import ViTForImageClassification, ViTConfig
import hydra

def create_model(num_encoder_layers_frozen = 0, num_classes=101, pretrained=True):
    if pretrained:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            label2id={i:i for i in range(num_classes)},
            id2label={i:i for i in range(num_classes)},
            ignore_mismatched_sizes = True)
    else:
        config = ViTConfig(
            num_labels=num_classes,
            image_size=224,
            patch_size=16,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
        model = ViTForImageClassification(config)

    for i in range(num_encoder_layers_frozen):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = False
    return model
