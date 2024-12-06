# should prolly make this more generalizable, or maybe not if we just stick with this guy for this project
from transformers import ViTForImageClassification

def create_model():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    return model
