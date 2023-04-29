from datasets import Dataset

from transformers import ViTImageProcessor
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

def load_transforms(transform_type: str):
    #preprocess
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)

    if transform_type == 'train':
        transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
    else:
        transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )
    return transforms

def preprocess(ds, ds_type: str):
    _transforms = load_transforms(ds_type)

    def transforms(examples):
        examples['pixel_values'] = [_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    ds.set_transform(transforms)

    return ds
