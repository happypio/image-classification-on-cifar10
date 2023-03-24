from datasets import load_dataset
from transformers import ViTImageProcessor
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)




#load dataset
def load_cifar10():    
    train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']
    return train_ds, val_ds, test_ds




#preprocess
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
normalize = Normalize(mean=image_mean, std=image_std)


def preprocess_train(ds):
    def train_transforms(examples):
        _train_transforms = Compose(
                [
                    RandomResizedCrop(size),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    normalize,
                ]
            )
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples
    ds.set_transform(train_transforms)
    return ds


def preprocess_val_test(val_ds, test_ds):
    def val_transforms(examples):
        _val_transforms = Compose(
                [
                    Resize(size),
                    CenterCrop(size),
                    ToTensor(),
                    normalize,
                ]
            )
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)
    return val_ds, test_ds
