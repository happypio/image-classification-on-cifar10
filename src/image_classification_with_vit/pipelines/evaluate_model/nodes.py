from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def evaluate_model(test_ds):

    model = ViTForImageClassification.from_pretrained("./data/06_model/model")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    args = TrainingArguments(f"test-cifar-10", remove_unused_columns=False)
    
    trainer = Trainer(
            model,
            args,
            data_collator=collate_fn,
            tokenizer=processor
        )
    
    outputs = trainer.predict(test_ds)
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    labels = test_ds.features['label'].names
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.title(f"Model accuracy is: {(sum(y_true == y_pred) / len(y_true) * 100):.2f}%")
    plt.savefig("./data/08_reporting/confusion_matrix.png")
