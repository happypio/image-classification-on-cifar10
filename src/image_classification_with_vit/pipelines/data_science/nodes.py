import pandas as pd
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sn




def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))





def train_model(train_ds, val_ds):
    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=10,
                                                  id2label=id2label,
                                                  label2id=label2id)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    metric_name = "accuracy"
    
    args = TrainingArguments(
        f"test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir='logs',
        remove_unused_columns=False)
    
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor)
    
    trainer.train()
    
    return trainer







def evaluate_model(test_ds, model):
    outputs = model.predict(test_ds)
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    confusion_matrix = pd.crosstab(
        y_true, y_pred, rownames=["Actual"], colnames=["Predicted"]
    )
    sn.heatmap(confusion_matrix, annot=True)
    return plt
