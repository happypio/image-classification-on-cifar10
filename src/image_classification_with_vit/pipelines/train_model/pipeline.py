"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model
from .processing_nodes import preprocess

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=preprocess,
                 inputs=["train_ds", "params:train_transform"],
                 outputs="preprocessed_train_ds",
                 name="preprocess_train_node"),

            node(func=preprocess,
                 inputs=["val_ds", "params:val_transform"],
                 outputs='preprocessed_val_ds',
                 name="preprocess_val_node"),

            node(func=train_model,
                 inputs=[
                     "preprocessed_train_ds", 
                     "preprocessed_val_ds", 
                     "train_ds",
                     "params:learning_rate",
                     "params:num_train_epochs",
                     "params:weight_decay",
                     "params:device"
                     ],
                 outputs=None,
                 name="train_model_node"),
        ]
    )
