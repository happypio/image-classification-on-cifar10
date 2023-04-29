"""
This is a boilerplate pipeline 'download_data'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_cifar10

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=load_cifar10,
                 inputs=["params:train_size", "params:test_size", "params:val_prct"],
                 outputs=["train_ds", "val_ds", "test_ds"],
                 name="load_cifar10_node"),
        ])
