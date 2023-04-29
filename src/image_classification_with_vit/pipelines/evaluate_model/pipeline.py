"""
This is a boilerplate pipeline 'evaluate_model'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model
from ..train_model.processing_nodes import preprocess

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=preprocess,
                 inputs=["test_ds", "params:test_transform"],
                 outputs="preprocessed_test_ds",
                 name="preprocess_test_node"),

            node(func=evaluate_model,
                 inputs=["preprocessed_test_ds"],
                 outputs=None,
                 name="evaluate_model_node")
        ]
    )
