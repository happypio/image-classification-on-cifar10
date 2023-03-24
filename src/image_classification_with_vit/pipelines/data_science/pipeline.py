from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model, train_model




def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=train_model,
                 inputs=["preprocessed_train_ds", "preprocessed_val_ds"],
                 outputs="model",
                 name="train_model_node"),
            
            node(func=evaluate_model,
                 inputs=["preprocessed_test_ds", "model"],
                 outputs="confusion_matrix",
                 name="evaluate_model_node")
        ]
    )
