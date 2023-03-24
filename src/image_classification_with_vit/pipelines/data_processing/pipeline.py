from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_cifar10, preprocess_train, preprocess_val_test




def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=load_cifar10,
                 inputs=None,
                 outputs=["train_ds", "val_ds", "test_ds"],
                 name="load_cifar10_node"),
            
            node(func=preprocess_train,
                 inputs="train_ds",
                 outputs="preprocessed_train_ds",
                 name="preprocess_train_node"),
            
            node(func=preprocess_val_test,
                 inputs=["val_ds", "test_ds"],
                 outputs=["preprocessed_val_ds", "preprocessed_test_ds"],
                 name="preprocess_val_test_node")
        ]
    )



