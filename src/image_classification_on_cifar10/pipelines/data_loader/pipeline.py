"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_data,
                inputs=["params:train", "params:train_transform"],
                outputs="cifar_10_train_data",
                name="download_train_data",
            ),
            node(
                func=download_data,
                inputs=["params:test", "params:test_transform"],
                outputs="cifar_10_test_data",
                name="download_test_data",
            ),
        ]
    )
