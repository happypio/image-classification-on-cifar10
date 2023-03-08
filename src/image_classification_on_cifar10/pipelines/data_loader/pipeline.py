"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_train_data, download_test_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=download_train_data,
            inputs="params:train",
            outputs="cifar_10_train_data",
            name='download_train_data'
        ),
        node(
            func=download_test_data,
            inputs="params:test",
            outputs="cifar_10_test_data",
            name='download_test_data'
        )
    ])
