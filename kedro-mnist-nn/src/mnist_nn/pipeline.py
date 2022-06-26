"""
This is a boilerplate pipeline
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["mnist-data-train", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split",
            ),
            node(
                func=get_data_shape,
                inputs=["X_train"],
                outputs=["c_train", "r_train"],
                name="data_shape",
            ),
            node(
                func=train_wcost,
                inputs=["X_train", "y_train", "X_test", "y_test","parameters","c_train","r_train"],
                outputs= "final-model-pkl",
                name="nn_training_model",
            ),
            node(
                func=plot_cost_function,
                inputs= "final-model-pkl",
                outputs= "plot-cost-function",
                name="plot_cost_function",
            ),
        ]
    )