import logging
from re import A
from typing import Any, Dict, Tuple
from attr import s

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

from .neural_network import *

def split_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    data_train = data.sample( frac=parameters["train_fraction"], random_state=parameters["random_state"] )
    data_test = data.drop(data_train.index)
    
    X_train = data_train.drop(columns=parameters["target_column"])
    X_train = np.array(X_train.T)
    X_train = X_train / 255.
    
    X_test = data_test.drop(columns=parameters["target_column"])
    X_test = np.array(X_test.T)
    X_test = X_test / 255.
    
    y_train = data_train[parameters["target_column"]]
    y_test = data_test[parameters["target_column"]]

    return X_train, X_test, y_train, y_test

def get_data_shape(data):
    
    data = np.array(data)
    c, r = data.shape
    
    return c, r

def plot_cost_function(nn_model):
    
    cost_train = nn_model['cost_function']['cost_train']
    cost_test = nn_model['cost_function']['cost_test']
    
    plt.plot(cost_train)
    plt.plot(cost_test)
    
    return plt

# Treino de modelo com Cost Function (1 - acuracia) do modelo e dos dados de teste
def train_wcost(X, Y, X_test, Y_test, parameters, c_train, r_train):
    
    cost_train = []
    cost_test = []
    epoch = []
    
    model = {}
    cost_function = {}
    
    W1, b1, W2, b2 = input_layer(parameters["nodes"] , X, c_train) # Cria parametros de Weight e Bias
    
    for i in range(parameters["iterations"]):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, r_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, parameters["alpha"])
        
        predictions = get_predictions(A2)
        accuracy = get_accuracy(predictions, Y)
        
        _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
        predictions_test = get_predictions(A2_test)
        accuracy_test = get_accuracy(predictions_test, Y_test)
        
        cost_train.append(1 - accuracy)
        cost_test.append(1 - accuracy_test)
        epoch.append(i)
        
        if i % 10 == 0: # Para cada 10 iterations, print
            print("Iteration: ", i)
            print("Cost Function Train: ", 1 - accuracy)
            print("Cost Function Test: ", 1 - accuracy_test)
            
    
    model["W1"] = W1
    model["b1"] = b1    
    model["W2"] = W2
    model["b2"] = b2
        
    cost_function["cost_train"] = cost_train
    cost_function["cost_test"] = cost_test
    
    nn_model = {"model": model, "cost_function": cost_function}

    return nn_model