import numpy as np
from sklearn.metrics import mean_absolute_error

def weighted_average_ensemble(predictions, weights):
    """
    Perform weighted average ensemble of predictions.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary of model predictions.
    weights : dict
        Dictionary of model weights.
    
    Returns:
    --------
    np.array
        Weighted average prediction.
    """
    return sum(weights[model] * pred for model, pred in predictions.items())

def calculate_model_weights(models, X_val, y_val):
    """
    Calculate weights for each model based on validation performance.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models.
    X_val : np.array
        Validation features.
    y_val : np.array
        Validation target.
    
    Returns:
    --------
    dict
        Dictionary of model weights.
    """
    errors = {}
    for name, model in models.items():
        pred = model.predict(X_val)
        errors[name] = mean_absolute_error(y_val, pred)
    
    total_error = sum(errors.values())
    weights = {name: (total_error - error) / ((len(models) - 1) * total_error) for name, error in errors.items()}
    return weights

if __name__ == "__main__":
    # This is a placeholder for testing the ensemble functions
    predictions = {
        'model1': np.array([1, 2, 3]),
        'model2': np.array([2, 3, 4]),
        'model3': np.array([3, 4, 5])
    }
    weights = {'model1': 0.5, 'model2': 0.3, 'model3': 0.2}
    
    result = weighted_average_ensemble(predictions, weights)
    print("Ensemble prediction:", result)
