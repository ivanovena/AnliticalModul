import numpy as np
from river import base
from river.utils import rolling

class ALMARegressor(base.Regressor):
    """Arnaud Legoux Moving Average Regressor.
    
    The ALMA is a moving average algorithm that aims to reduce lag and noise.
    It's a weighted moving average where the weights are a Gaussian distribution,
    whose center and width can be adjusted.
    
    Parameters:
    -----------
    alpha : float (default=0.1)
        Learning rate for updating weights
    window_size : int (default=10)
        Size of the rolling window for ALMA calculation
    sigma : float (default=6.0)
        Controls the width of the distribution of weights
    offset : float (default=0.85)
        Controls the position of the distribution of weights (0 to 1)
    """
    
    def __init__(self, alpha=0.1, window_size=10, sigma=6.0, offset=0.85):
        self.alpha = alpha
        self.window_size = window_size
        self.sigma = sigma
        self.offset = offset
        self.weights = np.zeros(window_size)
        self._y_history = []  # Changed from rolling.Window
        self._max_window_size = window_size
        self._init_weights()
    
    def _init_weights(self):
        """Initialize ALMA weights using Gaussian distribution"""
        m = np.floor(self.offset * (self.window_size - 1))
        s = self.window_size / self.sigma
        
        denom = 0
        
        # Create weights using Gaussian distribution
        for i in range(self.window_size):
            w = np.exp(-((i - m) ** 2) / (2 * s ** 2))
            self.weights[i] = w
            denom += w
        
        # Normalize weights
        if denom != 0:
            self.weights /= denom
    
    def predict_one(self, x):
        """Predict the next value"""
        if len(self._y_history) < self.window_size:
            # Default prediction is the feature value if not enough history
            return next(iter(x.values())) if isinstance(x, dict) else x
        
        # Use the ALMA weights to predict the next value
        y_hist = np.array(self._y_history)
        return np.sum(y_hist * self.weights)
    
    def learn_one(self, x, y):
        """Update the model with a single learning example"""
        self._y_history.append(y)
        
        # Maintain the window size by removing old elements
        if len(self._y_history) > self._max_window_size:
            self._y_history.pop(0)
        
        # Only update the weights if we have enough history
        if len(self._y_history) >= self.window_size:
            # Calculate current prediction (used for error calculation)
            y_hist = np.array(self._y_history)
            y_pred = np.sum(y_hist * self.weights)
            
            # Calculate error
            error = y - y_pred
            
            # Update weights using gradient descent
            # We add a small factor to prioritize more recent data
            recency_factor = np.linspace(0.8, 1.0, self.window_size)
            
            # Update each weight individually
            for i in range(self.window_size):
                gradient = -2 * error * y_hist[i] * recency_factor[i]
                self.weights[i] -= self.alpha * gradient
            
            # Re-normalize weights
            self.weights /= np.sum(self.weights)
        
        return self
