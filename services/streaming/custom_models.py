from river import base
import numpy as np

class ALMARegressor(base.Regressor):
    """
    Arnaud Legoux Moving Average Regressor.
    
    A regression model based on the ALMA indicator commonly used in technical analysis.
    The ALMA indicator provides a smoother price line than other moving averages while
    maintaining better responsiveness to price changes.
    
    Parameters
    ----------
    alpha: float
        The smoothing factor. Higher values make the moving average more responsive,
        but less smooth. Default is 0.1.
    window_size: int
        The number of observations to consider. Default is 10.
    sigma: float
        Controls the smoothness of the ALMA. Default is 6.
    offset: float
        Controls the responsiveness of the ALMA. Default is 0.85.
    """
    
    def __init__(self, alpha=0.1, window_size=10, sigma=6, offset=0.85):
        self.alpha = alpha
        self.window_size = window_size
        self.sigma = sigma
        self.offset = offset
        self.weights = self._calculate_weights()
        self.buffer = []  # Store last 'window_size' observations
        self._last_prediction = 0
        
    def _calculate_weights(self):
        """Calculate ALMA weights for the given parameters."""
        m = np.floor(self.offset * (self.window_size - 1))
        s = self.window_size / self.sigma
        weights = np.zeros(self.window_size)
        
        for i in range(self.window_size):
            weights[i] = np.exp(-((i - m) ** 2) / (2 * s ** 2))
            
        # Normalize weights
        weights /= np.sum(weights)
        return weights
        
    def learn_one(self, x, y):
        """Update the model with a single observation."""
        # Add new observation to buffer
        self.buffer.append(y)
        
        # Keep only the last window_size observations
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
            
        # Update last prediction if we have enough data
        if len(self.buffer) == self.window_size:
            weighted_sum = sum(w * val for w, val in zip(self.weights, self.buffer))
            self._last_prediction = weighted_sum
            
        return self
        
    def predict_one(self, x):
        """Predict the target value for a single observation."""
        # If we don't have enough data, use simple average
        if len(self.buffer) < self.window_size:
            if not self.buffer:
                return 0
            return sum(self.buffer) / len(self.buffer)
            
        return self._last_prediction
