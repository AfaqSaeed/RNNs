import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        # epsilon represents the smallest representable number
        # prevents values close to log(0) because log(0) -> -inf (undefined)
        self.epsilon = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        # store predictions y^ for backward pass
        self.prediction_tensor = prediction_tensor
        # for all labels = 1 is implemented as elementwise multiplication of label tensor with -ln(y^ + eps)
        loss_value = np.sum((-(label_tensor) * np.log((prediction_tensor + self.epsilon))))
        return loss_value

    def backward(self, label_tensor):
        # error depends only on label and prediction (+ eps to prevent dividing by zero)
        return - label_tensor / (self.prediction_tensor + self.epsilon)