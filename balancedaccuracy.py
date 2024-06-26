import torch

# NOTE: This will be the calculation of balanced accuracy for your classification task
# The balanced accuracy is defined as the average accuracy for each class.
# The accuracy for an indiviual class is the ratio between correctly classified example to all examples of that class.
# The code in train.py will instantiate one instance of this class.
# It will call the reset methos at the beginning of each epoch. Use this to reset your
# internal states. The update method will be called multiple times during an epoch, once for each batch of the training.
# You will receive the network predictions, a Tensor of Size (BATCHSIZExCLASSES) containing the logits (output without Softmax).
# You will also receive the groundtruth, an integer (long) Tensor with the respective class index per example.
# For each class, count how many examples were correctly classified and how many total examples exist.
# Then, in the getBACC method, calculate the balanced accuracy by first calculating each individual accuracy
# and then taking the average.

# Balanced Accuracy
class BalancedAccuracy:
    def __init__(self, nClasses):
        # TODO: Setup internal variables
        # NOTE: It is good practive to all reset() from here to make sure everything is properly initialized
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        # TODO: Reset internal states.
        # Called at the beginning of each epoch
        self.correct_per_class = torch.zeros(self.nClasses)
        self.total_per_class = torch.zeros(self.nClasses)

    def update(self, predictions, groundtruth):
        # TODO: Implement the update of internal states
        # based on current network predictios and the groundtruth value.
        #
        # Predictions is a Tensor with logits (non-normalized activations)
        # It is a BATCH_SIZE x N_CLASSES float Tensor. The argmax for each samples
        # indicated the predicted class.
        #
        # Groundtruth is a BATCH_SIZE x 1 long Tensor. It contains the index of the
        # ground truth class.
        # Convert logits to predicted classes
        _, predicted_classes = torch.max(predictions, 1)
        # Update counts for correctly classified and total per class
        for i in range(self.nClasses):
            self.correct_per_class[i] += (
                predicted_classes[groundtruth == i] == groundtruth[groundtruth == i]
            ).sum()
            self.total_per_class[i] += (groundtruth == i).sum()

    def getBACC(self):
        # TODO: Calculcate and return balanced accuracy
        # based on current internal state
        # Calculate individual class accuracies
        class_accuracies = self.correct_per_class / self.total_per_class
        # Calculate balanced accuracy
        balanced_accuracy = class_accuracies.mean()
        return balanced_accuracy.item()  # Convert to Python scalar
