import torch.nn as nn


class NeuronioLoss(nn.Module):
    def __init__(self):
        super(NeuronioLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, model_output, target):
        # Extract the elements for the respective losses
        bce_input = model_output[..., 0]
        bce_target = target[0]
        mse_input = model_output[..., 1]
        mse_target = target[1]

        # Compute the losses
        bce_loss = self.bce_loss(bce_input, bce_target)
        mse_loss = self.mse_loss(mse_input, mse_target)

        # Balance the losses with a factor of 0.5 each
        loss = 0.5 * bce_loss + 0.5 * mse_loss

        return loss
