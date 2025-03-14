import torch
import torch.nn as nn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, factor, kernel_size=1, stride=1):
        super(CNNAdapter, self).__init__()
        if input_dim != output_dim:
            logger.info(f"Detected Adapter's input/output of different dimension ({input_dim}/{output_dim}), a projection layer will be used")
            self.projection_layer = nn.Linear(in_features=input_dim, out_features=output_dim)
            input_dim = output_dim
        else:
            self.projection_layer = None

        self.expand_linear = nn.Conv1d(in_channels=input_dim,
                                       out_channels=input_dim * factor,
                                       kernel_size=kernel_size,
                                       stride=stride)
        self.relu = nn.ReLU()
        self.shrink_linear = nn.Conv1d(in_channels=input_dim * factor,
                                       out_channels=output_dim,
                                       kernel_size=kernel_size,
                                       stride=stride)


    def forward(self, x):
        # Shape of x: [T, B, C]
        if self.projection_layer is not None:
            x = self.projection_layer(x)
        x = x.permute(0, 2, 1)
        out = self.expand_linear(x)
        out = self.relu(out)
        out = self.shrink_linear(out)
        out = out.permute(0, 2, 1)

        return out
