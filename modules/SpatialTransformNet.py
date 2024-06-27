import torch
import torch.nn as nn
import torch.optim as optim

class SpatialTransformNetwork(nn.Module):
    def __init__(self, in_channels):
        super(SpatialTransformNetwork, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(10, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta.expand(x.shape[0], 2, 3), x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        return self.stn(x)


#test
model_res_from_start_avg_sp = nn.Sequential(
    SpatialTransformNetwork(3),

    BasicResBlock(3, 64),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BasicResBlock(64, 10),
    nn.Dropout(0.3),

    nn.AdaptiveAvgPool2d(1),
    Flatten(),
)

# Set the type of all data in this model to be FloatTensor
model_res_from_start_avg_sp.type(gpu_dtype)

loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
optimizer = optim.Adam(model_res_from_start_avg_sp.parameters(), lr=1e-2)  # lr sets the learning rate of the optimizer
train_plot(model_res_from_start_avg_sp, loader_train, loader_val, loss_fn, optimizer, num_epochs=60)