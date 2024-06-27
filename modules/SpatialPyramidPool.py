import torch
import torch.nn as nn
import torch.optim as optim

class Spatial_Pyramid_Pool(nn.Module):
    def __init__(self, out_pool_size, pooling_type=nn.MaxPool2d):
        '''
        output_size: the height and width after spp layer
        pooling_type: the type of pooling
        '''
        super(Spatial_Pyramid_Pool, self).__init__()
        self.out_pool_size = out_pool_size
        self.pooling_type = pooling_type

    def forward(self, previous_conv):
        N, C, H, W = previous_conv.size()

        for i in range(len(self.out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(H / self.out_pool_size[i]))
            w_wid = int(math.ceil(W / self.out_pool_size[i]))
            h_pad = (h_wid * self.out_pool_size[i] - H + 1) // 2
            w_pad = (w_wid * self.out_pool_size[i] - W + 1) // 2
            maxpool = self.pooling_type(kernel_size=(h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if (i == 0):
                spp = x.view(N, -1)
                # print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp, x.view(N, -1)), 1)
        return spp


# test using the Stanford Assignment http://cs231n.github.io/assignments2017/assignment2/ Q5
test0 = nn.Sequential(BasicResBlock(3, 128),
                      nn.MaxPool2d(kernel_size=2, stride=2),

                      BasicResBlock(128, 128),
                      BasicResBlock(128, 128),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.Dropout2d(p=0.5),

                      Spatial_Pyramid_Pool([4, 3, 2, 1]),

                      nn.Linear(3840, 10)
                      )

# Set the type of all data in this model to be FloatTensor
test0.type(gpu_dtype)

loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
optimizer = optim.Adam(test0.parameters(), lr=1e-2)  # lr sets the learning rate of the optimizer
train_plot(test0, loader_train, loader_val, loss_fn, optimizer, num_epochs=20)