import torch

class Linear(torch.nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        self.linear1 = torch.nn.Linear(1024, 256)
        self.relu1 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(256, 2)

    def forward(self, input):
        x = self.linear1(input)
        x = self.relu1(x)
        x = self.linear2(x)
        return x