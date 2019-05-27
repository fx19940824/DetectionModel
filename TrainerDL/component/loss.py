import torch

class soft_cross_entropy(torch.nn.Module):
    def __init__(self):
        super(soft_cross_entropy, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, pred, soft_targets):
        return torch.mean(torch.sum(- soft_targets * self.logsoftmax(pred), 1))

