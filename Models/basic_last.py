import torch
from torch import nn
from torch.nn import Module, Parameter
import math

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

'''
This module only use last item to predict the next
'''
class BasicLast(Module):
    def __init__(self, opt, n_node):
        super(BasicLast, self).__init__()
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.n_node = n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, masks, targets):
        last_id = [inputs[i][j-1] for i,j in zip(range(len(inputs)), torch.sum(masks, 1))]
        last_id = torch.Tensor(last_id).long().cuda()
        inputs = self.embedding(inputs)
        last_inputs = self.embedding(last_id)
        b = self.embedding.weight[1:]
        scores = torch.matmul(last_inputs, b.transpose(1, 0))
        return scores
