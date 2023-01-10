import torch
import torch.nn as nn
import torch.nn.functional as F


class Projected_Adaptor(nn.Module):
    def __init__(self, lm_head, num_switches, embed_dim, vocab_size, rank,
                 epsilon, init_var, position="output"):
        super().__init__()
        assert rank > 0
        self.projector1 = nn.Parameter(torch.randn(
            num_switches, embed_dim, rank
        ) * init_var)
        self.projector2 = nn.Parameter(torch.randn(
            num_switches, embed_dim, rank
        ) * init_var)
        self.rank = rank
        self.lm_head = lm_head
        self.epsilon = epsilon
        self.position = position
        self.num_switches = num_switches
        self.init_var = init_var
        self.switch_values = torch.zeros(num_switches)

    def set_value(self, switch_values):
        self.switch_values = switch_values

    def forward(self, state):
        ori_weight = self.lm_head.weight.detach()
        projector = self.projector1.matmul(self.projector2.transpose(1, 2))
        projector = (projector[None] * self.switch_values[:, :, None, None]
                     ).sum(1)
        weight = ori_weight[None] + self.epsilon * ori_weight.matmul(
            projector)
        if self.position == "output":
            return state.matmul(weight.transpose(1, 2))
            # ori_output = state.matmul(ori_weight.transpose(0, 1))
            # bs, length, dim = state.shape
            # state = state.view(-1, dim)
            # added = self.projector2.transpose(1, 2).matmul(
            #     state.transpose(0, 1)).transpose(1, 2)
            # added = added.matmul(self.projector1.transpose(1, 2))
            # added = self.epsilon * (
            #     added.view(-1, bs, length, dim) *
            #     self.switch_values.transpose(0, 1)[:, :, None, None]
            # ).sum(0)
            # added = added.matmul(ori_weight.transpose(0, 1))
            # return ori_output + added
        elif self.position == "input":
            raise NotImplementedError()
            # return torch.stack([
            #     F.embedding(_one_state, _one_weight)
            #     for _one_state, _one_weight in zip(state, weight)
            # ])
        else:
            raise Exception

    def regularization_term(self):
        if self.rank <= 0:
            return self.projector.pow(2).sum()
        else:
            return self.projector1.pow(2).sum() + self.projector2.pow(2).sum()

    def parameters(self):
        if self.rank <= 0:
            return [self.projector]
        else:
            return [self.projector1, self.projector2]

    def state_dict(self):
        if self.rank <= 0:
            return {"projector": self.projector}
        else:
            return {"projector1": self.projector1,
                    "projector2": self.projector2}

    def load_state_dict(self, state_dict):
        if self.rank <= 0:
            self.projector.data = state_dict["projector"]
        else:
            self.projector1.data = state_dict["projector1"]
            self.projector2.data = state_dict["projector2"]
