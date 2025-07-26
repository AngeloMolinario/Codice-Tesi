import torch
from torch import nn


class PromptLearner(nn.Module):
    def __init__(self, num_prompt: int, emb_size: int):
        super().__init__()
        self.num_prompt = num_prompt
        self.emb_size = emb_size
        self.prompt_tokens = nn.Parameter(torch.randn(num_prompt, emb_size))

    def forward(self, x, is_cls_present: bool = False):
        batch_size = x.shape[0]

        cntx = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1) # [B, num_prompt, D]

        if not is_cls_present:
            x = torch.cat([cntx, x], dim=1)
        else:
            x = torch.cat([
                x[:, :1, :],
                cntx,
                x[:, 1:, :]
            ], dim=1)

        return x