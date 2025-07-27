import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossMultiClass(nn.Module):

    def __init__(self, alpha, gamma=2.0, reduction='mean'):
        super().__init__()

        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32) if not isinstance(alpha, torch.Tensor) else alpha.float()
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        probs = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = probs[range(inputs.size(0)), targets]
        pt = torch.clamp(pt, min=1e-8)  # prevent log(0) or nan

        focal_term = (1 - pt).pow(self.gamma)

        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
