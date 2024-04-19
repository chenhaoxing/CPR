import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from utils import cls_acc, delete_tensor, my_acc, calculate_time
from model.base import BaseModel


class TipAdapter(BaseModel):

    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)

    def logits(self, features, beta, alpha):
        clip_logits = 100. * features @ self.clip_weights
        affinity = features @ self.cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
        tip_logits = clip_logits + cache_logits * alpha
        return tip_logits

    def evaluate(self, test_features, test_labels):
        best_beta, best_alpha = self.init_get_best_param()
        tip_logits = self.logits(test_features, best_beta, best_alpha)
        acc = cls_acc(tip_logits, test_labels)
        print('Tip Adapter test acc = {:.2f}'.format(acc))

