import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from utils import cls_acc, delete_tensor, my_acc, calculate_time
from model.base import BaseModel
import torch.nn as nn


class TipAdapterF(BaseModel):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)
        self.adapter = nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(
            clip_model.dtype).cuda()
        self.adapter.weight = nn.Parameter(self.cache_keys.t())
        self.weight_save_path = cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt"

    def logits(self, features, beta, alpha):
        clip_logits = 100. * features @ self.clip_weights
        affinity = self.adapter(features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
        tip_logits = clip_logits + cache_logits * alpha
        return tip_logits

    def evaluate(self, test_features, test_labels):
        self.adapter.weight = torch.load(self.weight_save_path)
        self.cache_keys = self.adapter.weight.T
        #beta, alpha = self.init_get_best_param()
        beta, alpha = self.cfg['init_beta'], self.cfg['init_alpha']
        tip_logits = self.logits(test_features, beta, alpha)
        acc = cls_acc(tip_logits, test_labels)
        print('Tip Adapter F test acc = {:.2f}'.format(acc))

    def train(self, test_features, test_labels, train_loader):
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.cfg['lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(train_loader))

        beta, alpha = self.cfg['init_beta'], self.cfg['init_alpha']
        best_acc = 0.0

        for train_idx in range(self.cfg['train_epoch']):
            # Train
            self.adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                tip_logits = self.logits(image_features, beta, alpha)

                loss = F.cross_entropy(tip_logits, target)

                acc = cls_acc(tip_logits, target)
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                           correct_samples, all_samples,
                                                                           sum(loss_list) / len(loss_list)))

            # Eval
            self.adapter.eval()

            affinity = self.adapter(test_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
            clip_logits = 100. * test_features @ self.clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, test_labels)

            print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(self.adapter.weight, self.weight_save_path)
        print('best train acc = {:.2f}'.format(best_acc))
