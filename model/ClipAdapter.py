import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from utils import cls_acc, delete_tensor, my_acc, calculate_time
from model.base import BaseModel
import torch.nn as nn


class CLipAdapter(BaseModel):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)
        self.text_adapter = nn.Sequential(
            nn.Linear(1024, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024, bias=False),
            nn.ReLU(inplace=True)
        ).to(clip_model.dtype).cuda()
        self.visual_adapter = nn.Sequential(
            nn.Linear(1024, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024, bias=False),
            nn.ReLU(inplace=True)
        ).to(clip_model.dtype).cuda()
        self.weight_save_path = self.cfg['cache_dir'] + "/best_ClipAdapter_" + str(self.cfg['shots']) + "shots.pt"

    def logits(self, features, beta=None, alpha=None):
        image_features = self.adapt(features)
        text_features = self.clip_weights.T
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100. * image_features @ text_features.t()
        return logits

    def evaluate(self, test_features, test_labels):
        self.visual_adapter = torch.load(self.weight_save_path)
        logits = self.logits(test_features)
        acc = cls_acc(logits, test_labels)
        print('CoOp test acc = {:.2f}'.format(acc))

    def train(self, test_features, test_labels, train_loader):
        optimizer = torch.optim.AdamW(self.visual_adapter.parameters(), lr=self.cfg['lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(train_loader))
        best_acc, best_epoch = 0.0, 0
        for train_idx in range(self.cfg['train_epoch']):
            self.visual_adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                clip_adapter_logits = self.logits(image_features)
                loss = F.cross_entropy(clip_adapter_logits, target)
                acc = cls_acc(clip_adapter_logits, target)
                correct_samples += acc / 100 * len(clip_adapter_logits)
                all_samples += len(clip_adapter_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                           correct_samples, all_samples,
                                                                           sum(loss_list) / len(loss_list)))
            self.visual_adapter.eval()
            clip_adapter_logits = self.logits(test_features)
            acc = cls_acc(clip_adapter_logits, test_labels)

            print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(self.visual_adapter,
                           self.weight_save_path)
        print('best train acc = {:.2f}'.format(best_acc))

    def adapt(self, features):
        x = self.visual_adapter(features)
        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features
