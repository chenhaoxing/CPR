import torch.nn.functional as F
from utils import cls_acc, my_acc, delete_tensor
import torch.nn as nn
import torch
from tqdm import tqdm
from model.base import BaseModel, TextEncoder, PromptLearner,CustomCLIP
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from clip import clip
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont, ImageFile
import torchvision.transforms as transforms
import json


class AutoPromptModel(nn.Module):
    def __init__(self, cfg, nums, visual_proto_emb, classnames, clip_model):
        super(AutoPromptModel, self).__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, "a photo of a")
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

        for name, param in self.prompt_learner.named_parameters():
            #if "prompt_learner" not in name: # and "adapter" not in name:
            if "ctx" not in name: 
                param.requires_grad_(False)
            else:
                print(name)
        
        for name, param in self.text_encoder.named_parameters():
            #if "prompt_learner" not in name: # and "adapter" not in name:
            if "ctx" not in name: 
                param.requires_grad_(False)
            else:
                print(name)
        
        self.text_prompt_channel_num = nums
        self.prototypes = visual_proto_emb

        self.query = nn.Linear(self.text_prompt_channel_num, self.text_prompt_channel_num)
        self.key = nn.Linear(self.text_prompt_channel_num, self.text_prompt_channel_num)
        self.value = nn.Linear(self.text_prompt_channel_num, self.text_prompt_channel_num)
        self.proj = nn.Linear(self.text_prompt_channel_num, self.text_prompt_channel_num)
        self.out = nn.Linear(self.text_prompt_channel_num, self.text_prompt_channel_num)

        nn.init.normal_(self.query.weight, mean=0, std=np.sqrt(2.0 / (self.text_prompt_channel_num)))
        nn.init.normal_(self.key.weight, mean=0, std=np.sqrt(2.0 / (self.text_prompt_channel_num)))
        nn.init.normal_(self.value.weight, mean=0, std=np.sqrt(2.0 / (self.text_prompt_channel_num)))
        nn.init.xavier_normal_(self.out.weight)
        self.norm1 = nn.LayerNorm(self.text_prompt_channel_num)
        self.dropout = nn.Dropout(0.1)
        self.query.half()
        self.key.half()
        self.out.half()
        self.value.half()
        self.norm1.half()
        self.proj.half()
        
    def infer(self, x):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        q, k, v = self.query(x), self.key(self.prototypes), self.value(self.prototypes)
        scores = torch.matmul(q, k.transpose(-2, -1)) / ((self.text_prompt_channel_num)**0.5)
        attn = F.softmax(scores, dim=-1)
        v = torch.matmul(attn, v)
        att_weights = self.out(v) + self.proj(x)
        visual_att_weights = self.norm1(att_weights)

        q, k, v = self.query(x), self.key(text_features), self.value(text_features)
        scores = torch.matmul(q, k.transpose(-2, -1)) / ((self.text_prompt_channel_num)**0.5)
        attn = F.softmax(scores, dim=-1)
        v = torch.matmul(attn, v)
        att_weights = self.out(v) + self.proj(x)
        text_att_weights = self.norm1(att_weights)
        weights = visual_att_weights + text_att_weights
        residual_weights = weights.unsqueeze(1)
        text_prompt = text_features.unsqueeze(0)

        all_prompts = residual_weights + text_prompt
        all_prompts = all_prompts / all_prompts.norm(dim=-1, keepdim=True)

        return all_prompts

    def forward(self, x):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        

        q, k, v = self.query(x), self.key(self.prototypes), self.value(self.prototypes)
        scores = torch.matmul(q, k.transpose(-2, -1)) / ((self.text_prompt_channel_num)**0.5)
        attn = F.softmax(scores, dim=-1)
        v = torch.matmul(attn, v)
        att_weights = self.out(v) + self.proj(x)
        visual_att_weights = self.norm1(att_weights)

        q, k, v = self.query(x), self.key(text_features), self.value(text_features)
        scores = torch.matmul(q, k.transpose(-2, -1)) / ((self.text_prompt_channel_num)**0.5)
        attn = F.softmax(scores, dim=-1)
        v = torch.matmul(attn, v)
        att_weights = self.out(v) + self.proj(x)
        text_att_weights = self.norm1(att_weights)

        weights = visual_att_weights + text_att_weights

        residual_weights = weights.unsqueeze(1)
        text_prompt = text_features.unsqueeze(0)

        # # b, cls_num, channel_num
        all_prompts = residual_weights + text_prompt
        # all_prompts = self.norm1(all_prompts)
        all_prompts = all_prompts / all_prompts.norm(dim=-1, keepdim=True)
        
        # # b, 1, channel_num
        x = x.unsqueeze(1)
        # # b, channel_num, 1
        x = x.permute(0, 2, 1)

        logits = all_prompts @ x
        logits = logits.squeeze(-1)

        return 100. * logits, all_prompts

class CPR(BaseModel):
    def __init__(self, cfg, classnames, templete, cupl_path, clip_model):
        super().__init__(cfg, classnames,templete, cupl_path, clip_model)
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.train_tranform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])
        self.template = templete
        self.cupl_path = cupl_path
        self.clip_weights, self.cupl_weights = self.get_text_prompt_embeddings(classnames, clip_model)
        self.clip_weights, self.cupl_weights = self.clip_weights.detach(), self.cupl_weights.detach()
        self.cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-07)
        self.auto_prompt = AutoPromptModel(cfg, self.clip_weights.shape[1], self.visual_prototypes, classnames, clip_model).cuda()
        
    def get_text_prompt_embeddings(self, classnames, clip_model):
        f = open(self.cupl_path)
        prompts = json.load(f)

        with torch.no_grad():
            original_clip_weights = []
            cupl_clip_weights = []

            for classname in classnames:
                classname = classname.replace('_', ' ')

                template_texts = [t.format(classname) for t in self.template]
                cupl_texts = prompts[classname]
                cupl_texts = template_texts + cupl_texts
                cupl_texts_token = clip.tokenize(cupl_texts, truncate=True).cuda()
                
                texts = 'a photo of a' + classname  + '.'
                texts = clip.tokenize(texts).cuda()
                with torch.no_grad():
                    cupl_class_embeddings = clip_model.encode_text(cupl_texts_token)
                    cupl_class_embeddings /= cupl_class_embeddings.norm(dim=-1, keepdim=True)
                    cupl_class_embedding = cupl_class_embeddings.mean(dim=0)
                    cupl_class_embedding /= cupl_class_embedding.norm()

                    # prompt ensemble for ImageNet
                    class_embeddings = clip_model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

                # glyph_imgs_weights.append(glyph_img_embedding)
                original_clip_weights.append(class_embeddings)
                cupl_clip_weights.append(cupl_class_embedding)

            clip_weights = torch.stack(original_clip_weights, dim=1).cuda()
            clip_weights = clip_weights.squeeze(0)

            cupl_weights = torch.stack(cupl_clip_weights, dim=1).cuda()
            cupl_weights = cupl_weights.permute(1, 0)

        return clip_weights, cupl_weights


    def logits(self, features, beta=None, alpha=None):
        text_scores = 100. * self.text_prompt(features)
        return text_scores

    def evaluate(self, test_features, test_loader, update=False):
        
        self.new_auto_prompt = AutoPromptModel(self.cfg, self.clip_weights.shape[1], self.visual_prototypes, self.classnames, self.clip_model).cuda()
        state_dict = torch.load(self.cfg['cache_dir'] + "/best_CoOp_text_prompt_2" + str(self.cfg['shots']) + "shots.pt")
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]

        if "prompt_learner.token_midfix" in state_dict:
            del state_dict["prompt_learner.token_midfix"]
        self.new_auto_prompt.load_state_dict(state_dict, strict=False)
        self.new_auto_prompt.eval()
        
        if update:
            print("********* Using Prototype Rectification **********")
            correct_samples, all_samples = 0, 0
            for i, (images, target) in enumerate(tqdm(test_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                            
                all_prompts = self.new_auto_prompt.infer(image_features)
                similarity_matrix = all_prompts @ test_features.unsqueeze(0).permute(0, 2, 1)

                alpha, k = 0.95, 5

                for j, batch_similarity_matrix in enumerate(similarity_matrix):
                    values, indices = torch.topk(batch_similarity_matrix, k, dim=1)
                    selected_values = test_features[indices]
                    class_prototype = selected_values.mean(dim=1)
                    all_prompts[j] = alpha * all_prompts[j] + (1-alpha) * class_prototype

                image_features = image_features.unsqueeze(1)
                image_features = image_features.permute(0, 2, 1)
                logits = all_prompts @ image_features
                logits = logits.squeeze(-1)
                CoOp_logits = 100. * logits
                acc = cls_acc(CoOp_logits, target)
                correct_samples += acc / 100 * len(CoOp_logits)
                all_samples += len(CoOp_logits)

        else:
            print("********* Using Original Prototype **********")
            correct_samples, all_samples = 0, 0
            for i, (images, target) in enumerate(tqdm(test_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                            
                all_prompts = self.new_auto_prompt.infer(image_features)
                image_features = image_features.unsqueeze(1)
                image_features = image_features.permute(0, 2, 1)
                logits = all_prompts @ image_features
                logits = logits.squeeze(-1)
                CoOp_logits = 100. * logits
                acc = cls_acc(CoOp_logits, target)
                correct_samples += acc / 100 * len(CoOp_logits)
                all_samples += len(CoOp_logits)

        acc = 100 *correct_samples/all_samples

        print('CoOp test acc = {:.2f}'.format(acc))
        return acc

    def train(self, test_features, test_labels, train_loader, test_loader):
        optimizer = torch.optim.AdamW(self.auto_prompt.parameters(), lr=self.cfg['lr'], eps=1e-3)
                
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(train_loader))

        beta, alpha = self.cfg['init_beta'], self.cfg['init_alpha']
        best_acc, best_epoch = 0.0, 0

        for train_idx in range(self.cfg['train_epoch']):
            # Train
            self.auto_prompt.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            align_text_loss_list = []
            align_visual_loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    # get image feat and normalization
                    image_features = self.clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                CoOp_logits, all_prompts = self.auto_prompt(image_features)

                loss = F.cross_entropy(CoOp_logits, target)
                
                cur_text_prototype = self.cupl_weights.unsqueeze(0).expand_as(all_prompts)
                score1 = self.cos(all_prompts, cur_text_prototype)
                loss2 = 1.0 - torch.mean(score1)

                loss = loss + 8 * loss2
                acc = cls_acc(CoOp_logits, target)
                correct_samples += acc / 100 * len(CoOp_logits)
                all_samples += len(CoOp_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                           correct_samples, all_samples,
                                                                           sum(loss_list) / len(loss_list)))
            torch.cuda.empty_cache()

            self.auto_prompt.eval()
            correct_samples, all_samples = 0, 0
            for i, (images, target) in enumerate(tqdm(test_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                CoOp_logits, all_prompts = self.auto_prompt(image_features)
                acc = cls_acc(CoOp_logits, target)
                correct_samples += acc / 100 * len(CoOp_logits)
                all_samples += len(CoOp_logits)
            acc = 100 * correct_samples/all_samples

            print("**** CoOp's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(self.auto_prompt.state_dict(),
                           self.cfg['cache_dir'] + "/best_CoOp_text_prompt_2" + str(self.cfg['shots']) + "shots.pt")
        print('best train acc = {:.2f}'.format(best_acc))