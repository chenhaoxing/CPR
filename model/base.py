import numpy as np
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import search_hp, pre_load_clip_weight, cls_acc, build_cache_model, pre_load_features, delete_tensor, my_acc
import torch
import torch.nn as nn
from torch.nn import functional as F
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, ctx_init):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        self.CSC = False
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt.cuda()).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if self.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'
    
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix.cuda(),  # (n_cls, 1, dim)
                    ctx.cuda(),     # (n_cls, n_ctx, dim)
                    suffix.cuda(),  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

def _get_base_text_features(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()
    
    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)
    
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class BaseModel:
    def __init__(self, cfg, classnames,  templete, cupl_path, clip_model):
        self.cfg = cfg
        self.clip_model = clip_model
        self.clip_weights = pre_load_clip_weight(cfg)
        self.cache_keys, self.cache_values = build_cache_model(cfg, clip_model)
        self.prototypes = self.clip_weights.T
        self.val_features, self.val_labels = pre_load_features(cfg, 'val', clip_model)
        self.pse_cache_keys, self.pse_cache_values = None, None
        self.visual_prototypes = self.calculate_visual_prototypes()
        
    def logits(self, features, beta=None, alpha=None):
        raise NotImplementedError

    def evaluate(self, test_features, test_labels):
        raise NotImplementedError

    def calculate_visual_prototypes(self):
        features, labels = self.cache_keys, self.cache_values

        features = features.permute(1, 0)

        assert features.size(0) == labels.size(0), "特征向量的样本数应与标签的样本数相匹配"
    
        num_classes = labels.size(1)
        feature_dim = features.size(1)
        
        prototypes = torch.zeros(num_classes, feature_dim, device=features.device, dtype=features.dtype)

        for class_idx in range(num_classes):
            class_mask = labels[:, class_idx].bool()
            class_features = features[class_mask]
            
            if class_features.size(0) > 0:  
                class_prototype = class_features.mean(dim=0)
            else:
                class_prototype = torch.zeros(feature_dim, device=features.device, dtype=features.dtype)

            prototypes[class_idx] = class_prototype
        
        return prototypes
        

    def init_get_best_param(self):
        beta_list = [i * (self.cfg['search_scale'][0] - 0.1) / self.cfg['search_step'][0] + 0.1 for i in
                     range(self.cfg['search_step'][0])]
        alpha_list = [i * (self.cfg['search_scale'][1] - 0.1) / self.cfg['search_step'][1] + 0.1 for i in
                      range(self.cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0.0, 0.0
        affinity = self.val_features @ self.cache_keys
        for beta in beta_list:
            for alpha in alpha_list:
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
                clip_logits = 100. * self.val_features @ self.clip_weights
                tip_logits = clip_logits + cache_logits * alpha

                acc = cls_acc(tip_logits, self.val_labels)

                if acc > best_acc:
                    # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
                # print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
        return best_beta, best_alpha

    def update_prototype(self, class_id, features):
        epsilon = 0.999
        pse_prototype = torch.mean(features, dim=0)
        self.prototypes[class_id] = epsilon * self.prototypes[class_id] + (1 - epsilon) * pse_prototype
        features = epsilon * features + (1 - epsilon) * self.prototypes[class_id]
        return features

    def save_pse_cache(self):
        torch.save(self.pse_cache_keys, self.cfg['cache_dir'] + '/pse_keys_' + str(self.cfg['shots']) + "shots.pt")
        torch.save(self.pse_cache_values, self.cfg['cache_dir'] + '/pse_values_' + str(self.cfg['shots']) + "shots.pt")