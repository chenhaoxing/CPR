import os.path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import clip
import json
import time

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def my_acc(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)


def delete_tensor(arr, del_row):
    n = arr.cpu().detach().numpy()
    n = np.delete(n, del_row.cpu().detach().numpy(), 0)
    n = torch.from_numpy(n).cuda()
    return n


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print("Function {} executed in {:.2f} seconds.".format(func.__name__, (end - start)))

        return result

    return wrapper


def clip_classifier(cfg, classnames, template, clip_model, prompt_path=None):
    if os.path.exists(cfg['cache_dir'] + "/text_weights_cupl_t.pt"):
        print('**************** used CUPL !! *******************')
        return torch.load(cfg['cache_dir'] + "/text_weights_cupl_t.pt")
    else:
        raise NotImplementedError
    
    f = open(prompt_path)
    prompts = json.load(f)

    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = template_texts + cupl_texts
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_t.pt")
    return clip_weights


def pre_load_clip_weight(cfg):
    if os.path.exists(cfg['cache_dir'] + "/text_weights_cupl_t.pt"):
        print('**************** used CUPL !! *******************')
        return torch.load(cfg['cache_dir'] + "/text_weights_cupl_t.pt")
    else:
        # 
        print('**************** Processing self-defined prompt !! *******************')
        clip_weights = clip_classifier(cfg, classnames, template, clip_model, prompt_path=None)
        return clip_weights
        

def build_cache_model(cfg, clip_model, train_loader_cache=None):
    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values



def pre_load_features(cfg, split, clip_model, loader=None):
    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")

    return features, labels


@torch.no_grad()
def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None, RA=None):
    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                     range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                      range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0
        if adapter:
            affinity = adapter(features)
        elif RA:
            features = RA(features)
            cache_keys = RA(cache_keys.T).T
            affinity = features @ cache_keys
        else:

            affinity = features @ cache_keys
        for beta in beta_list:
            for alpha in alpha_list:
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha

                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def choose_channels(cfg, cache_keys, val_features, test_features, clip_weights):
    cache_keys = cache_keys.t().cuda()
    cfg['w'] = cfg['w_training_free']
    indices = cal_criterion(cfg, clip_weights, cache_keys)

    new_clip_weights = clip_weights[indices, :]  # [800, 47]
    new_cache_keys = cache_keys[:, indices]  # [47, 800]
    new_test_features = test_features[:, indices]  # [1692, 800]
    new_val_features = val_features[:, indices]  # [1128, 800]
    print('norm !')
    new_clip_weights = new_clip_weights / new_clip_weights.norm(dim=0, keepdim=True)
    new_cache_keys = new_cache_keys / new_cache_keys.norm(dim=-1, keepdim=True)
    new_test_features = new_test_features / new_test_features.norm(dim=-1, keepdim=True)
    new_val_features = new_val_features / new_val_features.norm(dim=-1, keepdim=True)

    return new_cache_keys, new_val_features, new_test_features, new_clip_weights, indices


def convert_seconds(seconds):
    hours, remain = divmod(seconds, 3600)
    minutes, seconds = divmod(remain, 60)
    print(f"{hours} hours, {minutes} minutes, {seconds} seconds")


def search_pse_hp(cfg, pse_cache_keys, pse_cache_values, val_features, new_val_features, val_labels, clip_weights,
                  val_cache_logits):
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                 range(cfg['search_step'][0])]
    gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in
                  range(cfg['search_step'][2])]
    best_acc = 0
    best_beta, best_gamma = 0.0, 0.0

    now_affinity = new_val_features @ pse_cache_keys.T
    clip_logits = 100. * val_features @ clip_weights

    for beta in beta_list:
        for gamma in gamma_list:
            now_cache_logits = ((-1) * (beta - beta * now_affinity)).exp() @ pse_cache_values
            logits = 2 * clip_logits + val_cache_logits + gamma * now_cache_logits
            acc = cls_acc(logits, val_labels)

            if acc > best_acc:
                best_acc = acc
                best_beta = beta
                best_gamma = gamma
    #print("\nAfter searching, the best accuarcy: {:.2f}., best gamma = {}\n".format(best_acc, best_gamma))
    return best_beta, best_gamma


def Tip_logits(cfg, cache_values, now_cache_values, test_features, clip_weights,
               new_cache_keys, now_new_cache_keys, new_test_features,
               best_beta, best_alpha, now_best_beta, best_gamma):
    affinity = new_test_features @ new_cache_keys.T
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    now_affinity = new_test_features @ now_new_cache_keys.T
    now_cache_logits = ((-1) * (now_best_beta - now_best_beta * now_affinity)).exp() @ now_cache_values

    clip_logits = 100. * test_features @ clip_weights

    tip_logits = (2 * clip_logits + cache_logits * best_alpha) + best_gamma * now_cache_logits

    return tip_logits


def init_psedu_cache(cfg, cache_values, test_features, new_test_features,
                     clip_weights, new_cache_keys, best_beta, best_alpha):
    class_num = cache_values.shape[1]
    affinity = new_test_features @ new_cache_keys.T
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    clip_logits = 100. * test_features @ clip_weights
    first_tip_logits = clip_logits + best_alpha * cache_logits
    best_scores, best_class_id = torch.max(first_tip_logits, dim=1)
    init_new_cache_keys = []
    init_cache_values = []

    for class_id in range(class_num):
        class_positions = best_class_id == class_id
        if class_positions.any():
            class_examples_scores = best_scores * class_positions
            _, good_examples = torch.topk(class_examples_scores, k=1)
            test_features_values = torch.zeros([len(good_examples), class_num]).cuda()
            test_features_values[:, class_id] = 1
            init_new_cache_keys.append(new_test_features[good_examples])
            init_cache_values.append(test_features_values)
    init_new_cache_keys = torch.cat(init_new_cache_keys, 0)
    init_cache_values = torch.cat(init_cache_values, 0).half()

    return init_new_cache_keys, init_cache_values


def get_best_param(cfg, val_features, new_val_features, val_labels, new_cache_keys, cache_values, clip_weights):
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                 range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                  range(cfg['search_step'][1])]

    best_acc = 0
    best_beta, best_alpha = 0.0, 0.0
    affinity = new_val_features @ new_cache_keys.T
    for beta in beta_list:
        for alpha in alpha_list:
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * val_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            acc = cls_acc(tip_logits, val_labels)

            if acc > best_acc:
                # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha
            # print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
    return best_beta, best_alpha

def evolving_cache_predict(cfg, new_cache_keys, cache_values,
                           new_val_features, val_features, val_labels,
                           new_test_features, test_features, test_labels,
                           clip_weights):
    class_num = cache_values.shape[1]
    shot_num = int(cache_values.shape[0] / class_num)

    remain_test_features = test_features
    remain_test_label = test_labels
    support_set_values = [cache_values[i * shot_num: (i + 1) * shot_num] for i in range(len(cache_values))]

    pseudolabel = [-1] * len(test_features)
    originial_id = dict(zip(list(range(len(test_features))), list(range(len(test_features)))))

    new_support_set = [new_cache_keys[i * shot_num: (i + 1) * shot_num] for i in range(len(new_cache_keys))]
    new_remain_test_features = new_test_features

    best_beta, best_alpha = get_best_param(cfg, val_features, new_val_features, val_labels, new_cache_keys, cache_values, clip_weights)

    pse_new_cache_keys, pse_cache_values = init_psedu_cache(cfg, cache_values, test_features, new_test_features,
                                                            clip_weights, new_cache_keys, best_beta, best_alpha)

    val_affinity = new_val_features @ new_cache_keys.T
    val_cache_logits = ((-1) * (best_beta - best_beta * val_affinity)).exp() @ cache_values * best_alpha

    while len(remain_test_features) > 0:

        now_cache_values = torch.cat(support_set_values, 0)
        now_new_cache_keys = torch.cat(new_support_set, 0)

        #now_best_beta, best_gamma = search_pse_hp(cfg, now_new_cache_keys, now_cache_values, val_features,
        #                                          new_val_features, val_labels, clip_weights, val_cache_logits)
        now_best_beta, best_gamma = search_pse_hp(cfg, pse_new_cache_keys, pse_cache_values, val_features,
                                                  new_val_features, val_labels, clip_weights, val_cache_logits)

        logits = Tip_logits(cfg, cache_values=cache_values,
                            now_cache_values=pse_cache_values,
                            test_features=remain_test_features,
                            clip_weights=clip_weights,
                            new_cache_keys=new_cache_keys,
                            now_new_cache_keys=pse_new_cache_keys,
                            new_test_features=new_remain_test_features,
                            best_beta=best_beta, best_alpha=best_alpha,
                            now_best_beta=now_best_beta, best_gamma=best_gamma
                            )
        '''
        logits = Tip_logits(cfg, cache_values=cache_values,
                            now_cache_values=now_cache_values,
                            test_features=remain_test_features,
                            clip_weights=clip_weights,
                            new_cache_keys=new_cache_keys,
                            now_new_cache_keys=now_new_cache_keys,
                            new_test_features=new_remain_test_features,
                            best_beta=best_beta, best_alpha=best_alpha,
                            now_best_beta=now_best_beta, best_gamma=best_gamma
                            )
        '''
        best_scores, best_class_id = torch.max(logits, dim=1)

        to_remove = []
        for class_id in range(class_num):
            class_positions = best_class_id == class_id
            if class_positions.any():

                class_examples_scores = best_scores * class_positions
                _, good_examples = torch.topk(class_examples_scores, k=1)

                if len(good_examples) > 0:
                    test_features_values = torch.zeros([len(good_examples), class_num]).cuda()
                    test_features_values[:, class_id] = 1

                    for e in good_examples.cpu().detach().numpy():
                        pseudolabel[originial_id[e]] = class_id
                    to_remove.append(good_examples)
                    new_support_set_i = new_support_set[class_id]
                    new_Q_i = new_remain_test_features[good_examples]
                    new_support_set[class_id] = torch.cat((new_support_set_i, new_Q_i), 0)
                    support_set_values[class_id] = torch.cat((support_set_values[class_id], test_features_values),
                                                             0).half()
                    pse_cache_values = torch.cat((pse_cache_values, test_features_values), 0).half()
                    pse_new_cache_keys = torch.cat((pse_new_cache_keys, new_Q_i), 0)

        for i in range(len(test_features)):
            l = torch.cat(to_remove, 0).cpu().detach().numpy()
            originial_id[i - sum([k < i for k in l])] = originial_id[i]

        remain_test_features = delete_tensor(remain_test_features, torch.cat(to_remove, 0))
        new_remain_test_features = delete_tensor(new_remain_test_features, torch.cat(to_remove, 0))
        remain_test_label = delete_tensor(remain_test_label, torch.cat(to_remove, 0))

    return pseudolabel


def pseudolabel_use_clip(unlabel_data_features, clip_weights):
    clip_logits = 100. * unlabel_data_features @ clip_weights
    best_scores, best_class_id = torch.max(clip_logits, dim=1)
    return best_scores, best_class_id


def TransProp(cache_keys, cache_values, val_features, beta):
    # feature processing
    n, m = cache_keys.size(1), val_features.size(0)
    all_feature = torch.cat((torch.transpose(cache_keys, 0, 1), val_features), 0)  # cat features：(n+m)*d
    affinity = all_feature @ torch.transpose(all_feature, 0, 1) - torch.eye(n + m).cuda(0)  # compute affinity matrix
    adj_matrix = torch.exp(affinity * beta)  # compute adj matrix

    # knn graph
    _, topk_index = torch.topk(adj_matrix, 10, 1)
    mask = torch.zeros_like(adj_matrix)
    mask = mask.scatter(1, topk_index, 1)
    mask = (mask + torch.t(mask) > 0).type(torch.float32)
    adj_matrix = adj_matrix * mask

    # graph normalization
    D = torch.sum(adj_matrix, 1, True)
    adj_matrix = adj_matrix / D  # row normalization

    # adj part
    a_uu, a_ul = adj_matrix[n:, n:], adj_matrix[n:, :n]  # part adj matrix

    # label propagation：cache_logits = (I-a_uu)^{-1} * a_ul * cache_values
    cache_logits = torch.matmul(torch.inverse(torch.eye(m).cuda(0).type(torch.float32) - a_uu) @ a_ul,
                                cache_values.type(torch.float32))
    return cache_logits


def cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=False):
    save_path = '/home/data_91_c/zhouqf/Tip-Adapter/caches/{}'.format(cfg['dataset'])
    save_file = '{}/criterion_{}_{}shot.pt'.format(save_path, cfg['backbone'].replace('/', ''), cfg['shots'])

    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1)
    cache_feat = None
    cache_feat = cache_keys.reshape(cate_num, cfg['shots'], feat_dim)

    if os.path.exists(save_file):
        print('Loading criterion...')
        sim = torch.load(save_file)

    elif only_use_txt:
        print('Calculating criterion...')

        feats = text_feat.squeeze()
        print(feats.shape)

        sim_sum = torch.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                if i != j:
                    sim_sum += feats[i, :] * feats[j, :]
                    count += 1
        sim = sim_sum / count
        torch.save(sim, save_file)
    else:
        print('Calculating criterion...')

        feats = torch.cat([text_feat, cache_feat], dim=1)
        samp_num = feats.shape[1]

        sim_sum = torch.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                for m in range(samp_num):
                    for n in range(samp_num):
                        if i != j:
                            sim_sum += feats[i, m, :] * feats[j, n, :]
                            count += 1
        sim = sim_sum / count
        torch.save(sim, save_file)

    criterion = (-1) * cfg['w'][0] * sim + cfg['w'][1] * torch.var(clip_weights, dim=1)
    _, indices = torch.topk(criterion, k=cfg['feat_num'])
    return indices


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        x = x.to(torch.float32)
        out = self.sharedMLP(x)
        return self.sigmoid(out).squeeze((2, 3))


class ResidualAdapter(nn.Module):
    def __init__(self, c_in, reduction=4, ratio=0.2):
        super(ResidualAdapter, self).__init__()
        self.ratio = ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.ratio * self.fc(x) + (1 - self.ratio) * x
        y = y / y.norm(dim=-1, keepdim=True)
        return y


class Lora(nn.Module):
    def __init__(self, in_dim, out_dim, rank=256):
        super(Lora, self).__init__()

        self.W_A = nn.Parameter(torch.empty(in_dim, rank), requires_grad=True)  # LoRA weight A
        self.W_B = nn.Parameter(torch.empty(rank, out_dim), requires_grad=True)  # LoRA weight B
        nn.init.kaiming_uniform_(self.W_A, a=np.sqrt(5))
        nn.init.zeros_(self.W_B)

    def forward(self, cache_keys):
        res_keys = self.W_A @ self.W_B
        new_cache_keys = cache_keys.clone()
        new_cache_keys = new_cache_keys + res_keys
        # new_cache_keys = new_cache_keys / new_cache_keys.norm(dim=-1, keepdim=True)
        return new_cache_keys


class Lora_rep(nn.Module):
    def __init__(self, class_num, shots, out_dim, rank=32):
        super(Lora_rep, self).__init__()
        self.W_A = nn.Parameter(torch.empty(class_num, rank), requires_grad=True)  # LoRA weight A
        self.W_C = nn.Parameter(torch.eye(rank), requires_grad=True)
        self.W_B = nn.Parameter(torch.empty(rank, out_dim), requires_grad=True)  # LoRA weight B
        nn.init.kaiming_uniform_(self.W_A, a=np.sqrt(5))
        nn.init.zeros_(self.W_B)
        self.shots = shots
        self.out_dim = out_dim

    def forward(self, cache_keys):
        res = self.W_A @ self.W_C @ self.W_B
        # res = self.W_A @ self.W_B
        res_keys = res.unsqueeze(1).repeat(1, self.shots, 1).reshape(-1, self.out_dim)
        res_text = res.t()
        new_cache_keys = cache_keys.clone()
        new_cache_keys = new_cache_keys.reshape(-1, self.out_dim)
        new_cache_keys = new_cache_keys + res_keys

        # new_clip_weight =

        return new_cache_keys
