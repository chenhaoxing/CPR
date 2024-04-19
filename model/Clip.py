from model.base import BaseModel
from utils import cls_acc
import torch
from utils import delete_tensor
class Clip(BaseModel):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)

    def logits(self, features, beta=None, alpha=None):
        logits = features @ self.clip_weights
        return logits

    def evaluate(self, test_features, test_labels):
        clip_logits = self.logits(test_features)
        acc = cls_acc(clip_logits, test_labels)
        print('Clip test acc = {:.2f}'.format(acc))

class EvClip(Clip):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)

    def Evlogits(self, features, pse_beta, pse_alpha):
        text_logits = features @ self.clip_weights
        affinity = features @ self.pse_cache_keys.T
        pse_cache_logits = ((-1) * (pse_beta - pse_beta * affinity)).exp() @ self.pse_cache_values
        logits = text_logits + pse_alpha * pse_cache_logits
        return logits

    def evaluate(self, test_features, test_labels):
        pass

    def evolve_predict(self, test_features):
        self.pse_cache_keys, self.pse_cache_values = self.init_psedu_cache(test_features)

        remain_test_features = test_features
        pseudolabel = [-1] * len(test_features)
        original_id = dict(zip(list(range(len(test_features))), list(range(len(test_features)))))
        class_num = self.cache_values.shape[1]
        while len(remain_test_features) > 0:
            pse_beta, pse_alpha = self._get_best_param()

            logits = self.Evlogits(remain_test_features, pse_beta, pse_alpha)
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
                            pseudolabel[original_id[e]] = class_id
                        to_remove.append(good_examples)
                        Q_i = remain_test_features[good_examples]
                        self.pse_cache_keys = torch.cat((self.pse_cache_keys, Q_i), 0)
                        self.pse_cache_values = torch.cat((self.pse_cache_values, test_features_values), 0).half()
            for i in range(len(test_features)):
                l = torch.cat(to_remove, 0).cpu().detach().numpy()
                original_id[i - sum([k < i for k in l])] = original_id[i]

            remain_test_features = delete_tensor(remain_test_features, torch.cat(to_remove, 0))
        return pseudolabel


    def init_psedu_cache(self, test_features):
        class_num = self.clip_weights.shape[1]
        logits = self.logits(test_features)
        best_scores, best_class_id = torch.max(logits, dim=1)
        init_pse_cache_keys = []
        init_pse_cache_values = []

        for class_id in range(class_num):
            class_positions = best_class_id == class_id
            if class_positions.any():
                class_examples_scores = best_scores * class_positions
                _, good_examples = torch.topk(class_examples_scores, k=1)
                test_features_values = torch.zeros([len(good_examples), class_num]).cuda()
                test_features_values[:, class_id] = 1
                pse_features = self.update_prototype(class_id, test_features[good_examples])
                init_pse_cache_keys.append(pse_features)
                init_pse_cache_values.append(test_features_values)
        init_pse_cache_keys = torch.cat(init_pse_cache_keys, 0)
        init_pse_cache_values = torch.cat(init_pse_cache_values, 0).half()

        return init_pse_cache_keys, init_pse_cache_values

    def _get_best_param(self):

        pse_beta_list = [i * (self.cfg['search_scale'][0] - 0.1) / self.cfg['search_step'][0] + 0.1 for i in
                         range(self.cfg['search_step'][0])]
        pse_alpha_list = [i * (self.cfg['search_scale'][2] - 0.1) / self.cfg['search_step'][2] + 0.1 for i in
                          range(self.cfg['search_step'][2])]

        best_acc = 0
        best_beta, best_alpha = 0.0, 0.0
        now_affinity = self.val_features @ self.pse_cache_keys.T
        text_logits = 100. * self.val_features @ self.clip_weights

        for beta in pse_beta_list:
            for alpha in pse_alpha_list:
                now_cache_logits = ((-1) * (beta - beta * now_affinity)).exp() @ self.pse_cache_values
                logits = text_logits + alpha * now_cache_logits
                acc = cls_acc(logits, self.val_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
        return best_beta, best_alpha