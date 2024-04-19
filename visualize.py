import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import torch
from utils import build_cache_model, pre_load_features
import yaml
import matplotlib.cm as cm

dataset = 'ucf101'

pse_keys = torch.load('/home/data_91_c/zhouqf/Tip-Adapter/caches/' + dataset + '/pse_keys_1shots.pt').detach().cpu().numpy()

pse_values = torch.load('/home/data_91_c/zhouqf/Tip-Adapter/caches/' + dataset + '/pse_values_1shots.pt').detach().cpu().numpy()
pse_y = np.argmax(pse_values, axis=1)
print(pse_values.shape)
cfg = yaml.load(open('./configs/oxford_flowers.yaml', 'r'), Loader=yaml.Loader)
test_labels = torch.load('/home/data_91_c/zhouqf/Tip-Adapter/caches/' + dataset + '/test_l.pt').detach().cpu().numpy()
test_keys = torch.load('/home/data_91_c/zhouqf/Tip-Adapter/caches/' + dataset + '/test_f.pt').detach().cpu().numpy()


cache_keys = torch.load('/home/data_91_c/zhouqf/Tip-Adapter/caches/' + dataset + '/keys_1shots.pt')
cache_values = torch.load('/home/data_91_c/zhouqf/Tip-Adapter/caches/' + dataset + '/values_1shots.pt')
cache_keys = cache_keys.T.detach().cpu().numpy()
cache_values = cache_values.detach().cpu().numpy()

select_indices =  [15, 41, 42]
print(select_indices)
cache_y = np.argmax(cache_values, axis=1)
cache_indices = np.in1d(cache_y, select_indices)
cache_keys = cache_keys[cache_indices]
cache_y = cache_y[cache_indices]

test_indices = np.in1d(test_labels, select_indices)
test_keys = test_keys[test_indices]
test_labels = test_labels[test_indices]

pse_indices = np.in1d(pse_y, select_indices)
pse_keys = pse_keys[pse_indices]
pse_y = pse_y[pse_indices]

ground_keys = np.concatenate((cache_keys, test_keys), axis=0)
ground_values = np.concatenate((cache_y, test_labels), axis=0)

ground_truth_centroids = []
for i in range(len(select_indices)):
    class_samples = ground_keys[ground_values == select_indices[i]]
    class_mean = np.mean(class_samples, axis=0)
    ground_truth_centroids.append(class_mean)
ground_truth_centroids = np.array(ground_truth_centroids)

pse_centroids = []
for i in range(len(select_indices)):
    class_samples = np.concatenate((pse_keys[pse_y == select_indices[i]], cache_keys[cache_y == select_indices[i]]), axis=0)
    class_mean = np.mean(class_samples, axis=0)
    pse_centroids.append(class_mean)
pse_centroids = np.array(pse_centroids)

cache_centroids = []
for i in range(len(select_indices)):
    class_samples = cache_keys[cache_y == select_indices[i]]
    class_mean = np.mean(class_samples, axis=0)
    cache_centroids.append(class_mean)
cache_centroids = np.array(cache_centroids)

tsne = TSNE(n_components=2)
all_data = np.concatenate((test_keys, cache_centroids, pse_centroids, ground_truth_centroids), axis=0)
all_data_tsne = tsne.fit_transform(all_data)

test_tsne = all_data_tsne[:len(test_keys)]
cache_centroids_tsne = all_data_tsne[len(test_keys):len(test_keys) + len(cache_centroids)]
pse_centroids_tsne = all_data_tsne[len(test_keys) + len(cache_centroids): len(test_keys) + len(cache_centroids) + len(pse_centroids)]
ground_truth_tsne = all_data_tsne[len(test_keys) + len(cache_centroids) + len(pse_centroids):]


print('sum test samples for plot = ', test_tsne.shape[0])
blue = (0, 0, 1)
red = (1, 0, 0)
color_list = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'gray']
marker_list = ['o', 'x', '+', '*', '^', 'v', '<', '>', 'p', 'h']
labels = select_indices
plt.figure(figsize=(10, 8))
for i in range(len(select_indices)):
    plt.scatter(test_tsne[test_labels == select_indices[i], 0], test_tsne[test_labels == select_indices[i], 1], c=color_list[i], alpha=0.5)
    plt.scatter(cache_centroids_tsne[i, 0], cache_centroids_tsne[i, 1], c=color_list[i], marker='p', s=300,  edgecolor='black', linewidth=3, label=labels[i])
    plt.scatter(pse_centroids_tsne[i, 0], pse_centroids_tsne[i, 1], c=color_list[i], marker='h', s=300, edgecolor='black', linewidth=3, label=labels[i])
    plt.scatter(ground_truth_tsne[i, 0], ground_truth_tsne[i, 1], c=color_list[i], marker='v', s=300, edgecolor='black', linewidth=3, label=labels[i])
plt.legend()
plt.title('t-SNE visualization of training and testing set samples')
plt.savefig('./pic/' + dataset + '_1shots.png')
'''
plt.clf()


tsne = TSNE(n_components=2)
all_data = np.concatenate((test_keys, pse_centroids), axis=0)
all_data_tsne = tsne.fit_transform(all_data)

test_tsne = all_data_tsne[:len(test_keys)]
pse_centroids_tsne = all_data_tsne[len(test_keys):]
print(pse_centroids_tsne)
color_list = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'gray']
marker_list = ['o', 'x', '+', '*', '^', 'v', '<', '>', 'p', 'h']

plt.figure(figsize=(10, 8))
for i in range(len(select_indices)):
    plt.scatter(test_tsne[test_labels == select_indices[i], 0], test_tsne[test_labels == select_indices[i], 1], c=color_list[i], alpha=0.5)
    plt.scatter(pse_centroids_tsne[i, 0], pse_centroids_tsne[i, 1], c=color_list[i], marker='p', s=300, edgecolor='black', linewidth=3, label=labels[i])
plt.legend()
plt.title('t-SNE visualization of training and testing set samples')
plt.savefig('./pic/' + dataset + '_1shots_after.png')
'''