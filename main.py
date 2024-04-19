import random
import argparse
import yaml
import torch
from utils import *
from dataload import CustomDataload, ImagenetDataload
from model.TipAdapter import TipAdapter, EvTipAdapter
from model.CPR import CPR
from model.Clip import Clip
from model.TipAdapterF import TipAdapterF
from model.ClipAdapter import CLipAdapter
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--percentiles_holder', type=int, default=5)
    parser.add_argument('--model', dest='model')
    parser.add_argument('--subsample', type=str, default='all')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('/mnt/new/yansan/datasets/VLM_Adapt/caches', cfg['dataset'])
    cfg['cache_dir'] = cache_dir
    cfg['percentiles_holder'] = args.percentiles_holder

    if args.shots:
        cfg['shots'] = args.shots
        print('******************** shots = %d *************************' % args.shots)
    print('******************** dataset = %s *************************' % cfg['dataset'])


    base2new = True

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(3)
    torch.manual_seed(3)

    if base2new:
        cfg['subsample'] = 'base'
    else:
        cfg['subsample'] = 'all'
    data = CustomDataload(cfg, clip_model, preprocess)

    '''
    Training Free: CLip, TipAdapter
    '''
    if args.model == 'Clip':
        model = Clip(cfg, clip_model)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'TipAdapter':
        model = TipAdapter(cfg, clip_model)
        model.evaluate(data.test_features, data.test_labels)

    '''
    Training: Tip-Adapter-F, CoOp, Clip-Adapter, CPR
    '''
    if args.model == 'TipAdapterF':
        model = TipAdapterF(cfg, clip_model)
        model.train(data.test_features, data.test_labels, data.train_loader_F)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'CPR':
        model = CPR(cfg, data.dataset.classnames, data.dataset.template, data.dataset.cupl_path, clip_model)
        model.train(data.test_features, data.test_labels, data.train_loader_F, data.test_loader)
        base_acc = model.evaluate(data.test_features, data.test_loader, update=False)

        if base2new:
            cfg['subsample'] = 'new'
            cfg['load_cache'] = False
            cfg['load_pre_feat'] = False
            data = CustomDataload(cfg, clip_model, preprocess)

            model = CPR(cfg, data.dataset.classnames, data.dataset.template, data.dataset.cupl_path, clip_model)
            new_acc = model.evaluate(data.test_features, data.test_loader, update=False)
            H = 2 * base_acc * new_acc / (new_acc + base_acc)
            print('CoOp base acc = {:.2f}, new acc = {:.2f}, H = {:.2f}'.format(base_acc, new_acc, H))

    if args.model == 'ClipAdapter':
        model = CLipAdapter(cfg, clip_model)
        model.train(data.test_features, data.test_labels, data.train_loader_F)
        model.evaluate(data.test_features, data.test_labels)

