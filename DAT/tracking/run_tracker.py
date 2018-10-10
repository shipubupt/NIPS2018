import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import time
import argparse
import json
from PIL import Image

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0, '../modules')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)
import guided_backprop

import scipy
import gc

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)
    return feats


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

def forwordgradients(model,batch_posimgids,batch_possamples,GBP,vector):
    #opts['ft_layers'] = ['conv', 'fc']
    #model.set_learnable_params(opts['ft_layers'])


    uniqueimgid = np.unique(batch_posimgids)
    theuniI=0
    for theuniqid in uniqueimgid:
        theuniexps = []
        for ind in range(batch_possamples.shape[0]):
            if batch_posimgids[ind] == theuniqid:
                theuniexps.append(batch_possamples[ind])

        theuniexps = torch.from_numpy(np.stack(theuniexps, 0)).view(-1, 4).numpy()
        theimg = Image.open(img_list[theuniqid]).convert('RGB')
        extractor = RegionExtractor(theimg, theuniexps, opts['img_size'], opts['padding'], opts['batch_test'])
        for i, regions in enumerate(extractor):
            regions = Variable(regions)
            if opts['use_gpu']:
                regions = regions.cuda()
            if i == 0:
                feats = regions
            else:
                feats = torch.cat((feats, regions), 0)
        if theuniI==0:
            posregions=feats
        else:
            posregions = torch.cat((posregions, feats), 0)
        theuniI=theuniI+1



    guided_grads = GBP.generate_gradients(posregions,vector)
    #opts['ft_layers'] = ['fc']
    #model.set_learnable_params(opts['ft_layers'])
    return guided_grads


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, posimgids,possamples,negimgids,negsamples,imglist,GBP, in_layer='fc4'):
    model.train()
    #spcriterion = spatialBinaryLoss()
    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while (len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while (len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        batch_posimgids=posimgids[pos_cur_idx.cpu().numpy()]
        batch_negimgids=negimgids[neg_cur_idx.cpu().numpy()]

        batch_possamples=possamples[pos_cur_idx.cpu().numpy()]
        batch_negsamples=negsamples[neg_cur_idx.cpu().numpy()]


        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            batch_negimgids=batch_negimgids[top_idx.cpu().numpy()]
            batch_negsamples=batch_negsamples[top_idx.cpu().numpy()]

            model.train()


        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        '''
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()
        '''

        #gradientforward

        posguided_grads=forwordgradients(model,batch_posimgids,batch_possamples,GBP,[0,10])
        sumposguided_grads= torch.sum(torch.abs(posguided_grads) ** 2, 1)
        posgradmean=torch.nn.functional.avg_pool2d(sumposguided_grads,[sumposguided_grads.size(1),sumposguided_grads.size(2)])

        posgraddelta=(sumposguided_grads-posgradmean)**2
        posgraddelta=torch.sqrt(torch.nn.functional.avg_pool2d(posgraddelta,[posgraddelta.size(1),posgraddelta.size(2)]))
        #posgradscore=1/posgraddelta

        posguided_grads_1 = forwordgradients(model, batch_posimgids, batch_possamples, GBP, [10, 0])
        sumposguided_grads_1 = torch.sum(torch.abs(posguided_grads_1) ** 2, 1)
        posgradmean_1 = torch.nn.functional.avg_pool2d(sumposguided_grads_1,
                                                     [sumposguided_grads_1.size(1), sumposguided_grads_1.size(2)])

        posgraddelta_1 = (sumposguided_grads_1 - posgradmean_1) ** 2
        posgraddelta_1 = torch.sqrt(
            torch.nn.functional.avg_pool2d(posgraddelta_1, [posgraddelta_1.size(1), posgraddelta_1.size(2)]) )
        # posgradscore=1/posgraddelta


        #spPosloss=spcriterion(posgradscore,-1)

        '''
        model.zero_grad()
        spPosloss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()
        '''


        negguided_grads=forwordgradients(model,batch_negimgids,batch_negsamples,GBP,[10,0])
        sumnegguided_grads=torch.sum(torch.abs(negguided_grads) ** 2, 1)
        neggradmean = torch.nn.functional.avg_pool2d(sumnegguided_grads,
                                                     [sumnegguided_grads.size(1), sumnegguided_grads.size(2)])

        neggraddelta = (sumnegguided_grads - neggradmean) ** 2
        neggraddelta = torch.sqrt(
            torch.nn.functional.avg_pool2d(neggraddelta, [neggraddelta.size(1), neggraddelta.size(2)]) )
        #neggradscore = 1 / neggraddelta


        negguided_grads_1 = forwordgradients(model, batch_negimgids, batch_negsamples, GBP, [0, 10])
        sumnegguided_grads_1 = torch.sum(torch.abs(negguided_grads_1) ** 2, 1)
        neggradmean_1 = torch.nn.functional.avg_pool2d(sumnegguided_grads_1,
                                                     [sumnegguided_grads_1.size(1), sumnegguided_grads_1.size(2)])

        neggraddelta_1 = (sumnegguided_grads_1 - neggradmean_1) ** 2
        neggraddelta_1 = torch.sqrt(
            torch.nn.functional.avg_pool2d(neggraddelta_1, [neggraddelta_1.size(1), neggraddelta_1.size(2)]) )

        #spNegloss = spcriterion(neggradscore, 1)

        '''
        model.zero_grad()
        spNegloss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()
        '''


        ###############v3 method##########
        spaloss=(1e10*posgraddelta.sum())/(1e10*posgradmean.sum())+(1e10*neggraddelta.sum())/(1e10*neggradmean.sum())\
                +(1e10*posgradmean_1.sum())/(1e10*posgraddelta_1.sum())+(1e10*neggradmean_1.sum())/(1e10*neggraddelta_1.sum())


        thelamb=5
        ###############v3 method##########

        finalloss=thelamb*spaloss+loss
        #print spaloss
        model.zero_grad()
        finalloss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()




        #print "Iter %d, Loss %.4f, spaLoss %.4f" % (iter, finalloss.data[0], spaloss.data[0])


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False):
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
    GBP = guided_backprop.GuidedBackprop(model, 1)
    # Init criterion and optimizer
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    # Train bbox regressor
    bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    # Draw pos/neg samples
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)

    pos_imgids=np.array([[0]]*pos_feats.size(0))
    neg_imgids=np.array([[0]]*neg_feats.size(0))

    feat_dim = pos_feats.size(-1)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'],
            pos_imgids, pos_examples, neg_imgids, neg_examples,img_list,GBP)

    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]

    pos_examples_all=[pos_examples[:opts['n_pos_update']]]
    neg_examples_all=[neg_examples[:opts['n_neg_update']]]

    pos_imgids_all = [pos_imgids[:opts['n_pos_update']]]
    neg_imgids_all = [neg_imgids[:opts['n_neg_update']]]



    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # Estimate target bbox
        samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']

        # Expand search area at failure
        if success:
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Copy previous result at failure
        if not success:
            target_bbox = result[i - 1]
            bbreg_bbox = result_bb[i - 1]

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(pos_generator, target_bbox,
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox,
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = forward_samples(model, image, pos_examples)
            neg_feats = forward_samples(model, image, neg_examples)
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)

            pos_examples_all.append(pos_examples)
            neg_examples_all.append(neg_examples)

            pos_imgids_all.append(np.array([[i]]*pos_feats.size(0)))
            neg_imgids_all.append(np.array([[i]]*neg_feats.size(0)))



            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
                del pos_examples_all[0]
                del pos_imgids_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]
                del neg_examples_all[0]
                del neg_imgids_all[0]



        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)

            pos_examples_data=torch.from_numpy(np.stack(pos_examples_all[-nframes:],0)).view(-1, 4).numpy()
            neg_examples_data=torch.from_numpy(np.stack(neg_examples_all,0)).view(-1, 4).numpy()

            pos_imgids_data=torch.from_numpy(np.stack(pos_imgids_all[-nframes:],0)).view(-1, 1).numpy()
            neg_imgids_data=torch.from_numpy(np.stack(neg_imgids_all,0)).view(-1, 1).numpy()


            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'],
                  pos_imgids_data, pos_examples_data, neg_imgids_data, neg_examples_data,img_list,GBP)

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)

            pos_examples_data = torch.from_numpy(np.stack(pos_examples_all, 0)).view(-1, 4).numpy()
            neg_examples_data = torch.from_numpy(np.stack(neg_examples_all, 0)).view(-1, 4).numpy()

            pos_imgids_data = torch.from_numpy(np.stack(pos_imgids_all, 0)).view(-1, 1).numpy()
            neg_imgids_data = torch.from_numpy(np.stack(neg_imgids_all, 0)).view(-1, 1).numpy()

            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'],
                  pos_imgids_data, pos_examples_data, neg_imgids_data, neg_examples_data,img_list,GBP)

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                if gt.shape[0] > i:
                    gt_rect.set_xy(gt[i, :2])
                    gt_rect.set_width(gt[i, 2])
                    gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '%04d.jpg' % (i)), dpi=dpi)

        if gt is None:
            print "Frame %d/%d, Score %.3f, Time %.3f" % \
                  (i, len(img_list), target_score, spf)
        else:
            if gt.shape[0] > i:
                print "Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                      (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf)

    fps = len(img_list) / spf_total

    return result, result_bb, fps



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()
    args.display = False
    args.savefig = True
    assert (args.seq != '' or args.json != '')

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)


    # Run tracker
    result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

    # Save result
    res = {}
    res['res'] = result_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(result_path, 'w'), indent=2)
