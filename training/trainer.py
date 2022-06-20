# Author: Jacek Komorowski
# Warsaw University of Technology

# Train on Oxford dataset (from PointNetVLAD paper) using BatchHard hard negative mining.

import os
from datetime import datetime
import numpy as np
import torch
import pickle
import tqdm
import sys
sys.path.append(os.pardir)
import pathlib
import time

from torch.utils.tensorboard import SummaryWriter

from eval.evaluate import evaluate, print_eval_stats
from misc.utils import MinkLocParams, get_datetime
from models.loss import make_loss
from models.model_factory import model_factory

LOG_DIR = 'log'
LOG_FOUT = None
def log_string(out_str, will_print= False):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    if will_print:
        print(out_str)

def print_eval_stats(stats):
    for database_name in stats:
        log_string('Dataset: {}'.format(database_name), True)
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        log_string(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']), True)
        print(stats[database_name]['ave_recall'])

    
    
def print_stats(stats, phase):
    if 'num_pairs' in stats:
        # For batch hard contrastive loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Pairs per batch (all/non-zero pos/non-zero neg): {:.1f}/{:.1f}/{:.1f}'
        log_string(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pairs'],
                       stats['pos_pairs_above_threshold'], stats['neg_pairs_above_threshold']), True)
    elif 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        log_string(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']), True)
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        log_string(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']), True)

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        log_string(s.format(*l), True)


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def do_train(dataloaders, params: MinkLocParams, debug=False, visualize=False):
    # 用来log的
    global LOG_FOUT
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
#     LOG_FOUT.write(str() + '\n')
    
    
    # Create model class
    s = get_datetime()
    model = model_factory(params)
    model_name = "PointNetGeM"
#     model_name = 'model_' + params.model_params.model + '_' + s
#     print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, model_name)
    
    # 尝试加载模型
    model.load_state_dict(torch.load(model_pathname + '_final.pth'))
    log_string("Load Model", will_print= True)
                          
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"

    print('Model device: {}'.format(device))

    loss_fn = make_loss(params)

    # Training elements
    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################

    now = datetime.now()
    logdir = os.path.join("../tf_logs", now.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(logdir)

    ###########################################################################
    #
    ###########################################################################

    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}

    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        break
#     for epoch in range(1, params.epochs + 1):
        __epoch_start_time = time.time()
#         print(">_: EPOCH:{ep} -------------".format(ep=epoch))
#         if epoch > 10: break
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch

            count_batches = 0
            __cur_batch = 0
            
            for batch, positives_mask, negatives_mask in dataloaders[phase]:
                __Batch_start_time = time.time()
                log_string(">_: Batch:{}".format(__cur_batch))
                __cur_batch += 1
                # batch is (batch_size, n_points, 3) tensor
                # labels is list with indexes of elements forming a batch
                count_batches += 1
                batch_stats = {}

                if debug and count_batches > 2:
                    break

                batch = {e: batch[e].to(device) for e in batch}
                
#                 print(batch, len(batch), batch['cloud'], batch['cloud'].shape)
                myBatch = batch['cloud'].unsqueeze(1)
#                 print(myBatch.shape)
#                 return None
            
                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()
                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue

                optimizer.zero_grad()
                if visualize:
                    #visualize_batch(batch)
                    pass

                with torch.set_grad_enabled(phase == 'train'):
                    # Compute embeddings of all elements
                    embeddings = model(myBatch)
                    loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask)

                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)
                    batch_stats['loss'] = loss.item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_stats.append(batch_stats)
                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
            
                __Batch_end_time = time.time()
                log_string("Batch use time: {T:.4f}".format(T=__Batch_end_time-__Batch_start_time))
            # ******* PHASE END *******
            # Compute mean stats for the epoch
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        # ******* EPOCH END *******
        __epoch_end_time = time.time()
        log_string("One epoch need time: {T:.4f}".format(T = __epoch_end_time-__epoch_start_time))
        
        if scheduler is not None:
            scheduler.step()

        loss_metrics = {'train': stats['train'][-1]['loss']}
        if 'val' in phases:
            loss_metrics['val'] = stats['val'][-1]['loss']
        writer.add_scalars('Loss', loss_metrics, epoch)

        if 'num_triplets' in stats['train'][-1]:
            nz_metrics = {'train': stats['train'][-1]['num_non_zero_triplets']}
            if 'val' in phases:
                nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
            writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

        elif 'num_pairs' in stats['train'][-1]:
            nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
                          'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
            if 'val' in phases:
                nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
                nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
            writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

        if params.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()

    print('')
    print("跳过")

    # Save final model weights
#     final_model_path = model_pathname + '_final.pth'
#     torch.save(model.state_dict(), final_model_path)

    stats = {'train_stats': stats, 'params': params}

    # Evaluate the final model
    model.eval()
    final_eval_stats = evaluate(model, device, params)
    print('Final model:')
    print_eval_stats(final_eval_stats)
    stats['eval'] = {'final': final_eval_stats}
    print('')

    # Pickle training stats and parameters
    pickle_path = model_pathname + '_stats.pickle'
    pickle.dump(stats, open(pickle_path, "wb"))

    # Append key experimental metrics to experiment summary file
#     model_params_name = os.path.split(params.model_params.model_params_path)[1]
    model_params_name = ""
    config_name = os.path.split(params.params_path)[1]
    _, model_name = os.path.split(model_pathname)
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    export_eval_stats("experiment_results.txt", prefix, final_eval_stats)


def export_eval_stats(file_name, prefix, eval_stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in ['oxford', 'university', 'residential', 'business']:
            ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
