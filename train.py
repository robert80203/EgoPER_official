##############################################################################################
# The code is modified from ActionFormer: https://github.com/happyharrycn/actionformer_release
##############################################################################################

# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our implementation
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


################################################################################
def main(args):
    """main function that handles training / inference"""
    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    #########################
    ####-- Commented this out
    # if not os.path.exists(cfg['output_folder']):
    #     os.mkdir(cfg['output_folder'])
    #########################
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    ckpt_folder = os.path.join(args.cp_dir, cfg_filename + '_' + str(args.output))
    print(ckpt_folder)
    # ./ckpt/EgoPER/oatmeal_aod_bgr1.0_final
    print('checkpoint_dir: ', ckpt_folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok = True)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    # rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    rng_generator = fix_random_seed(args.seed, include_cuda=True) # change to other seed! - 5/11

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    
    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], feat_dirname = args.feat_dirname, data_root_dir = args.data_root_dir, **cfg['dataset']
    )

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    cfg['model']['input_dim'] = args.input_dim
    cfg['devices'] = ['cuda:0']
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    if cfg['model']['train_cfg']['contrastive']:
        model.init_prototypes()

    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'], strict=False)
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint    
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    print('Maximum epoch:', max_epochs)
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer = tb_writer,
            print_freq = args.print_freq,
            use_contrastive = cfg['train_cfg']['contrastive'],
            batch_size = cfg['loader']['batch_size'],
            max_videos = args.maxvideos
        )

        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            if (epoch + 1) > int(max_epochs * 0.85):
                print('saving checkpoint...')
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
                )
            
    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=2, type=int,
                        help='print frequency (default: 2 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--maxvideos', default=15, type=int,
                        help='number of training videos in each prototype (maxvideos * batch size)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--feat_dirname', default='feature_10fps', type=str,
                        help='feature directory name')
    ###########
    #-- Changes here
    parser.add_argument('--input_dim', default = 768, type = int,
                       help = 'feature input dimension')
    parser.add_argument('--cp_dir', default = 'temp', type = str,
                       help = 'checkpoint directory')
    parser.add_argument('--seed', default = 0, type = int,
                       help = 'random seed')
    parser.add_argument('--data_root_dir', default = './data', type = str,
                       help = 'data root directory')
    ###########
    args = parser.parse_args()
    main(args)
