import json
import os
from pathlib import Path
import sys
from ..home import get_project_base

BASE = get_project_base()

def already_finished(logdir):
    fulldir = os.path.join(BASE, logdir)
    if os.path.exists(fulldir) and os.path.exists(os.path.join(fulldir, "FINISH_PROOF")):
        return True
    else:
        return False

def resume_ckpt(cfg, logdir):
    """
    return global_step, ckpt_file
    """
    if cfg.aux.resume == "" or ( not os.path.exists(logdir) ):
        print("No resume, Train from Scratch")
        return 0, None

    elif cfg.aux.resume == "max":

        if already_finished(logdir) and cfg.aux.skip_finished:
            print('----------------------------------------')
            print("Exp %s %s already finished, Skip it!" % (cfg.aux.exp, cfg.aux.runid))
            print('----------------------------------------')
            sys.exit()

        # find the latest ckpt
        ckptdir = os.path.join(logdir, 'ckpts')
        network_ckpts = os.listdir(ckptdir)
        network_ckpts = [ os.path.join(ckptdir, f) for f in network_ckpts ]
        if len(network_ckpts) == 0:
            print("No resume, Train from Scratch")
            return 0, None

        iterations = [ int(os.path.basename(f)[:-4].split("-")[-1]) for f in network_ckpts ]
        load_iteration = max(iterations)
        ckpt_file = os.path.join(ckptdir, "network.iter-%d.net" % load_iteration )
        print("Resume from", ckpt_file)
        return load_iteration, ckpt_file

    else: # resume is a path to a network ckpt
        assert os.path.exists(cfg.aux.resume)
        assert "Split%d" % cfg.split in cfg.aux.resume

        load_iteration = os.path.basename(cfg.aux.resume)
        load_iteration = int(load_iteration.split('.')[1].split('-')[1])
        print("Resume from", cfg.aux.resume)
        return load_iteration, cfg.aux.resume

def resume_wandb_runid(logdir):

    # if has prev_cfg, try reading from it
    prev_cfg_file = os.path.join(logdir, 'args.json')
    if os.path.exists(prev_cfg_file):
        with open(prev_cfg_file) as fp:
            prev_cfg = json.load(fp)
            if "wandb_id" in prev_cfg['aux']:
                return prev_cfg['aux']['wandb_id']

    # if has wandb folder, try reading from it
    logdir = Path(logdir)
    latest = logdir / "wandb" / "latest-run"
    if latest.exists():
        latest = latest.resolve()
        runid = latest.name.split('-')[-1]
        return runid

    # if none above works
    return None