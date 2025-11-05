# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import time
import sys
import torch
import logging
import json
import numpy as np
import random
import pickle

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from src.options import Options
from src import data, beir_utils, dist_utils, utils
from src import moco, inbatch


logger = logging.getLogger(__name__)

# ---------------------------
# Device selection
# ---------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def train(opt, model, optimizer, scheduler, step):

    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)

    logger.info("Data loading")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer
    collator = data.Collator(opt=opt)
    train_dataset = data.load_data(opt, tokenizer)
    logger.warning(f"Data loading finished for rank {dist_utils.get_rank()}")
    logger.info(f"Train dataset length: {len(train_dataset)}")

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    epoch = 1

    model.train()
    while step < opt.total_steps:
        train_dataset.generate_offset()

        logger.info(f"Start epoch {epoch}")
        for i, batch in enumerate(train_dataloader):
            logger.info(f"Batch {i}/{len(train_dataloader)} (epoch {epoch})")
            step += 1

            # Move tensors to device
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}

            train_loss, iter_stats = model(**batch, stats_prefix="train")

            train_loss.backward()
            optimizer.step()

            scheduler.step()
            model.zero_grad()

            run_stats.update(iter_stats)

            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3f}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                if device.type == "cuda":
                    log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                logger.info(log)
                run_stats.reset()

            if step % opt.eval_freq == 0:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    encoder = model.module.get_encoder()
                else:
                    encoder = model.get_encoder()
                eval_model(
                    opt, query_encoder=encoder, doc_encoder=encoder, tokenizer=tokenizer, tb_logger=tb_logger, step=step
                )

                if dist_utils.is_main():
                    utils.save(model, optimizer, scheduler, step, opt, opt.output_dir, f"lastlog")

                model.train()

            if dist_utils.is_main() and step % opt.save_freq == 0:
                utils.save(model, optimizer, scheduler, step, opt, opt.output_dir, f"step-{step}")

            if step > opt.total_steps:
                break
        epoch += 1


def eval_model(opt, query_encoder, doc_encoder, tokenizer, tb_logger, step):
    for datasetname in opt.eval_datasets:
        metrics = beir_utils.evaluate_model(
            query_encoder,
            doc_encoder,
            tokenizer,
            dataset=datasetname,
            batch_size=opt.per_gpu_eval_batch_size,
            norm_doc=opt.norm_doc,
            norm_query=opt.norm_query,
            beir_dir=opt.eval_datasets_dir,
            score_function=opt.score_function,
            lower_case=opt.lower_case,
            normalize_text=opt.eval_normalize_text,
        )

        message = []
        if dist_utils.is_main():
            for metric in ["NDCG@10", "Recall@10", "Recall@100"]:
                message.append(f"{datasetname}/{metric}: {metrics[metric]:.2f}")
                if tb_logger is not None:
                    tb_logger.add_scalar(f"{datasetname}/{metric}", metrics[metric], step)
            logger.info(" | ".join(message))


if __name__ == "__main__":
    logger.info("Start")

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    
    directory_exists = os.path.isdir(opt.output_dir)
    checkpoint_exists = False
    if directory_exists:
        checkpoint_path = os.path.join(opt.output_dir, "checkpoint", "latest", "checkpoint.pth")
        checkpoint_exists = os.path.isfile(checkpoint_path)

    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    utils.init_logger(opt)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if opt.contrastive_mode == "moco":
        model_class = moco.MoCo
    elif opt.contrastive_mode == "inbatch":
        model_class = inbatch.InBatch
    else:
        raise ValueError(f"contrastive mode: {opt.contrastive_mode} not recognised")

    print(f"directory_exists: {directory_exists}")
    print(f"checkpoint_exists: {checkpoint_exists}")
    print("the directory is :" + opt.output_dir)
    if not directory_exists and opt.model_path == "none":
        model = model_class(opt).to(device)
        optimizer, scheduler = utils.set_optim(opt, model)
        step = 0
    elif directory_exists and checkpoint_exists:
        model_path = os.path.join(opt.output_dir, "checkpoint", "latest")
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class,
            model_path,
            opt,
            reset_params=False,
        )
        model = model.to(device)
        step = 6500
        logger.info(f"Model loaded from {opt.output_dir}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class,
            opt.model_path,
            opt,
            reset_params=False if opt.continue_training else True,
        )
        model = model.to(device)
        if not opt.continue_training:
            step = 0
        logger.info(f"Model loaded from {opt.model_path}")

    logger.info(utils.get_parameters(model))

    logger.info("Start training")
    train(opt, model, optimizer, scheduler, step)
