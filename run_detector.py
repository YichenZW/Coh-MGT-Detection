# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Based on code from the above authors, modifications made by Xi'an Jiaotong University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import argparse
import logging
import random
import wandb
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.optim import AdamW
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from functools import partial
from util import glue_compute_metrics as compute_metrics
from util import (
    glue_convert_examples_to_features as convert_examples_to_features,
)
from util import glue_output_modes as output_modes
from util import glue_processors as processors

from modeling_roberta import (
    RobertaForGraphBasedSequenceClassification,
    RobertaForGraphBasedSequenceClassification_CL,
    RobertaForGraphBasedSequenceClassification_MBCL,
    EncoderForMBCL,
    RobertaForGraphBasedSequenceClassification_RFCL,
)

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def number_h(num):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")

def generate_shaped_nodes_mask(nodes, max_seq_length, max_nodes_num):
    nodes_mask = np.zeros(shape=(max_nodes_num, max_seq_length))
    nodes_num = min(len(nodes), max_nodes_num)

    for i in range(nodes_num):
        span = nodes[i]
        if span[0] != -1:
            if span[0] < max_seq_length - 1:
                end_pos = (
                    span[1] if span[1] < max_seq_length - 1 else max_seq_length - 1
                )
                nodes_mask[i, span[0] + 1 : end_pos + 1] = 1
            else:
                continue
    return nodes_mask, nodes_num

def generate_shaped_edge_mask(adj_metric, nodes_num, max_nodes_num, relation_n):
    if nodes_num != 0:
        if relation_n != 0:
            new_adj_metric = np.zeros(shape=(relation_n, max_nodes_num, max_nodes_num))
            for i in range(relation_n):
                new_adj_metric[i][:nodes_num, :nodes_num] = adj_metric[i][
                    :nodes_num, :nodes_num
                ]
        else:
            new_adj_metric = np.zeros(shape=(max_nodes_num, max_nodes_num))
            new_adj_metric[:nodes_num, :nodes_num] = adj_metric[:nodes_num, :nodes_num]
    return new_adj_metric

def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Total Params:", number_h(total_params))
    print("Total Trainable Params:", number_h(total_trainable_params))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    best_acc, best_f1 = 0.0, 0.0
    global_step, epochs_trained, steps_trained_in_current_epoch = 0, 0, 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)
    max_acc, max_acc_f1, max_f1, max_f1_acc = 0.0, 0.0, 0.0, 0.0
    for idx, _ in enumerate(train_iterator):
        tr_loss = 0.0
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "nodes_index_mask": batch[4],
                "adj_metric": batch[5],
                "node_mask": batch[6],
                "sen2node": batch[7],
                "sentence_mask": batch[8],
                "sentence_length": batch[9],
                "batch_id": batch[10],
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                ) 
            
            outputs, _ = model(**inputs)
            
            loss = outputs[0]
            wandb.log({"train/loss": loss})
            if args.n_gpu > 1:
                loss = loss.mean() 
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(
                "loss {}".format(
                    round(tr_loss * args.gradient_accumulation_steps / (step + 1), 4)
                )
            )
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  
                model.zero_grad()
                global_step += 1
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss
                    wandb.log({"eval/loss": loss_scalar})

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0] and args.save_steps > 0 and args.do_eval:
            results = evaluate(args, model, tokenizer, checkpoint=str(idx))
            logger.info("the results is {}".format(results))
            if results["acc"] > max_acc:
                max_acc = results["acc"]
                max_acc_f1 = results["f1"]
            if results["f1"] > max_f1:
                max_f1 = results["f1"]
                max_f1_acc = results["acc"]
            if results["f1"] > best_f1:
                best_f1 = results["f1"]

                output_dir = os.path.join(
                    args.output_dir,
                    "seed-{}".format(args.seed),
                    "checkpoint-{}-{}".format(idx, best_f1),
                )  
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(
                    args, os.path.join(output_dir, "training_{}.bin".format(idx))
                )

                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return_res = {
        "max_acc": max_acc,
        "max_acc_f1": max_acc_f1,
        "max_f1": max_f1,
        "max_f1_acc": max_f1_acc,
    }
    if args.do_ray:
        tune.report(
            accuracy=max_acc, max_acc_f1=max_acc_f1, f1=max_f1, max_f1_acc=max_f1_acc
        )
    return global_step, tr_loss / global_step, return_res, output_dir


def mb_train(args, train_dataset, encoder_q, encoder_k, dataloader, tokenizer):
    """Train the model"""
    global memory_queue
    encoder_q.train()

    total_params = sum(p.numel() for p in encoder_q.parameters())
    total_trainable_params = sum(
        p.numel() for p in encoder_q.parameters() if p.requires_grad
    )

    print("Encoder Params:", number_h(total_params))
    print("Encoder Trainable Params:", number_h(total_trainable_params))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in encoder_q.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in encoder_q.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    best_f1 = 0.0
    global_step, epochs_trained, steps_trained_in_current_epoch = 0, 0, 0
    tr_loss, logging_loss = 0.0, 0.0
    encoder_q.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)
    max_acc, max_acc_f1, max_f1, max_f1_acc = 0.0, 0.0, 0.0, 0.0
    for idx, _ in enumerate(train_iterator):
        tr_loss = 0.0
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            encoder_q.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "nodes_index_mask": batch[4],
                "adj_metric": batch[5],
                "node_mask": batch[6],
                "sen2node": batch[7],
                "sentence_mask": batch[8],
                "sentence_length": batch[9],
                "batch_id": batch[10],
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            q_outputs, q_rep = encoder_q(**inputs)
            # Model outputs are always tuple in transformers (see doc).

            if args.n_gpu > 1:
                loss = loss.mean() 
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(
                "loss {}".format(
                    round(tr_loss * args.gradient_accumulation_steps / (step + 1), 4)
                )
            )
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        encoder_q.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step() 
                encoder_q.zero_grad()
                global_step += 1
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, encoder_q, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss
                    wandb.log({"train/loss": loss_scalar})

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0] and args.save_steps > 0 and args.do_eval:
            results = evaluate(args, encoder_q, tokenizer, checkpoint=str(idx))
            logger.info("the results is {}".format(results))
            if results["f1"] > max_f1:
                max_f1 = results["f1"]
                max_f1_acc = results["acc"]
            if results["acc"] > max_acc:
                max_acc = results["acc"]
                max_acc_f1 = results["f1"]
            if results["f1"] > best_f1:
                best_f1 = results["f1"]

                output_dir = os.path.join(
                    args.output_dir,
                    "seed-{}".format(args.seed),
                    "checkpoint-{}-{}".format(idx, best_f1),
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    encoder_q.module if hasattr(encoder_q, "module") else encoder_q
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(
                    args, os.path.join(output_dir, "training_{}.bin".format(idx))
                )
                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return_res = {
        "max_acc": max_acc,
        "max_acc_f1": max_acc_f1,
        "max_f1": max_f1,
        "max_f1_acc": max_f1_acc,
    }
    if args.do_ray:
        tune.report(
            accuracy=max_acc, max_acc_f1=max_acc_f1, f1=max_f1, max_f1_acc=max_f1_acc
        )
    return global_step, tr_loss / global_step, return_res, output_dir


def evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode="dev"):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=True, mode=mode
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly.
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Evaluation
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds, out_label_ids = None, None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                    "nodes_index_mask": batch[4],
                    "adj_metric": batch[5],
                    "node_mask": batch[6],
                    "sen2node": batch[7],
                    "sentence_mask": batch[8],
                    "sentence_length": batch[9],
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2]
                        if args.model_type in ["bert", "xlnet", "albert"]
                        else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs, _ = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )
        probs = preds
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        wandb.log(
            {
                "eval/acc": result["acc"],
                "eval/f1": result["f1"],
                "eval/acc_and_f1": result["acc_and_f1"],
            }
        )
    return results


def load_and_cache_examples(
    args, task, tokenizer, evaluate=False, mode="train", dataset_name="", rel=""
):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(dataset_name),
            str(rel),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        if mode == "train":
            examples = processor.get_train_examples(args.with_relation, args.data_dir)
        elif mode == "dev":
            examples = processor.get_dev_examples(args.with_relation, args.data_dir)
        elif mode == "test":
            examples = processor.get_test_examples(args.with_relation, args.data_dir)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            # Pad on the left for xlnet
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    all_nodes_index_mask = []
    all_adj_metric = [] 
    all_node_mask = [] 
    all_sen2node = []
    all_sen_mask = []
    all_sen_length = []
    all_nsp_score = []
    all_nodes_ent_emb = []
    no_ent_emb, all_ent = 0, 0

    for f in features:
        nodes_mask, node_num = generate_shaped_nodes_mask(
            f.nodes_index, args.max_seq_length, args.max_nodes_num
        )
        nmask = np.zeros(args.max_nodes_num)
        nmask[:node_num] = 1
        all_node_mask.append(nmask) 

        adj_metric = generate_shaped_edge_mask(
            f.adj_metric, node_num, args.max_nodes_num, args.with_relation
        )
        all_nodes_index_mask.append(nodes_mask) 
        all_adj_metric.append(adj_metric)

        sen2node_mask = np.zeros(shape=(args.max_sentences, args.max_nodes_num))

        sen_mask = np.zeros(args.max_sentences - 1)
        sen_mask[: len(f.sen2node) - 1] = 1
        all_sen_mask.append(sen_mask)
        all_sen_length.append(
            len(f.sen2node)
            if len(f.sen2node) <= args.max_sentences
            else args.max_sentences
        )

        for idx in range(len(f.sen2node)):
            if idx >= args.max_sentences:
                break
            all_sennodes = f.sen2node[idx]
            for sennode in all_sennodes:
                if sennode < args.max_nodes_num:
                    sen2node_mask[idx, sennode] = 1
        all_sen2node.append(sen2node_mask)

    all_nodes_index_mask = torch.tensor(all_nodes_index_mask, dtype=torch.float)
    all_node_mask = torch.tensor(all_node_mask, dtype=torch.int)
    all_adj_metric = torch.tensor(all_adj_metric, dtype=torch.float)
    all_sen2node_mask = torch.tensor(all_sen2node, dtype=torch.float)
    all_sen_mask = torch.tensor(all_sen_mask, dtype=torch.float)
    all_sen_length = torch.tensor(all_sen_length, dtype=torch.long)

    batch_id = torch.tensor(list(range(0, len(all_labels))))
    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels,
        all_nodes_index_mask,
        all_adj_metric,
        all_node_mask,
        all_sen2node_mask,
        all_sen_mask,
        all_sen_length,
        batch_id,
    )
    return dataset


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    default=os.path.join(os.getcwd(), "data"),
    type=str,
    help="The input data dir.",
)
parser.add_argument(
    "--model_type",
    default="roberta",
    type=str,
    help="Base model for CoCo",
)
parser.add_argument(
    "--model_name_or_path",
    default="roberta-base",
    type=str,
    help="Base model for CoCo with size",
)
parser.add_argument(
    "--task_name",
    default="deepfake",
    type=str,
)
parser.add_argument(
    "--output_dir",
    default=os.path.join(os.getcwd(), "gpt2_500_test"),
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
)

parser.add_argument(
    "--config_name",
    default="",
    type=str,
    help="Pretrained config name or path if not the same as model_name",
)
parser.add_argument(
    "--train_file", default="p\=0.96.jsonl", type=str, help="training file"
)
parser.add_argument(
    "--dev_file", default="p\=0.96.jsonl", type=str, help="training file"
)
parser.add_argument(
    "--test_file", default="p\=0.96.jsonl", type=str, help="training file"
)
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--max_seq_length",
    default=512,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--do_train", 
    default=True, 
    help="Whether to run training.")
parser.add_argument(
    "--do_eval", 
    default=True, 
    help="Whether to run eval on the dev set."
)
parser.add_argument(
    "--do_test", 
    default=True, 
    help="Whether to run test on the dev set."
)
parser.add_argument(
    "--evaluate_during_training",
    action="store_true",
    help="Run evaluation during training at each logging step.",
)
parser.add_argument(
    "--do_lower_case",
    action="store_true",
    help="Set this flag if you are using an uncased model.",
)
parser.add_argument(
    "--per_gpu_train_batch_size",
    default=16,
    type=int,
    help="Batch size per GPU/CPU for training.",
)
parser.add_argument(
    "--per_gpu_eval_batch_size",
    default=16,
    type=int,
    help="Batch size per GPU/CPU for evaluation.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--learning_rate",
    default=1e-5,
    type=float,
    help="The initial learning rate for Adam.",
)
parser.add_argument(
    "--weight_decay", 
    default=0.01, 
    type=float, 
    help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_epsilon", 
    default=1e-8, 
    type=float, 
    help="Epsilon for Adam optimizer."
)
parser.add_argument(
    "--max_grad_norm", 
    default=1.0, 
    type=float, 
    help="Max gradient norm."
)
parser.add_argument(
    "--num_train_epochs",
    default=15,
    type=float,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument(
    "--warmup_steps", 
    default=0, 
    type=int, 
    help="Linear warmup over warmup_steps."
)
parser.add_argument(
    "--logging_steps", 
    type=int, 
    default=125, 
    help="Interval certain steps to log."
)
parser.add_argument(
    "--save_steps", 
    type=int, 
    default=500, 
    help="Interval certain steps to save checkpoint."
)
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument(
    "--no_cuda", 
    action="store_true", 
    help="Avoid using CUDA when available"
)
parser.add_argument(
    "--overwrite_output_dir",
    type=bool,
    default=True,
    help="Overwrite the content of the output directory",
)
parser.add_argument(
    "--overwrite_cache",
    default=True,
    help="Overwrite the cached training and evaluation sets",
)
parser.add_argument(
    "--seed", 
    type=int, 
    default=0, 
    help="Random seed."
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument(
    "--local_rank", 
    type=int, 
    default=-1, 
    help="For distributed training: local_rank"
)
parser.add_argument(
    "--server_ip", 
    type=str, 
    default="", 
    help="For distant debugging."
)
parser.add_argument(
    "--server_port", 
    type=str, 
    default="", 
    help="For distant debugging."
)
parser.add_argument(
    "--max_nodes_num", 
    type=int, 
    default=150, 
    help="Maximum of number of nodes when input."
)
parser.add_argument(
    "--max_sentences", 
    type=int, 
    default=45, 
    help="Maximum of number of sentences when input."
)
parser.add_argument(
    "--max_sen_replen",
    type=int,
    default=128,
    help="Maximum of length of sentences representation (after relu).",
)
parser.add_argument(
    "--attention_maxscore",
    type=int,
    default=16,
    help="Weight of the max similarity score inside self-attention.",
)
parser.add_argument(
    "--loss_type",
    default="rfcl",
    type=str,
    help="Loss Type, include: normal, scl, mbcl, rfcl. rfcl is the complete version of CoCo, normal is the baseline.",
)
parser.add_argument(
    "--gcn_layer",
    default=2,
    type=int,
    help="Number of layers of GAT, recommand 2.",
)
parser.add_argument(
    "--dataset_name",
    default="gpt3.5_mixed_500",
    type=str,
    help="Name of the dataset, if blank will use Grover dataset",
)
parser.add_argument(
    "--do_ray",
    default=False,
    type=bool,
    help="Searching hyperparameter by Ray Tune or not",
)
parser.add_argument(
    "--with_relation",
    default=2,
    type=int,
    help="number of relation in Relation-GCN, >=2 for multi-relation, and =0 for the vanilla GCN.",
)
parser.add_argument(
    "--wandb_note",
    default="CoCo_rf",
    type=str,
    help="To describe the name of Wandb record.",
)

args = parser.parse_args()


def get_train_idx_by_label(dataset):
    train_idx_by_label = {}
    for i in range(2):
        train_idx_by_label[i] = [
            idx for idx in range(len(dataset)) if int(dataset[idx][3]) == i
        ]
    return train_idx_by_label


def run(conf, data_dir=None):
    args.seed = conf["seed"]
    args.data_dir = data_dir

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    print(device)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args)

    # Login wandb account (you can delete this section if you do not need)
    wandb_api_key = "your/wandb/key"
    os.system("wandb login {}".format(wandb_api_key))

    init_args = {}
    if "MLFLOW_EXPERIMENT_ID" in os.environ:
        init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "Machine-Generated Text Detection"),
        name="CoCo_{}_s{}_{}".format(args.loss_type, args.seed, args.wandb_note),
        entity=os.getenv("WANDB_ENTITY", "your/account/name"),
        reinit=True,
        **init_args,
    )
    wandb.config.update(args, allow_val_change=True)
    wandb.define_metric("train/loss")
    wandb.define_metric("eval/accuracy")
    wandb.define_metric("eval/f1")
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        task_specific_params={
            "gcn_layer": args.gcn_layer,
            "max_nodes_num": args.max_nodes_num,
            "max_sentences": args.max_sentences,
            "max_sen_replen": args.max_sen_replen,
            "attention_maxscore": args.attention_maxscore,
            "relation_num": args.with_relation,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    train_dataset = load_and_cache_examples(
        args,
        args.task_name,
        tokenizer,
        evaluate=False,
        mode="train",
        dataset_name=args.dataset_name,
        rel=("relation" if args.with_relation > 0 else ""),
    )

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.loss_type == "scl":
        model = RobertaForGraphBasedSequenceClassification_CL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
    elif args.loss_type == "mbcl":
        model = None
        memory_dataset = load_and_cache_examples(
            args,
            args.task_name,
            tokenizer,
            evaluate=False,
            mode="train",
            dataset_name=args.dataset_name,
            rel=("relation" if args.with_relation > 0 else ""),
        )
        mb_sampler = SequentialSampler(memory_dataset)
        train_idx_by_label = get_train_idx_by_label(memory_dataset)
        mb_dataloader = DataLoader(memory_dataset, sampler=mb_sampler)
        encoder_k = EncoderForMBCL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            mb_dataloader=mb_dataloader,
            train_idx_by_label=train_idx_by_label,
        )
        encoder_q = EncoderForMBCL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            mb_dataloader=mb_dataloader,
            train_idx_by_label=train_idx_by_label,
        )
        encoder_q.to(args.device)
        encoder_k.to(args.device)
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False
    elif args.loss_type == "rfcl":
        model = RobertaForGraphBasedSequenceClassification_RFCL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
    elif args.loss_type =='normal':  # Baseline
        model = RobertaForGraphBasedSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
    else:
        raise NotImplementedError

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    # Begin training
    if args.loss_type != "mbcl":
        if args.do_train:
            global_step, tr_loss, res, output_dir = train(
                args, train_dataset, model, tokenizer
            )
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        final_output = None
        if output_dir is not None:
            final_output = output_dir
        # Saving best practice: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.do_train and (
            args.local_rank == -1 or torch.distributed.get_rank() == 0
        ):
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = RobertaForGraphBasedSequenceClassification.from_pretrained(
        #     args.output_dir)
        # tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        # model.to(args.device)
    else:
        if args.do_train:
            global_step, tr_loss, res, output_dir = mb_train(
                args, train_dataset, encoder_q, encoder_k, mb_dataloader, tokenizer
            )
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        final_output = None
        if output_dir is not None:
            final_output = output_dir
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.do_train and (
            args.local_rank == -1 or torch.distributed.get_rank() == 0
        ):
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation (post hoc)
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split(
                "/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(args.device)
            result = evaluate(args, model, tokenizer,
                              prefix=prefix, mode='dev')
            result = dict((k + "_{}".format(global_step), v)
                          for k, v in result.items())
            results.update(result)

    # Test
    if args.do_test and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        best_model_file = final_output
        logger.info("Evaluate the following checkpoints: %s", best_model_file) # Load best checkpoints.
        model = AutoModelForSequenceClassification.from_pretrained(best_model_file).to(args.device)
        results = evaluate(args, model, tokenizer, prefix='', mode='test')
        if args.do_ray:
            tune.report(test_accuracy = results["acc"],
                        test_f1 = results["f1"],
                        )
    return res


def main():
    data_dir = os.path.abspath("path/to/datset_with_graph")
    if args.do_ray == True:
        import ray
        ray.init()
        config = {
            "seed": tune.choice([10, 11, 12, 13, 14, 15]),
        }  # Can list any random seed here
        scheduler = ASHAScheduler(metric="accuracy", mode="max")
        reporter = CLIReporter(
            metric_columns=[
                "accuracy",
                "max_acc_f1",
                "f1",
                "max_f1_acc",
                "test_accuracy",
                "test_f1",
            ]
        )
        result = tune.run(
            partial(run, data_dir=data_dir),
            resources_per_trial={"cpu": 1, "gpu": 1},
            config=config,
            num_samples=8,
            scheduler=scheduler,
            progress_reporter=reporter,
        )
        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print(
            "Best trial final validation accuracy: {}".format(
                best_trial.last_result["accuracy"]
            )
        )
    else:
        for seed in [10, 11, 12, 13, 14, 15]:  # Can list any random seeds here
            config = {
                "seed": seed,
            }
            run(config, data_dir)


if __name__ == "__main__":
    main()
