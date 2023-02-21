import json
import time
import shutil
import random

from torch._C import Value
import jsonpickle
import os
from typing import List, Dict, Optional
from statistics import mean

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

# YE: important
import torch.distributed

from tqdm import trange, tqdm
from numpy.lib.function_base import average
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, \
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPT2ForSequenceClassification, GPTJConfig, GPTJForCausalLM, \
    GPTJForSequenceClassification
from transformers import __version__ as transformers_version
from transformers import glue_compute_metrics
from transformers import glue_output_modes

import log
import preprocessor
from utils import InputFeatures, DictDataset, set_seed
from pstar_model import GPT2PrefixForSequenceClassification, GPTJPrefixForSequenceClassification

logger = log.get_logger('root')

CONFIG_NAME = 'wrapper_config.json'
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"
MLM_WRAPPER = "mlm"
PREFIX_SEQUENCE_CLASSIFIER_WRAPPER = "prefix_sequence_classifier"

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER, PREFIX_SEQUENCE_CLASSIFIER_WRAPPER]


PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: preprocessor.SequenceClassifierPreprocessor,
    MLM_WRAPPER: preprocessor.MLMPreprocessor,
    PREFIX_SEQUENCE_CLASSIFIER_WRAPPER: preprocessor.SequenceClassifierPreprocessor,
}

MODEL_CLASSES = {
    # 'bert': {
    #     'config': BertConfig,
    #     'tokenizer': BertTokenizer,
    #     SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification,
    #     MLM_WRAPPER: BertForMaskedLM
    # },
    # 'roberta': {
    #     'config': RobertaConfig,
    #     'tokenizer': RobertaTokenizer,
    #     SEQUENCE_CLASSIFIER_WRAPPER: RobertaForSequenceClassification,
    #     MLM_WRAPPER: RobertaForMaskedLM
    # },
    # 'xlm-roberta': {
    #     'config': XLMRobertaConfig,
    #     'tokenizer': XLMRobertaTokenizer,
    #     SEQUENCE_CLASSIFIER_WRAPPER: XLMRobertaForSequenceClassification,
    #     MLM_WRAPPER: XLMRobertaForMaskedLM
    # },
    # 'xlnet': {
    #     'config': XLNetConfig,
    #     'tokenizer': XLNetTokenizer,
    #     SEQUENCE_CLASSIFIER_WRAPPER: XLNetForSequenceClassification,
    # },
    # 'albert': {
    #     'config': AlbertConfig,
    #     'tokenizer': AlbertTokenizer,
    #     SEQUENCE_CLASSIFIER_WRAPPER: AlbertForSequenceClassification,
    #     MLM_WRAPPER: AlbertForMaskedLM
    # },
    'gpt2': {
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: GPT2ForSequenceClassification,
        MLM_WRAPPER: GPT2LMHeadModel,
        PREFIX_SEQUENCE_CLASSIFIER_WRAPPER: GPT2PrefixForSequenceClassification
    },

    'EleutherAI/gpt-j-6B': {
        'config': GPTJConfig,
        'tokenizer': GPT2Tokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: GPTJForSequenceClassification,
        MLM_WRAPPER: GPTJForCausalLM,
        PREFIX_SEQUENCE_CLASSIFIER_WRAPPER: GPTJPrefixForSequenceClassification
    },
}

EVALUATION_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_eval_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_eval_step,
    PREFIX_SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_eval_step,
}

TRAIN_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_train_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_train_step,
    PREFIX_SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_train_step,
}


class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(self, model_type: str, model_name_or_path: str, wrapper_type: str, task_name: str, max_seq_length: int,
                 label_list: List[str], pattern_id: int = 0, verbalizer_file: str = None, cache_dir: str = None, 
                 n_gpu: int = 1, model_parallel: bool = False, local_rank: int = -1, fp16: bool = False, fp16_opt_level: str = "01", 
                 save_steps: int = 0, evaluate_during_training: bool = False, hidden_dropout_prob: float = 0, pre_seq_len: int = 0, 
                 prefix_projection: bool = False, prefix_hidden_size: int = 512):
        """
        Create a new config.

        :param model_type: the model type (e.g., 'bert', 'roberta', 'albert')
        :param model_name_or_path: the model name (e.g., 'roberta-large') or path to a pretrained model
        :param wrapper_type: the wrapper type (one of 'mlm' and 'sequence_classifier')
        :param task_name: the task to solve
        :param max_seq_length: the maximum number of tokens in a sequence
        :param label_list: the list of labels for the task
        :param pattern_id: the id of the pattern to use
        :param verbalizer_file: optional path to a verbalizer file
        :param cache_dir: optional path to a cache dir
        """
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.wrapper_type = wrapper_type
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.verbalizer_file = verbalizer_file
        self.cache_dir = cache_dir
        self.n_gpu = n_gpu
        self.model_parallel = model_parallel
        self.local_rank = local_rank
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.save_steps = save_steps
        self.evaluate_during_training = evaluate_during_training
        self.optimizer = None
        self.scheduler = None
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.prefix_hidden_size = prefix_hidden_size


class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        """Create a new wrapper from the given config."""
        self.config = config
        config_class = MODEL_CLASSES[self.config.model_type]['config']
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]

        model_config = config_class.from_pretrained(
            config.model_name_or_path, num_labels=len(config.label_list), finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None, use_cache=False)
        if self.config.wrapper_type == "prefix_sequence_classifier":
            model_config.hidden_dropout_prob=config.hidden_dropout_prob
            model_config.pre_seq_len = config.pre_seq_len
            model_config.prefix_projection=config.prefix_projection 
            model_config.prefix_hidden_size=config.prefix_hidden_size

        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)  # type: PreTrainedTokenizer

        self.model = model_class.from_pretrained(config.model_name_or_path, config=model_config,
                                                 cache_dir=config.cache_dir if config.cache_dir else None)
        
        if self.config.model_type == 'EleutherAI/gpt-j-6B' or self.config.model_type == 'gpt2':
            # YE: Synchronizing threads before loading vocabulary
            if self.config.local_rank == 0:
                torch.distributed.barrier()

            #### modified to add extra token
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.config.pad_token_id = model_config.vocab_size
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer.mask_token = self.tokenizer.eos_token

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](self, self.config.task_name, self.config.pattern_id,
                                                                    self.config.verbalizer_file)
        # self.task_helper = TASK_HELPERS[self.config.task_name](self) if self.config.task_name in TASK_HELPERS else None
        self.task_helper = None

    @classmethod
    def from_pretrained(cls, path: str, load_and_save_best: bool = False) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        if load_and_save_best:
            wrapper.model = torch.load(path + "/BestModel.pt")
            model_to_save = wrapper.model.module if hasattr(wrapper.model, "module") else wrapper.model
            model_to_save.save_pretrained(path)

        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](
            wrapper, wrapper.config.task_name, wrapper.config.pattern_id, wrapper.config.verbalizer_file)
        # wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](wrapper) \
        #     if wrapper.config.task_name in TASK_HELPERS else None
        wrapper.task_helper = None
        return wrapper

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)
        logger.info("Saving model checkpoint to %s", path)
        if self.optimizer is None or self.scheduler is None:
            raise ValueError("Optimizer and Scheduler should not be None.")
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", path)

    # TODO: NEED TO MODIFY CONFIG
    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())


    # YE: Method to automatically split layers over available gpus
    def generate_device_map(self, n_gpu, n_layers):
        #n_gpu = torch.cuda.device_count() // Not needed anymore
        device_map = {i : [] for i in range(n_gpu)}
        shard = int((n_layers - 3) / (n_gpu - 1))
        remain = (n_layers - 3) - (shard * (n_gpu - 1))
        count = 0
        # First shard has additional embeddings layers, so we process it aside
        for _ in range(3):
            device_map[0].append(count)
            count += 1
    
        # We do now the other splits
        for i in range(1, n_gpu):
            for _ in range(shard):
                device_map[i].append(count)
                count += 1
            if remain > 0:
                device_map[i].append(count)
                count += 1
                remain -= 1
        # YE: maybe some layers left, in case n_layers is not multiple of n_gpu
        while count < n_layers:
            device_map[n_gpu -1].append(count)
            count += 1
        return device_map
        


    def train(self, task_train_data: List[InputExample], device, per_gpu_train_batch_size: int = 8, n_gpu: int = 1,
              num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
              learning_rate: float = 5e-5, adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
              logging_steps: int = 50, 
              alpha: float = 0.8, temperature: float = 1,
              max_steps=-1, 
              model_name_or_path: str = None, fp16: bool = False, fp16_opt_level: str = "01", save_steps: int = 0, 
              evaluate_during_training: bool = False, 
              eval_data: List[InputExample] = None,  per_gpu_eval_batch_size: int = 8,
              priming: bool = False, priming_num: int = 0, priming_description: str = '', 
              output_dir: str = None, seed: int = 42, **_):
        """
        Train the underlying language model.

        :param task_train_data: the training examples to use
        :param device: the training device (cpu/gpu)
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for knowledge distillation
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(task_train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        #### modified ##### 
        #### load from pretrained ####
        if os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.
            path.join(model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path, "optimizer.pt")))
            self.scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path, "scheduler.pt")))

        # Necessary (before fp16 init)
        is_model_sharded = self.config.model_parallel
        self.model.model_parallel = is_model_sharded

        #### modified fp16 ####
        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=fp16_opt_level)


        # YE: multi-gpu training. Differentiate between data parallelism and model sharding
        if n_gpu > 1 and not is_model_sharded:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(n_gpu)])

            # Distributed training (should be after apex fp16 initialization)
            local_rank = self.config.local_rank
            if local_rank != -1:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        elif is_model_sharded:
            # YE: for GPT-J, it is necessary to shard the model
            # step 1: calculate the device map
            device_map = self.generate_device_map(n_gpu, self.model.config.n_layer)
            # step 2: parallelize the model
            self.model.device_map = device_map
            self.model.transformer.parallelize(device_map) 
            logger.info("***** device_map TRAIN *****")
            logger.info(device_map)


        #### modified add logger ####
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * gradient_accumulation_steps,
        )
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        
        set_seed(seed)
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        max_met = -1
        max_met_step = -1
        BEST_ACCURACY = 0
        self.model.zero_grad()
        epoch_time_records = []

        train_iterator = trange(int(num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_start_time = time.time()
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = {k: t.to(device) for k, t in batch.items()}


                # if loss is None:
                loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch)
                # train_step_inputs = {
                #     'alpha': alpha, 'temperature': temperature
                # }
                # loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch, **train_step_inputs)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = self.scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        print(json.dumps({**logs, **{'step': global_step}}))

                    #### modified save model checkpoint ####
                    if save_steps > 0 and global_step % save_steps == 0:
                        tasks_to_metrics = {"sst-2": "acc", "rte": "acc", "qnli": "acc", "qqp": "acc", 
                                            "mrpc": "acc", "wnli": "acc", "mnli": "mnli/acc", "cola": "mcc", "sts-b": "corr"}
                        # Save model checkpoint
                        logs = {}
                        results = self.eval(eval_data, device, per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                            n_gpu=n_gpu, priming=priming, priming_num=priming_num, priming_description=priming_description,
                                            eval_output_dir=output_dir)
                        key = tasks_to_metrics[self.config.task_name]
                        value = results[key]
                        eval_key = "eval_{}".format(key)

                        if evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        
                            if BEST_ACCURACY < value:
                                BEST_ACCURACY = value
                                torch.save(self.model.state_dict(), output_dir + "/BestModelDict.pt")
                                print("Best Model Saved!")

                            logs[eval_key] = value

                        current_met = logs[eval_key]
                        if max_met <= logs[eval_key]:
                            max_met = logs[eval_key]
                            max_met_step = global_step

                        logs["eval_met_max"] = max_met
                        logs["eval_met_max_step"] = max_met_step

                        print(json.dumps({**logs, **{"step": global_step}}))

                        if current_met == max_met:
                            torch.save(self.model, output_dir + "/BestModel.pt")
                            print("Best Model Saved!")
                            best_folder_output_dir = os.path.join(output_dir, "best")
                            shutil.rmtree(best_folder_output_dir, ignore_errors=True)
                            if not os.path.exists(best_folder_output_dir):
                                os.makedirs(best_folder_output_dir)
                            self.save(path=best_folder_output_dir)

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < max_steps < global_step:
                train_iterator.close()
                break

            epoch_time = time.time() - epoch_start_time
            logger.info("==== one epoch training time %s ===="%(epoch_time))
            epoch_time_records.append(epoch_time)
        
        logger.info("==== one epoch average training time %s ===="%(mean(epoch_time_records)))

        return global_step, (tr_loss / global_step if global_step > 0 else -1)

    def eval(self, eval_data: List[InputExample], device, per_gpu_eval_batch_size: int = 8, n_gpu: int = 1,
             priming: bool = False, priming_num: int = 0, priming_description: str = '', 
             eval_output_dir: str = None) -> Dict:
        """
        Evaluate the underlying language model.

        :param eval_data: the evaluation examples to use
        :param device: the evaluation device (cpu/gpu)
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param priming: whether to use priming
        :return: a dictionary of numpy arrays containing the indices, logits, labels, and (optional) question_ids for
                 each evaluation example.
        """

        eval_dataset = self._generate_dataset(eval_data, priming=priming, priming_num=priming_num, priming_description=priming_description)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)


        is_model_sharded = self.config.model_parallel
        self.model.model_parallel = is_model_sharded
        
        logger.info("################  Running eval()")

        # YE: multi-gpu training. Differentiate between data parallelism and model sharding
        if n_gpu > 1 and not is_model_sharded:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(n_gpu)])

            # Distributed training (should be after apex fp16 initialization)
            local_rank = self.config.local_rank
            if local_rank != -1:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        elif is_model_sharded:
            # YE: for GPT-J, it is necessary to shard the model
            # step 1: calculate the device map
            device_map = self.generate_device_map(n_gpu, self.model.config.n_layer)
            # step 2: parallelize the model
            self.model.device_map = device_map
            self.model.transformer.parallelize(device_map) 
            logger.info("***** device_map TRAIN *****")
            logger.info(device_map)

        
        logger.info("***** Running evaluation {} *****")
        logger.info("  n_gpu = %d", n_gpu)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            with torch.no_grad():
                logits = EVALUATION_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)

        results = {
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }

        if self.config.task_name is not None and eval_output_dir is not None:
            output_mode = glue_output_modes[self.config.task_name]
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(preds)
            result = glue_compute_metrics(self.config.task_name, preds, out_label_ids)
            results.update(result)

            output_eval_file = os.path.join(eval_output_dir,"eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results  *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return results


    def _generate_dataset(self, data: List[InputExample], labelled: bool = True, priming: bool = False, priming_num: int = 0, priming_description: str = ""):
        features = self._convert_examples_to_features(data, labelled=labelled, priming=priming, priming_num=priming_num, priming_description=priming_description)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long)
        }
        return DictDataset(**feature_dict)


    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True,
                                      priming: bool = False, priming_num: int = 0, priming_description: str = "") -> List[InputFeatures]:
        features = []
        if priming:
            priming_example_used_nums = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(example, labelled=labelled, priming=priming, priming_num = priming_num, priming_description = priming_description)
            if priming:
                priming_example_used_nums.append(input_features.meta["priming_used_example_num"])
            if self.task_helper:
                self.task_helper.add_special_input_features(example, input_features)
            features.append(input_features)
            """
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                # logger.info(input_features.pretty_print(self.tokenizer))
                logger.info(input_features.input_ids)
                logger.info(input_features.attention_mask)
                logger.info(input_features.label)
            """
        if priming:
            if len(priming_description) > 0: logger.info(f"Provided task description: \"{priming_description}\"")
            logger.info(f"Priming number is {priming_num}; Average priming used examples is {average(priming_example_used_nums)}")
        return features


    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        return inputs


    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor],
                       alpha: float = 0, **_) -> torch.Tensor:
        """Perform a MLM training step."""

        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']

        outputs = self.model(**inputs)
        #prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs["logits"])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))
        return loss


    def sequence_classifier_train_step(self, batch: Dict[str, torch.Tensor], 
                                       temperature: float = 1, **_) -> torch.Tensor:
        """Perform a sequence classifier training step."""
        inputs = self.generate_default_inputs(batch)
        inputs['labels'] = batch['labels']
        outputs = self.model(**inputs)
        return outputs[0]

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        # print("in mlm_eval_step", flush=True)
        # print("batch", batch, flush=True)
        #return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs["logits"])

    def sequence_classifier_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a sequence classifier evaluation step."""
        inputs = self.generate_default_inputs(batch)
        return self.model(**inputs)[0]

