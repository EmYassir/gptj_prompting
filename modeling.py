import os
import json
import torch
import statistics
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers.data.metrics import simple_accuracy

import log
from utils import InputExample, exact_match, set_seed, save_logits, save_predictions
from wrapper import TransformerModelWrapper, WrapperConfig

logger = log.get_logger('root')


class PetConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    """Configuration for training a model."""

    def __init__(self, device: str = None, per_gpu_train_batch_size: int = 8, 
                 n_gpu: int = 1, model_parallel: bool = False, num_train_epochs: int = 3, max_steps: int = -1, gradient_accumulation_steps: int = 1,
                 weight_decay: float = 0.0, learning_rate: float = 5e-5, adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0, max_grad_norm: float = 1, 
                 alpha: float = 0.9999, temperature: float = 1):
        """
        Create a new training config.

        :param device: the device to use ('cpu' or 'gpu')
        :param per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train for
        :param max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
        :param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the maximum learning rate to use
        :param adam_epsilon: the epsilon value for Adam
        :param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        :param max_grad_norm: the maximum norm for the gradient
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for distillation
        """
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.n_gpu = n_gpu
        self.model_parallel = model_parallel
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.temperature = temperature


class EvalConfig(PetConfig):
    """Configuration for evaluating a model."""

    def __init__(self, device: str = None, n_gpu: int = 1, model_parallel: bool = False, local_rank: int = -1, fp16: bool = False, fp16_opt_level: str = "01", per_gpu_eval_batch_size: int = 8,
                 metrics: List[str] = None, decoding_strategy: str = 'default', priming: bool = False, priming_num: int = 0, priming_description: str = ''):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        :param priming: whether to use priming
        """
        self.device = device
        self.n_gpu = n_gpu
        self.model_parallel = model_parallel
        self.local_rank = local_rank
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics
        self.priming = priming
        self.priming_num = priming_num
        self.priming_description = priming_description

def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config)
    return model


def train_pet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
              ensemble_eval_config: EvalConfig, pattern_ids: List[int], output_dir: str, ensemble_repetitions: int = 3,
              train_data: List[InputExample] = None,
              eval_data: List[InputExample] = None, do_train: bool = True,
              do_eval: bool = True, no_distillation: bool = False, seed: int = 42):
    """
    Train and evaluate a new PET model for a given task.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ensemble_repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param reduction: the reduction strategy for merging predictions, either 'mean' or 'wmean'
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param no_distillation: if true, no distillation is performed
    :param seed: the random seed to use
    """

    # Step 1: Train an ensemble of models corresponding to individual patterns
    train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids, output_dir,
                       repetitions=ensemble_repetitions, train_data=train_data,
                       eval_data=eval_data, do_train=do_train, do_eval=do_eval,
                       seed=seed)

    return


def train_classifier(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig, output_dir: str,
                     repetitions: int = 1, train_data: List[InputExample] = None,
                     eval_data: List[InputExample] = None,
                     do_train: bool = True, do_eval: bool = True, seed: int = 42):
    """
    Train and evaluate a sequence classification model.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    train_pet_ensemble(model_config, train_config, eval_config, pattern_ids=[0], output_dir=output_dir,
                       repetitions=repetitions,
                       train_data=train_data, eval_data=eval_data, do_train=do_train,
                       do_eval=do_eval, seed=seed)




def train_pet_ensemble(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig,
                       pattern_ids: List[int], output_dir: str, repetitions: int = 3,
                       train_data: List[InputExample] = None, 
                       eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True,
                       seed: int = 42):
    """
    Train and evaluate an ensemble of PET models without knowledge distillation.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ipet_data_dir: optional directory containing additional training data for iPET
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param save_unlabeled_logits: whether logits for unlabeled examples should be saved in a file ``logits.txt``. This
           is required for both iPET and knowledge distillation.
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    for pattern_id in pattern_ids:
        for iteration in range(repetitions):

            model_config.pattern_id = pattern_id
            results_dict = {}

            pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, iteration)

            if os.path.exists(pattern_iter_output_dir):
                logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
                continue

            if not os.path.exists(pattern_iter_output_dir):
                os.makedirs(pattern_iter_output_dir)

            # YE: Synchronizing threads before loading pretrained model and tokenizer
            if model_config.local_rank not in [-1, 0]:
                torch.distributed.barrier()
            
            wrapper = init_model(model_config)

            # Training
            if do_train:
                results_dict.update(train_single_model(wrapper, train_data, eval_data, train_config, eval_config, 
                                                       pattern_iter_output_dir, seed=seed))

                with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                logger.info("Saving trained model at {}...".format(pattern_iter_output_dir))
                wrapper.save(pattern_iter_output_dir)
                train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
                eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
                logger.info("Saving complete")

                wrapper.model = None
                wrapper = None

                if not do_eval:
                    torch.cuda.empty_cache()

            # Evaluation
            if do_eval:
                logger.info("Starting evaluation...")
                if not wrapper:
                    wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir, load_and_save_best=True)

                eval_result = evaluate(wrapper, eval_data, eval_config, priming_data=train_data, eval_output_dir=pattern_iter_output_dir)

                save_predictions(os.path.join(pattern_iter_output_dir, 'predictions.jsonl'), wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

                scores = eval_result['scores']
                logger.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logger.info(scores)

                results_dict['test_set_after_training'] = scores
                with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                    json.dump(results_dict, fh)

                for metric, value in scores.items():
                    results[metric][pattern_id].append(value)

                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

    if do_eval:
        logger.info("=== OVERALL RESULTS ===")
        _write_results(os.path.join(output_dir, 'result_test.txt'), results)
    else:
        logger.info("=== ENSEMBLE TRAINING COMPLETE ===")



def train_single_model(model: TransformerModelWrapper, train_data: List[InputExample], eval_data: List[InputExample],  config: TrainConfig,
                       eval_config: EvalConfig = None, output_dir: str = None, seed: int = 42, return_train_set_results: bool = True):
    """
    Train a single model.

    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :param return_train_set_results: whether results on the train set before and after training should be computed and
           returned
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
    # ipet_train_data = []

    results_dict = {}

    model.model.to(device)

    # if train_data and return_train_set_results:
    #     results_dict['train_set_before_training'] = evaluate(model, train_data, eval_config)['scores']['acc']

    all_train_data = train_data

    if not all_train_data and not config.use_logits:
        logger.warning('Training method was called without training examples')
    else:
        global_step, tr_loss = model.train(
            all_train_data, device,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            alpha=config.alpha,
            temperature=config.temperature,
            model_name_or_path=model.config.model_name_or_path, 
            fp16=model.config.fp16, 
            fp16_opt_level=model.config.fp16_opt_level, 
            save_steps=model.config.save_steps, 
            evaluate_during_training=model.config.evaluate_during_training, 
            eval_data=eval_data,  
            per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
            priming=eval_config.priming, 
            priming_num=eval_config.priming_num, 
            priming_description=eval_config.priming_description, 
            output_dir=output_dir,
            seed=seed
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    # if train_data and return_train_set_results:
    #     results_dict['train_set_after_training'] = evaluate(model, eval_data, eval_config, eval_output_dir=output_dir)['scores']['acc']

    return results_dict


def evaluate(model: TransformerModelWrapper, eval_data: List[InputExample], config: EvalConfig,
             eval_output_dir: str = None, priming_data: List[InputExample] = None) -> Dict:
    """
    Evaluate a model.

    :param model: the model to evaluate
    :param eval_data: the examples for evaluation
    :param config: the evaluation config
    :param priming_data: an optional list of priming data to use
    :param priming_description: task description used in priming
    :return: a dictionary containing the model's logits, predictions and (if any metrics are given) scores
    """

    if config.priming:
        for example in eval_data:
            example.meta['priming_data'] = priming_data

    metrics = config.metrics if config.metrics else ['acc']
    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    model.model.to(device)
    results = model.eval(eval_data, device, per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu, priming=config.priming, priming_num=config.priming_num, priming_description=config.priming_description,
                         eval_output_dir=eval_output_dir)

    predictions = np.argmax(results['logits'], axis=1)
    scores = {}

    for metric in metrics:
        if metric == 'acc':
            scores[metric] = simple_accuracy(predictions, results['labels'])
        elif metric == 'f1':
            scores[metric] = f1_score(results['labels'], predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(results['labels'], predictions, average='macro')
        elif metric == 'em':
            scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
        elif metric == 'mcc':
            scores[metric] = matthews_corrcoef(results['labels'], predictions)
        else:
            raise ValueError(f"Metric '{metric}' not implemented")

    results['scores'] = scores
    results['predictions'] = predictions
    return results


def _write_results(path: str, results: Dict):
    with open(path, 'w') as fh:
        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')


