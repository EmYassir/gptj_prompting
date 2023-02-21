import argparse
import os
from typing import Tuple

import torch

from tasks import PROCESSORS, load_examples, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from utils import eq_div
from wrapper import WRAPPER_TYPES, MODEL_CLASSES, WrapperConfig
from modeling import TrainConfig, EvalConfig, train_pet, train_classifier
import log

logger = log.get_logger('root')

def load_pet_configs(args) -> Tuple[WrapperConfig, TrainConfig, EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                            wrapper_type=args.wrapper_type, task_name=args.task_name, label_list=args.label_list,
                            max_seq_length=args.pet_max_seq_length, verbalizer_file=args.verbalizer_file,
                            cache_dir=args.cache_dir, local_rank=args.local_rank, model_parallel = args.model_parallel, fp16=args.fp16, 
                            fp16_opt_level=args.fp16_opt_level, save_steps=args.save_steps, evaluate_during_training=args.evaluate_during_training)

    train_cfg = TrainConfig(device=args.device, per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
                            n_gpu=args.n_gpu, model_parallel = args.model_parallel,
                            num_train_epochs=args.pet_num_train_epochs, max_steps=args.pet_max_steps,
                            gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
                            weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                            adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                            max_grad_norm=args.max_grad_norm, alpha=args.alpha)

    eval_cfg = EvalConfig(device=args.device, n_gpu=args.n_gpu, model_parallel = args.model_parallel, local_rank = args.local_rank, fp16 = args.fp16, 
                        fp16_opt_level = args.fp16_opt_level,  per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size, metrics=args.metrics, priming=args.priming, 
                        priming_num = args.priming_num, priming_description=args.priming_description)

    return model_cfg, train_cfg, eval_cfg


def load_sequence_classifier_configs(args) -> Tuple[WrapperConfig, TrainConfig, EvalConfig]:
    """
    Load the model, training and evaluation configs for a regular sequence classifier from the given command line
    arguments. This classifier can either be used as a standalone model or as the final classifier for PET/iPET.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=args.wrapper_type, task_name=args.task_name,
                              label_list=args.label_list, max_seq_length=args.sc_max_seq_length,
                              verbalizer_file=args.verbalizer_file, cache_dir=args.cache_dir,
                              n_gpu=args.n_gpu, model_parallel = args.model_parallel,
                              fp16=args.fp16, fp16_opt_level=args.fp16_opt_level,
                              save_steps=args.save_steps, evaluate_during_training=args.evaluate_during_training,
                              hidden_dropout_prob=args.hidden_dropout_prob, pre_seq_len=args.pre_seq_len,
                              prefix_projection=args.prefix_projection, prefix_hidden_size=args.prefix_hidden_size)

    train_cfg = TrainConfig(device=args.device, per_gpu_train_batch_size=args.sc_per_gpu_train_batch_size,
                                n_gpu=args.n_gpu,
                                model_parallel = args.model_parallel,
                                num_train_epochs=args.sc_num_train_epochs, max_steps=args.sc_max_steps,
                                temperature=args.temperature,
                                gradient_accumulation_steps=args.sc_gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm)

    eval_cfg = EvalConfig(device=args.device, n_gpu=args.n_gpu, model_parallel = args.model_parallel, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.sc_per_gpu_eval_batch_size)

    return model_cfg, train_cfg, eval_cfg


def main():
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")

    # Required parameters
    parser.add_argument("--method", required=True, choices=['pet', 'sequence_classifier', "prefix_sequence_classifier"],
                        help="The training method to use. Either regular sequence classification, PET.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, 
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    # PET-specific optional parameters
    parser.add_argument("--wrapper_type", default="mlm", choices=WRAPPER_TYPES,
                        help="The wrapper type. Set this to 'mlm' for a masked language model like BERT or to 'plm' "
                             "for a permuted language model like XLNet (only for PET)")
    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling task (only for PET)")
    parser.add_argument("--temperature", default=2, type=float,
                        help="Temperature used for combining PVPs (only for PET)")
    parser.add_argument("--verbalizer_file", default=None,
                        help="The path to a file to override default verbalizers (only for PET)")
    parser.add_argument("--no_distillation", action='store_true',
                        help="If set to true, no distillation is performed (only for PET)")
    parser.add_argument("--pet_repetitions", default=3, type=int,
                        help="The number of times to repeat PET training and testing with different seeds.")
    parser.add_argument("--pet_max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--pet_per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--pet_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--pet_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--pet_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--pet_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # SequenceClassifier-specific optional parameters (also used for the final PET classifier)
    parser.add_argument("--sc_repetitions", default=1, type=int,
                        help="The number of times to repeat seq. classifier training and testing with different seeds.")
    parser.add_argument("--sc_max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization for sequence classification. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--sc_per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for sequence classifier training.")
    parser.add_argument("--sc_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for sequence classifier evaluation.")
    parser.add_argument('--sc_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass for "
                             "sequence classifier training.")
    parser.add_argument("--sc_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform for sequence classifier training.")
    parser.add_argument("--sc_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform for sequence classifier training. "
                             "Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    
    # YE: two useful parameters
    parser.add_argument("--model_parallel", action='store_true', help="Splits the model over available gpus")                 
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument('--priming', action='store_true',
                        help="Whether to use priming for evaluation")
    parser.add_argument('--priming_num', default=0, type=int,
                        help="The number of examples to use during priming; If sentences are too long, it will reduced to max number of examples that can fit into max_seq_length")
    parser.add_argument('--priming_description', default="", type=str,
                        help="Task description which will be used in priming if not blank.")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")

    parser.add_argument('--gpu', type=str, default="0")
    
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.")
    
    #### only for prefix tuning
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="prefix dropout probability")
    parser.add_argument("--pre_seq_len", type=int, default=10, help="prefix token length")
    parser.add_argument("--prefix_projection", action='store_true', help="Whether to use parameterization trick")
    parser.add_argument("--prefix_hidden_size", type=int, default=512, help="prefix layer hidden size")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # YE: Should be removed in our use-case scenario
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    logger.info("Parameters: {}".format(args))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # YE: setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    train_ex_per_label, test_ex_per_label = None, None
    train_ex, test_ex = args.train_examples, args.test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
        train_ex, test_ex = None, None

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET

    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)
    eval_data = load_examples(
        args.task_name, args.data_dir, eval_set, num_examples=test_ex, num_examples_per_label=test_ex_per_label)


    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)
    sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(args)

    if args.method == 'pet':
        train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg,
                      pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                      ensemble_repetitions=args.pet_repetitions,
                      train_data=train_data, 
                      eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval,
                      no_distillation=args.no_distillation, seed=args.seed)

    elif args.method == 'sequence_classifier' or args.method == 'prefix_sequence_classifier':
        train_classifier(sc_model_cfg, sc_train_cfg, sc_eval_cfg, output_dir=args.output_dir,
                             repetitions=args.sc_repetitions, train_data=train_data,
                             eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed)

    else:
        raise ValueError(f"Training method '{args.method}' not implemented")


if __name__ == "__main__":
    main()
