import argparse
import os
import log
import log
import random
import numpy as np
import torch


from transformers import glue_output_modes as output_modes



logger = log.get_logger("root")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--method", 
        required=True, 
        choices=['discrete_prompt', 'prefix_tuning', 'p-tuning', 'sequence_classifier'],
        help="The training method to use. Either regular sequence classification, discrete_prompt, prefix_tuning OR p-tuning."
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: ",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
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
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_rate", default=0, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--specific", type=str, default="", help="For specific cached file clarification.")
    parser.add_argument("--print_every_step", type=int, default=100, help="Print training loss every N steps")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
        and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or not args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    logger.info("Parameters: {}".format(args))

    # Set seed
    set_seed(args)

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
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    #pdb.set_trace()
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
       	args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2ForSequenceClassification.from_pretrained( #AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config
    )
    model.config.pad_token_id = config.vocab_size
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.resize_token_embeddings(len(tokenizer))

    #################### additional layer #####################
    #### init attn ####
    # logger.info("Reinitialize additional attention layer")
    # model.attn_layer.apply(init_normal_weights)

    #### init gru ####
    # logger.info("Reinitialize GRU layer")
    # model.gru.apply(init_normal_weights)
    # model.score.apply(init_normal_weights)
    
    #### freeze transformer ####
    # logger.info("Freeze model transformer parameters...")
    # freeze_model_transformer(model)

    #### freeze except last block ####
    # logger.info("Freeze model transformer parameters except the last block and the following layer normalization...")
    # freeze_transformer_except_last_block(model, config.n_layer)

    ########################################################### 

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    ########## calculate model size ##############
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %s, Trainable parameters: %s", total_params, trainable_params)
    ##############################################

    ##########test parameter##############
    # param_transf = []
    # param_other = []
    # for p in model.transformer.parameters():
    #     param_transf.append(p.clone().detach())
    # for p in model.attn_layer.parameters():
    #     param_other.append(p.clone().detach())
    # for p in model.score.parameters():
    #     param_other.append(p.clone().detach())

    # for p in model.state_dict().items():
    #     print(p[0], flush=True)
    # print("=================", flush=True)
    # for p in model.transformer.state_dict().items():
    #     param_transf.append(p)
    # for p in model.attn_layer.state_dict().items():
    #     param_other.append(p)
    # for p in model.score.state_dict().items():
    #     param_other.append(p)    
    ##################################

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model = torch.load(args.output_dir + "/BestModel.pt")
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
        model = GPT2ForSequenceClassification.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    result_test = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        #pdb.set_trace()
        tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = torch.load(args.output_dir + "/BestModel.pt")
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)


        
    ##########test parameter##############
    # param_transf2 = []
    # param_other2 = []
    # for p in model.transformer.parameters():
    #     param_transf2.append(p)
    # for p in model.attn_layer.parameters():
    #     param_other2.append(p)
    # for p in model.score.parameters():
    #     param_other2.append(p)
    # for p in model.transformer.state_dict().items():
    #     param_transf2.append(p)
    # for p in model.attn_layer.state_dict().items():
    #     param_other2.append(p)
    # for p in model.score.state_dict().items():
    #     param_other2.append(p)   

    # for p1, p2 in zip(param_transf, param_transf2):
    #     assert torch.equal(p1, p2)
    # for p1, p2 in zip(param_other, param_other2):
    #     assert not torch.equal(p1, p2)
    # print("assert is passed")

    # def compare_models(state_1, state_2):
    #     models_differ = 0
    #     for key_item_1, key_item_2 in zip(state_1, state_2):
    #         if torch.equal(key_item_1[1], key_item_2[1]):
    #             pass
    #         else:
    #             models_differ += 1
    #             if (key_item_1[0] == key_item_2[0]):
    #                 print('Mismtach found at', key_item_1[0])
    #             else:
    #                 raise Exception

    #     if models_differ == 0:
    #         print('Models match perfectly! :)')

    # print("compare transforemrs")
    # compare_models(param_transf, param_transf2)
    # compare_models2(param_other, param_other2)

    ##################################


    return results




if __name__ == "__main__":
    main()