import argparse

def common_args():
    parser = argparse.ArgumentParser()

    # task
    parser.add_argument("--train_file", type=str,
                        default="../data/nq-with-neg-train.txt")
    parser.add_argument("--predict_file", type=str,
                        default="../data/nq-with-neg-dev.txt")
    parser.add_argument("--num_workers", default=30, type=int)
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")


    parser.add_argument("--do_augment", action="store_true")
    parser.add_argument("--aug_save_path", type=str, default="")

    # model
    parser.add_argument("--model_name",
                        default="bert-base-uncased", type=str)
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--decoder_checkpoint", type=str,
                        default='./checkpoints/bart_decoder.pt')
    parser.add_argument("--max_c_len", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_aug_len", default=25, type=int)
    parser.add_argument("--max_q_len", default=50, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--predict_batch_size", default=512,
                        type=int, help="Total batch size for predictions.")

    return parser

def encode_args():
    """
    For encoding corpus into dense vectors
    """
    parser = common_args()
    parser.add_argument('--sparse', action='store_true', help='learned sparse encoding')
    parser.add_argument("--save_path", type=str, default=None)

    return parser.parse_args()


def train_args():
    parser = common_args()
    # optimization
    parser.add_argument('--prefix', type=str, default="eval")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--output_dir", default="./logs", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=128,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--save_checkpoints_steps", default=20000, type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument('--seed', type=int, default=3,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval_period', type=int, default=2500)
    parser.add_argument("--max_grad_norm", default=2.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--use_adam", action="store_true")
    parser.add_argument("--warmup_ratio", default=0, type=float,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--lm_weight", default=0.1, type=float)
    parser.add_argument("--aug_only", action="store_true", help='only train the query augmentation part')
    parser.add_argument("--dense_only", action="store_true",
                        help='dense retrieval only')

    parser.add_argument("--aug_option", type=str, default="title")


    parser.add_argument("--quantization", action='store_true')

    return parser.parse_args()
