from pprint import pprint
import argparse
from .utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default='EleutherAI/gpt-neo-2.7B')
    parser.add_argument("--adaptor_class", type=str, default="multiply")
    parser.add_argument("--adapted_component", type=str, default="final_layer")
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--init_var", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--num_switches", type=int, default=50)

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_file", type=str,
                        default="data/gun_control.txt")
    parser.add_argument("--regularization", type=float, default=0)

    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_step", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)

    print("arguments:")
    pprint(args.__dict__)
    return args
