import argparse

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM

from awq import AutoAWQForCausalLM
from awq.quantize.quantizer import AwqQuantizer
from awq.evaluation.perplexity import compute_perplexity
from awq.utils.data_utils import get_data, get_wikitext2
from awq.quantize.quant_ops import NVFP_GROUPSIZE, MXFP_GROUPSIZE

try:
    import wandb
except ImportError:
    wandb = None

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def auto_or_int(value):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Must be 'auto' or an integer, got '{value}'")
    
def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to quantized model.",
    )
    # Data params
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="The name or path to the calibration dataset.",
    )
    parser.add_argument(
        "--sequence_length", 
        default=8192, 
        type=int, 
        help="Length of calibration sequences."
    )
    parser.add_argument(
        "--num_sequences", 
        default=128, 
        type=int, 
        help="Number of calibration sequences."
    )
    # Quantization params
    parser.add_argument(
        "--format",
        type=str,
        default="int",
        choices=["int", "fp", "nvfp", "mxfp"],
        help="Quantization format.",
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        required=True,
        help="Weight quantization bitwidth.",
    )
    parser.add_argument(
        "--w_group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    # Logging params
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log to wandb."
    )
    # Misc params
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    # Eval params
    parser.add_argument("--eval_perplexity", action="store_true", help="whether to eval perplexity after quantization.")
    parser.add_argument("--eval_openllm", action="store_true", help="whether to eval OpenLLM v1 openllm after quantization.")
    # LM eval params
    parser.add_argument(
        "--lm_eval_batch_size",
        type=auto_or_int,
        default="auto",
        help="LM eval batch size to evaluate after quantization.",
    )
    parser.add_argument(
        "--lm_eval_tasks",
        nargs="+",
        type=str,
        default=["arc_easy", "arc_challenge", "winogrande", "piqa", "hellaswag"],
        help="LM eval tasks to evaluate after quantization.",
    )
    parser.add_argument(
        "--lm_eval_add_bos_token", 
        action="store_true",
        help="whether to add bos token in evaluation."
    )
    parser.add_argument(
        "--lm_eval_apply_chat_template",
        action="store_true",
        help="whether to apply chat template."
    )
    parser.add_argument(
        "--lm_eval_fewshot_as_multiturn",
        action="store_true",
        help="whether to process fewshot as multiturn." 
    )
    # Save params
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save quantized model",
    )
    # Parse arguments
    args = parser.parse_args()
    # Check and fix group_size (if needed)
    if args.format == "nvfp":
        if args.w_group_size != NVFP_GROUPSIZE:
            args.w_group_size = NVFP_GROUPSIZE
            print(f"Changed weight group_size to {NVFP_GROUPSIZE} for nvfp format.")
    elif args.format == "mxfp":
        if args.w_group_size != MXFP_GROUPSIZE:
            args.w_group_size = MXFP_GROUPSIZE
            print(f"Changed weight group_size to {MXFP_GROUPSIZE} for mxfp format.")
    # Check logging
    if args.log_wandb:
        assert wandb is not None, "wandb is not installed. Please install wandb `pip install wandb`."
    
    return args


def main():
    args = parse_args()
    # Set device
    device = "cuda"
    # Get dtype
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Init logger
    if args.log_wandb:
        wandb.init(config=args)
    # Model
    awq_model = AutoAWQForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=args.dtype, 
        device_map=device, # to avoid errors when model is split on mulitple GPUs
        low_cpu_mem_usage=True,
    )
    awq_model.config.use_cache = False
    awq_model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Sanity check
    if args.eval_openllm:
        assert hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None, "OpenLLM v1 works only with chat template."

    quantize_anything = args.w_bits < 16

    if quantize_anything:
        calibration_data = get_data(
            args.dataset_name_or_path,
            tokenizer,
            args.sequence_length,
            args.num_sequences,
            args.seed
        )
        quantizer = AwqQuantizer(
            awq_model,
            awq_model.model,
            tokenizer,
            w_bit=args.w_bits,
            group_size=args.w_group_size,
            version="v2",
            calib_data=calibration_data,
            split="none",
            text_column="text",
            duo_scaling=True,
            modules_to_not_convert=None,
            export_compatible=False,
            format=args.format,
            real_quant=False
        )
        quantizer.quantize()
            
    # Set to eval mode
    awq_model.requires_grad_(False).eval().to(device)

    if args.eval_perplexity:
        eval_data = get_wikitext2(tokenizer, args.sequence_length)
        ppl = compute_perplexity(awq_model, eval_data)
        print(f"Wikitext-2 perplexity: {round(ppl, 2):.2f}")
        if args.log_wandb:
            wandb.log({"eval/wikitext2_ppl": ppl})

    # OpenLLM v1 openllm (following https://arxiv.org/abs/2411.02355)
    if args.eval_openllm:

        results = {}
        lm = HFLM(
            pretrained=awq_model.model, 
            tokenizer=tokenizer, 
            batch_size=args.lm_eval_batch_size,
            max_length=4096, # from open LLM openllm
            add_bos_token=args.lm_eval_add_bos_token
        )
        task_manager = lm_eval.tasks.TaskManager()

        # MMLU CoT Llama-3.1
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="mmlu_cot_llama",
                batch_size=args.lm_eval_batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                task_manager=task_manager,
            )["results"]
        )
        # ArcC Llama-3.1
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="arc_challenge_llama",
                batch_size=args.lm_eval_batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                task_manager=task_manager,
            )["results"]
        )
        # GSM8K Llama-3.1
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="gsm8k_llama",
                batch_size=args.lm_eval_batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                task_manager=task_manager,
            )["results"]
        )
        # Hellaswag (10-shot)
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="hellaswag",
                num_fewshot=10,
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
        )
        # Winogrande (5-shot)
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="winogrande",
                num_fewshot=5,
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
        )
        # TruthfulQA (0-shot)
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="truthfulqa",
                num_fewshot=0,
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
        )

        # Log results
        if args.log_wandb:
            wandb.log({"eval/openllm": results}) 
        # Print formatted table
        print(make_table({"results": results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
            
    if args.save_path:
        awq_model.model.save_pretrained(args.save_path)  
        tokenizer.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()
