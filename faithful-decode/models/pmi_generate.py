"""
generates a response given a trained model
Inspied by https://github.com/huggingface/transformers/blob/v4.15.0/examples/pytorch/text-generation/run_generation.py
"""
from IPython.core.debugger import Pdb
from dataset import DialogueDataModule, SpecialVocab

import json
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

import torch
import yaml
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from transformers.generation_logits_process import (
    LogitsProcessorList,
    TopPLogitsWarper,
    LogitsProcessor,
)
import torch.nn.functional as F

from logits_processors import PMILogitsProcessor
import time

# sys.path.insert(0, Path(__file__).parent.parent.absolute().as_posix())

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("generate")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_LENGTH = int(1000)  # Hardcoded max length to avoid infinite loop


"""
p = [0.6, 0.2, 0.1, 0.05, 0.05]
fp = [0.6, 0.2, 0.1, -inf, -inf]
mp = [0, 0, 0, -inf, -inf]

(PMI + Prob)*Mask(top_p): PMI + fp
(PMI )*Mask(top_p): PMI + mp

"""


def set_default_args(args):
    if not args.control_tokens:
        args.control_tokens = []
    elif args.control_tokens[0] == "none":
        args.control_tokens = []
    elif args.control_tokens[0] == "all":
        args.control_tokens = ["<no-first-person>", "<high-prec>", "<entailed>"]

    args.do_generate = True
    args.predict_dataset_path = args.dataset_path

    args.do_train = False
    args.do_eval = False
    args.do_test = False
    args.ctrl = False
    args.max_negative_samples = 0

    args.pad_to_multiple_of = None

    hparams_path = Path(args.model_name_or_path).parent / "hparams.yaml"

    if hparams_path.exists():
        logger.info(
            "`hparams.yaml` found from which parameter values (max_history, pad_to_multiple_of) will be loaded"
        )

        with hparams_path.open("r") as hparams_file:
            train_hparams = yaml.safe_load(hparams_file)

        args.pad_to_multiple_of = train_hparams.get("pad_to_multiple_of", None)
        args.max_history = args.max_history or train_hparams.get("max_history", None)
        args.ctrl = train_hparams.get("ctrl", False)


def get_output_name(args) -> str:
    name = "generated"

    if args.dataset_path:
        name += f"_{Path(args.dataset_path).stem}"

    if args.num_return_sequences > 1:
        name += f"_n{args.num_return_sequences}"

    name += f"_maxHist{args.max_history}_maxLen{args.max_length}"

    if args.copy_knowledge == 1:
        name = "{}_copyknow".format(name)
    elif args.copy_gold == 1:
        name = "{}_copygold".format(name)
    else:
        if args.temperature != 1.0:
            name += f"_temp{args.temperature}"

        if args.repetition_penalty != 1.0:
            name += f"_repPen{args.repetition_penalty}"

        if args.do_sample or args.pmi_decode:
            if args.top_k > 0:
                name += f"_k{args.top_k}"

            if args.top_p > 0:
                name += f"_p{args.top_p}"

            if args.pmi_decode:
                if args.add_log_prob_in_pmi:
                    name += f"_addp{args.add_log_prob_in_pmi}"
                    name += f"_pmiw{args.pmi_weight}"

            if args.do_sample:
                name += "_sample"
        else:
            name += "_greedy"

        if args.ctrl and args.control_tokens:
            ctrl_tokens = ",".join([tok[1:-1] for tok in args.control_tokens])
            name += f"_{ctrl_tokens}"

    return name


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--copy_knowledge",
        type=int,
        default=0,
        help="just copy knowledge as response",
    )

    parser.add_argument(
        "--copy_gold",
        type=int,
        default=0,
        help="just copy gold as response",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path or url of the Json dataset.",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path to a trained model"
    )
    parser.add_argument("--max_seq_length", type=int, default=512)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path of the output directory to save the responses",
    )
    parser.add_argument(
        "--max_history",
        type=int,
        default=2,
        help="Number of previous exchanges to keep in history",
    )

    parser.add_argument(
        "--filter",
        type=int,
        default=0,
        help=" should filter out samples larger than Max sequence length?",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--min_length", type=int, default=2)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="value used to module the next token probabilities",
    )
    parser.add_argument(
        "--do_sample",
        type=int,
        default=0,
        # action="store_true",
        # default=False,
        help="Whether or not to use sampling ; use greedy decoding otherwise.",
    )

    parser.add_argument(
        "--pmi_decode",
        action="store_true",
        default=False,
        help="Whether or not to use pmi based decoding strategy.",
    )

    parser.add_argument(
        "--add_log_prob_in_pmi",
        # action="store_true",
        type=int,
        default=1,
        help="Whether or not to add log prob to  pmi for ranking .",
    )

    parser.add_argument(
        "--pmi_weight",
        type=float,
        default=0.5,
        help="weight of pmi vs log prob.",
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )

    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--exclude_knowledge",
        action="store_true",
        default=False,
        help="Whether to exclude knowledge from input sequences",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--num_workers", default=0, type=int, help="kwarg passed to DataLoader"
    )
    parser.add_argument(
        "--control_tokens",
        nargs="*",
        default=("<entailed>",),
        help="Prepend control tokens to the sequence for controlled generation "
        "(works only when model is trained with `--ctrl`). List of control tokens are: "
        "<entailed>, <non-entailed>, <first-person>, <no-first-person>, <high-prec>, <med-prec>, <low-prec>. "
        "To use all of them, simply pass `--control_tokens all` and for none, pass `--control_tokens none`.",
    )

    args = parser.parse_args()

    # if args.pmi_decode:
    #    assert not args.do_sample

    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    set_default_args(args)
    logger.info(f"Arguments: {pformat(args)}")

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        return_dict=True,
    )
    padding_side = "right" if config.is_encoder_decoder else "left"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, extra_ids=0, padding_side=padding_side
        )
    except ValueError:
        logger.warning(
            "Creating tokenizer failed, trying again without extra_ids (used only for T5). "
            "In this setting, the model may generate reserved tokens (<extra_id_%%>)."
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side
        )

    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path, config=config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, config=config
        )

    special_vocab = SpecialVocab(tokenizer, args.ctrl, initialized=True)

    logit_processor_list = None

    if args.pmi_decode:
        logit_processor_list = LogitsProcessorList()
        logit_processor_list.append(
            PMILogitsProcessor(
                top_p=args.top_p,
                add_log_prob=args.add_log_prob_in_pmi,
                pmi_weight=args.pmi_weight,
            )
        )

    model.to(args.device)

    if args.max_length < 0 and tokenizer.model_max_length > 0:
        args.max_length = tokenizer.model_max_length
    elif 0 < tokenizer.model_max_length < args.max_length:
        # No generation bigger than model size
        args.max_length = tokenizer.model_max_length
    elif args.max_length < 0:
        args.max_length = MAX_LENGTH  # avoid infinite loop

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    data_module = DialogueDataModule(special_vocab, args, config.is_encoder_decoder)
    data_module.setup("fit")

    logger.info(f"Test dataset size: {len(data_module.datasets['generate'])}")

    # Evaluation function and evaluator (evaluator output is the input of the metrics)

    if args.output:
        output_dir = Path(args.output)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = Path(args.model_name_or_path) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{get_output_name(args)}.jsonl"
    print("$$$$$$$$$$$$$$$ out file: ", out_file)
    logger.info(f"Results will be saved in `{out_file}`")

    example_idx = 0

    if True:
        with out_file.open("w", encoding="utf-8") as writer:
            predict_dataloader = data_module.predict_dataloader()

            start_time = time.time()
            for batch in tqdm(predict_dataloader, total=len(predict_dataloader)):
                model.eval()
                batch = {k: t.to(args.device) for k, t in batch.items()}
                input_ids = batch["input_ids"]

                if "token_type_ids" in batch:
                    gen_kwargs = {"token_type_ids": batch["token_type_ids"]}
                else:
                    # gen_kwargs = {"num_beams": 1}
                    gen_kwargs = {}

                # input_lengths = (input_ids != tokenizer.pad_token_id).int().sum(-1)
                # Pdb().set_trace()
                # responses: (batch_size * num_return_sequences, sequence_length)
                batch_size = input_ids.shape[0]

                if (args.copy_knowledge == 0) and (args.copy_gold == 0):
                    with torch.no_grad():
                        responses = getattr(model, "module", model).generate(
                            input_ids,
                            decoder_start_token_id=special_vocab.wizard_token_id,
                            do_sample=bool(
                                args.do_sample
                            ),
                            max_length=(
                                0 if config.is_encoder_decoder else input_ids.shape[-1]
                            )
                            + args.max_length,
                            min_length=args.min_length,
                            top_p=(
                                1.0

                                if (args.pmi_decode)
                                else args.top_p
                            ),
                            top_k=args.top_k,
                            temperature=args.temperature,
                            num_return_sequences=args.num_return_sequences,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True,
                            logits_processor=logit_processor_list,
                            **gen_kwargs,
                        )

                    # batch_size = input_ids.shape[0]

                    # responses: (batch_size, num_return_sequences, sequence_length)
                    responses = responses.reshape(
                        batch_size, args.num_return_sequences, -1
                    )
                    responses = responses.cpu().numpy()

                    if args.pmi_decode:
                        batch_size = batch_size // 2
                this_time = time.time()
                for b in range(batch_size):
                    example = data_module.datasets["generate"][example_idx]
                    out = {
                        "dialog_idx": example["dialog_idx"],
                        "response": example["response"],
                        "history": example["history"],
                        "knowledge": example["knowledge"],
                        "start_time": start_time,
                        "this_time": this_time,
                        "cumulative_time": (this_time - start_time)
                    }

                    if "original_response" in example:
                        out["original_response"] = example["original_response"]

                    if "BEGIN" in example:
                        out["BEGIN"] = example["BEGIN"]

                    if "VRM" in example:
                        out["VRM"] = example["VRM"]
                    # Pdb().set_trace()

                    if args.copy_knowledge == 1:
                        out["generated_response"] = [example["knowledge"]]
                    elif args.copy_gold == 1:
                        out["generated_response"] = [example["response"]]
                    else:
                        generated_responses = [
                            tokenizer.decode(
                                responses[b, i]

                                if config.is_encoder_decoder
                                else responses[b, i, input_ids.shape[1] :],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            ).strip()

                            for i in range(args.num_return_sequences)
                        ]

                        out["generated_response"] = [
                            resp if resp else " " for resp in generated_responses
                        ]

                        if not out["generated_response"]:
                            # Pdb().set_trace()
                            logger.warning(
                                f"Empty generated response at {example_idx}: {out}"
                            )

                    writer.write(json.dumps(out) + "\n")

                    example_idx += 1


if __name__ == "__main__":
    main()
