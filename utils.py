import subprocess
import re

import huggingface_hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import spacy

SEP_CHARS = [".", "?", "!", ",", ":"]
SPECIAL_TOKENS = [
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",
    "[UNK]",
    "[PAR]",
    "[DOC]",
    "[TLE]",
    "<P>",
    "</P>",
    "<Tr>",
]


def get_model_tokenizer_device(hf_token, model_id, verbose=True):

    assert torch.cuda.is_available(), "CUDA is not available!"

    if verbose:
        print("CUDA is available with devices:")
        for i in range(torch.cuda.device_count()):
            print(f"\t- Device {i}: {torch.cuda.get_device_name(i)}")

    if verbose:
        print("Logging in to Hugging Face ...")
    huggingface_hub.login(hf_token)

    if verbose:
        print("Loading model and tokenizer ... ", end="")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        output_attentions=True,
        trust_remote_code=True,
        device_map="auto",
    )
    if verbose:
        print("Success!")

    main_device = torch.device("cuda:0")
    return model, tokenizer, main_device


def norm_text(input_string, sep_chars=SEP_CHARS, special_tokens=SPECIAL_TOKENS):
    res = re.sub(r"\n+", " ", input_string)  # remove multiple newlines
    res = re.sub(r"\s+", " ", res)  # remove multiple whitespaces
    # replace " x " with "x " for x in {., ?, !, ,, :}
    for sep_char in sep_chars:
        res = re.sub(rf"\s+\{sep_char}\s+", f"{sep_char} ", res)
    # remove special tokens
    for token in special_tokens:
        res = res.replace(token, "")
    return res.strip()


def save_dict_to_pickle(dict_obj, file_path, verbose=False):
    with open(file_path, "wb") as f:
        pickle.dump(dict_obj, f)
        # file size
        f.seek(0, 2)
        file_size = f.tell()
        info = f"::func:save_dict_to_pickle:: [{file_size / 1024 / 1024:.2f} MB] Saved to {file_path}"
        if verbose:
            print(info)
        return file_path


def load_dict_from_pickle(file_path):
    with open(file_path, "rb") as f:
        print(f"::func:load_dict_from_pickle:: Loading from {file_path}... ", end="")
        dict_obj = pickle.load(f)
        print("Done.")
    return dict_obj


def get_single_cuda_device(gpu_id="auto", verbose=1):
    """
    Check if CUDA is available. If available, print information based on the verbosity level,
    and select the GPU with the lowest memory usage. Return the corresponding device.

    Args:
    verbose (int): Verbosity level (0: no output, 1: current info, 2: detailed info).

    Returns:
    torch.device: The selected device.

    Raises:
    ValueError: If an invalid verbosity level is provided.
    """
    from rich.console import Console
    from rich.style import Style

    console = Console()
    pos_style = Style(color="black", bgcolor="green")
    neg_style = Style(color="black", bgcolor="red")

    if verbose not in [0, 1, 2]:
        raise ValueError("Invalid verbosity level. Please choose 0, 1, or 2.")

    if not torch.cuda.is_available():
        if verbose > 0:
            console.print(
                "CUDA is not available. Using CPU.",
                style=neg_style,
            )
        return torch.device("cpu")

    num_gpus = torch.cuda.device_count()

    gpu_memory_usage, gpu_free_memory = [], []
    gpu_info = []

    for i in range(num_gpus):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,name",
                "--format=csv,nounits,noheader",
                f"--id={i}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            if verbose > 0:
                print(f"Failed to query GPU {i}. Error: {result.stderr}")
            return torch.device("cpu")

        # Extract the memory usage and total memory
        memory_used, memory_total, gpu_name = map(
            str.strip, result.stdout.strip().split(",")
        )
        memory_used = int(memory_used) / 1024  # Convert MiB to GB
        memory_total = int(memory_total) / 1024  # Convert MiB to GB
        gpu_memory_usage.append((i, memory_used))
        gpu_free_memory.append((i, memory_total - memory_used))
        gpu_info.append((i, gpu_name, memory_used, memory_total))

    if gpu_id == "auto":
        best_gpu = max(gpu_free_memory, key=lambda x: x[1])[0]
    else:
        best_gpu = gpu_id
    device = torch.device(f"cuda:{best_gpu}")
    free_mem = gpu_free_memory[best_gpu][1]

    if verbose > 0:
        console.print(
            f"[CUDA: {num_gpus} GPU(s) found] Using '{device}' with {free_mem:.2f}/{gpu_info[best_gpu][3]:.2f} GB free memory.",
            highlight=True,
        )
    return device


def get_sentence_token_spans(context_ids, tokenizer):
    context_text = tokenizer.decode(context_ids[0])
    context_tokens_text = [
        tokenizer.decode([token_id]).replace(" ", "") for token_id in context_ids[0]
    ]
    sents = [sent.text for sent in spacy.load("en_core_web_sm")(context_text).sents]
    # if a sent is all " ", then merge it with the next sent
    for i in range(len(sents)):
        # if sents[i].strip() == "":
        if len(sents[i].strip()) <= 5:
            if i < len(sents) - 1:
                sents[i + 1] = sents[i] + sents[i + 1]
                sents[i] = ""
            else:
                sents[i - 1] = sents[i - 1] + sents[i]
                sents[i] = ""
    sents = [sent for sent in sents if sent != ""]

    # find sentence token spans
    sent_token_spans = []
    tk_start_idx = 0

    for i, sent in enumerate(sents):
        sent = sent.lstrip(" ")
        sent_num_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        # find the end token index
        sent_text = sent.replace(" ", "")
        span_text = tokenizer.decode(
            context_ids[0, tk_start_idx : tk_start_idx + sent_num_tokens]
        ).replace(" ", "")
        span_include_sent = span_text.find(sent_text) >= 0
        sent_include_span = sent_text.find(span_text) >= 0
        len_span = sent_num_tokens
        if span_include_sent and sent_include_span:  # pass
            pass
        elif span_include_sent and not sent_include_span:  # span is longer
            while True:
                len_span -= 1
                del_token = context_tokens_text[tk_start_idx + len_span]
                span_text = span_text.rstrip(del_token)
                if span_text.find(sent_text) < 0:  # span is shorter than sent
                    # len_span += 1
                    span_text = span_text + del_token
                    break
        elif not span_include_sent:  # span is shorter
            while True:
                add_token = context_tokens_text[tk_start_idx + len_span]
                len_span += 1
                span_text = span_text + add_token
                if span_text.find(sent_text) >= 0:
                    break

        tk_end_idx = tk_start_idx + len_span
        sent_token_spans.append((tk_start_idx, tk_end_idx))
        tk_start_idx = tk_end_idx

        if not span_text.endswith(sent_text):  # last token contains the next sentence
            tk_start_idx -= 1

    assert len(sent_token_spans) == len(sents)

    return sent_token_spans, sents
