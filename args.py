import os
import argparse
import yaml


def load_config(config_file="config.yaml"):
    """Load the default configuration from a YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_args(config_file="config.yaml", using_notebook=False, verbose=1):

    if verbose:
        print(f"Loading default configuration from '{config_file}' ...")

    # Load default values from config file
    config = load_config(config_file)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Configuration for QA and SE Instructions"
    )

    # Add methods and datasets arguments
    ALL_MODELS = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ]
    ALL_METHODS = ["Base", "COT", "FullElicit", "PromptElicit", "SelfElicit"]
    ALL_DATASETS = ["HotpotQA", "NewsQA", "TQA", "NQ"]

    # Add arguments
    parser.add_argument(
        "--hf_token",
        type=str,
        default=config["hf_token"],
        help=f"Hugging Face API token",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=config["model_id"],
        help=f"The HuggingFace Model ID, should be one of {ALL_MODELS}",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=config["methods"],
        help=f"Method(s) to test, can be a list or a single value from {ALL_METHODS}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=config["datasets"],
        help=f"Dataset(s) to use, can be a list or a single value from {ALL_DATASETS}",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=config["alpha"],
        help="Threshold for SelfElicit method",
    )
    parser.add_argument(
        "--layer_span",
        type=tuple,
        default=tuple(config["layer_span"]),
        help="Layer span for SelfElicit method",
    )
    parser.add_argument(
        "--gpu_ids", nargs="+", default=config["gpu_ids"], help="GPU IDs"
    )
    parser.add_argument(
        "--n_samples", type=int, default=config["n_samples"], help="Number of samples"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=config["random_state"],
        help="Random state for reproducibility",
    )
    parser.add_argument(
        "--max_ans_tokens",
        type=int,
        default=config["max_ans_tokens"],
        help="Maximum answer length in tokens",
    )
    parser.add_argument(
        "--marker_impstart",
        type=str,
        default=config["marker_impstart"],
        help="Marker for the start of important information",
    )
    parser.add_argument(
        "--marker_impend",
        type=str,
        default=config["marker_impend"],
        help="Marker for the end of important information",
    )
    parser.add_argument(
        "--qa_inst", type=str, default=config["qa_inst"], help="QA instruction"
    )
    parser.add_argument(
        "--se_inst",
        type=str,
        default=config["se_inst"],
        help="QA instruction for SelfElicit",
    )
    parser.add_argument(
        "--cot_inst",
        type=str,
        default=config["cot_inst"],
        help="QA instruction with Chain of Thought prompt",
    )
    parser.add_argument(
        "--pe_inst",
        type=str,
        default=config["pe_inst"],
        help="Instruction for 1st-step extracting evidence in PromptElicit",
    )

    # Parse arguments
    if using_notebook:
        if verbose:
            print(
                "Parsing arguments from command line is disabled as using_notebook=True."
            )
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # Set environment variables for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.gpu_ids])
    if verbose:
        print("Using GPUs: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    # Validate 'model' input
    if args.model_id not in ALL_MODELS:
        raise ValueError(f"Invalid model: {args.model_id}. Must be one of {ALL_MODELS}")

    # Validate 'methods' input
    if isinstance(args.methods, list):
        for method in args.methods:
            if method not in ALL_METHODS:
                raise ValueError(
                    f"Invalid method: {method}. Must be one of {ALL_METHODS}"
                )
    elif args.methods not in ALL_METHODS:
        raise ValueError(
            f"Invalid method: {args.methods}. Must be one of {ALL_METHODS}"
        )

    # Validate 'datasets' input
    if isinstance(args.datasets, list):
        for dataset in args.datasets:
            if dataset not in ALL_DATASETS:
                raise ValueError(
                    f"Invalid dataset: {dataset}. Must be one of {ALL_DATASETS}"
                )
    elif args.datasets not in ALL_DATASETS:
        raise ValueError(
            f"Invalid dataset: {args.datasets}. Must be one of {ALL_DATASETS}"
        )

    # Fill in the instruction strings
    assert (
        "{MARKER_IMPSTART}" in args.se_inst and "{MARKER_IMPEND}" in args.se_inst
    ), "Instruction for SelfElicit must contain {MARKER_IMPSTART} and {MARKER_IMPEND}"
    args.se_inst = args.se_inst.format(
        MARKER_IMPSTART=args.marker_impstart, MARKER_IMPEND=args.marker_impend
    )

    if verbose:
        print("Arguments loaded successfully!\nArguments:")
        for key, value in vars(args).items():
            if key == "hf_token":
                print(f"\t{key:<10s}: {'*' * len(value)}")
            else:
                print(f"\t{key:<10s}: {value}")

    return args
