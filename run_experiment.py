"""
This script runs the SelfElicit experiment for the given model on multiple datasets and methods.
"""

# %%

from args import get_args

args = get_args()

# %%

from utils import get_model_tokenizer_device
from qa_agent import get_agents_dict, ContextQuestionAnsweringAgent
import tqdm
import numpy as np
import pandas as pd
from eval import evaluate
from dataloader import load_data
from self_elicit import (
    get_answer_base,
    get_answer_cot,
    get_answer_fullelicit,
    get_answer_promptelicit,
    get_answer_selfelicit,
)
import warnings

warnings.filterwarnings("ignore")


def run_experiment(device, agents_dict, args):

    methods = args.methods
    datasets_dict = {
        dataset: load_data(dataset, args.n_samples, args.random_state, True)
        for dataset in args.datasets
    }
    qa_res_eval_cols = [
        "em",
        "f1",
        "pr",
        "re",
    ]
    qa_res_columns = [
        "dataset",
        "idx",
        "true_ans",
        "model_ans",
        "method",
    ] + qa_res_eval_cols

    qa_results = []

    for dataset_name, dataset in datasets_dict.items():

        dataset_runstat = {
            "f1": {method: [] for method in methods},
            "em": {method: [] for method in methods},
        }

        iterator = tqdm.tqdm(range(len(dataset)), desc=f"DATA - {dataset_name:<10s}")
        for idx in iterator:
            context, question, true_ans_list = dataset.get_context_question_answer(idx)

            for method in methods:
                try:
                    if method == "Base":
                        model_ans = get_answer_base(
                            context, question, agents_dict, args
                        )
                    elif method == "COT":
                        model_ans = get_answer_cot(context, question, agents_dict, args)
                    elif method == "FullElicit":
                        model_ans = get_answer_fullelicit(
                            context, question, agents_dict, args
                        )
                    elif method == "PromptElicit":
                        model_ans = get_answer_promptelicit(
                            context, question, agents_dict, args
                        )
                    elif method == "SelfElicit":
                        model_ans, evidence_sents = get_answer_selfelicit(
                            context,
                            question,
                            agents_dict,
                            device,
                            args,
                            return_evidence=True,
                        )
                except:
                    continue

                true_ans_used, scores = evaluate(
                    true_ans_list, model_ans, sel_metric="f1"
                )

                qa_results.append(
                    [
                        dataset_name,
                        idx,
                        true_ans_used,
                        model_ans,
                        method,
                    ]
                    + [scores[col] for col in qa_res_eval_cols]
                )

                dataset_runstat["f1"][method].append(scores["f1"] * 100)
                dataset_runstat["em"][method].append(scores["em"] * 100)

            iterator.set_postfix(
                {
                    "f1": {
                        method: np.mean(dataset_runstat["f1"][method]).round(2)
                        for method in methods
                    },
                    "em": {
                        method: np.mean(dataset_runstat["em"][method]).round(2)
                        for method in methods
                    },
                }
            )

    qa_results = pd.DataFrame(qa_results, columns=qa_res_columns)
    return qa_results


def main():
    # Load model, tokenizer, and devices
    model, tokenizer, device = get_model_tokenizer_device(args.hf_token, args.model_id)
    # Prepare QA agent instances (with different instructions)
    agents_dict = get_agents_dict(model, tokenizer, device, args)
    # Run the experiment
    qa_results = run_experiment(device, agents_dict, args)
    # Save the results
    path = f"results/exp_[MODEL]{args.model_id.replace('/', '|')}_[METHOD]{'-'.join(args.methods)}_[DATA]{'-'.join(args.datasets)}.csv"
    print(f"Saving results to {path} ...", end="")
    qa_results.to_csv(path, index=False)
    print("Success!")


if __name__ == "__main__":
    main()
