from datasets import load_dataset
import numpy as np
import pandas as pd
import warnings

from utils import norm_text


def load_data(dataset_name, n_samples=1000, random_state=42, verbose=True):
    """
    Load a dataset by name and return a preprocessed dataset object.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load (e.g., "HotpotQA", "NewsQA").
    n_samples : int, optional (default=1000)
        Number of samples to load from the dataset.
    random_state : int, optional (default=42)
        Random seed for shuffling.
    verbose : bool, optional (default=True)
        Whether to print progress messages.

    Returns
    -------
    data : object
        An instance of the dataset class corresponding to the selected dataset.

    Raises
    ------
    ValueError
        If the specified dataset name is invalid.
    """
    # Set default parameters for dataset loading
    kwargs = {"n_samples": n_samples, "random_state": random_state, "verbose": verbose}

    # Match dataset name to the appropriate class
    if dataset_name == "HotpotQA":
        data = HotpotQA(**kwargs)
    elif dataset_name == "NewsQA":
        data = NewsQA(**kwargs)
    elif dataset_name == "TQA":
        data = TQA(**kwargs)
    elif dataset_name == "NQ":
        data = NQ(**kwargs)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    return data


class HotpotQA:
    """
    A class for loading and processing the HotpotQA dataset.


    Parameters
    ----------
    n_samples : int, optional
        Number of samples to load.

    shuffle : bool, optional (default=True)
        Whether to shuffle the dataset.

    random_state : int, optional (default=42)
        Random seed for shuffling.

    verbose : bool, optional (default=True)
        Whether to print progress messages.
    """

    HF_DATASET = "hotpotqa/hotpot_qa"

    def __init__(
        self,
        n_samples=None,
        shuffle=True,
        random_state=42,
        verbose=True,
    ):
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

        # Load hotpotQA dataset
        if verbose:
            print(f"Loading the HotpotQA dataset ...", end=" ")
        dataset = load_dataset(self.HF_DATASET, "distractor", trust_remote_code=True)[
            "validation"
        ]

        # Get the length of the dataset
        dataset_length = len(dataset)

        # Shuffle the dataset
        if shuffle:
            dataset = dataset.shuffle(seed=random_state)

        # Check if the dataset has fewer samples than requested
        if n_samples is None:
            dataset = dataset
        elif dataset_length < n_samples:
            warnings.warn(
                f"The dataset only has {dataset_length} samples that satisfy the filtering criteria."
            )
            dataset = dataset
        elif dataset_length >= n_samples:
            # Create a subset
            dataset = dataset.select(range(n_samples))

        if verbose:
            print("Success!")
            # print(
            #     f"Loaded {len(dataset)} samples from '{self.HF_DATASET}' (random_state={random_state}, shuffle={shuffle}, n_samples={n_samples})\n"
            # )

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_context_question(self, idx, use_gold=True, norm=False):
        """
        Get the context and question for a specific index.

        Parameters
        ----------
        idx : int
            Index of the sample.
        use_gold : bool, optional (default=True)
            Whether to use the gold context (supporting facts).
        norm : bool, optional (default=False)
            Whether to normalize the context text.

        Returns
        -------
        context : str
            The context text.
        question : str
            The question text.
        """
        if use_gold:
            context = self.get_gold_context(idx)
        else:
            context = self.get_context(idx)
        if norm:
            context = norm_text(context)
        question = self.dataset[idx]["question"]
        return context, question

    def get_context(self, idx):
        context = self.dataset[idx]["context"]
        title_sent_start_index = {}
        sent_counter = 0
        context_text = ""
        for i in range(len(context["title"])):
            title_sent_start_index[context["title"][i]] = sent_counter
            for j in range(len(context["sentences"][i])):
                context_text += context["sentences"][i][j]
                sent_counter += 1
            context_text += "\n"
        return context_text

    def get_gold_context(self, idx, return_list=False):
        context = self.dataset[idx]["context"]
        gold_facts = self.dataset[idx]["supporting_facts"]
        gold_sents = []

        for i in range(len(gold_facts["title"])):
            title = gold_facts["title"][i]
            title_id = context["title"].index(title)
            sent_id = gold_facts["sent_id"][i]
            sent_text = context["sentences"][title_id][sent_id]
            gold_sents.append(sent_text)

        if return_list:
            return gold_sents
        else:
            gold_context_text = ""
            for sent_text in gold_sents:
                gold_context_text += sent_text + " "
            return gold_context_text

    def get_answers(self, idx):
        return [self[idx]["answer"]]  # return list of answers

    def get_answer_char_spans(self, idx):
        context_text = self.get_context(idx)
        gold_sents = self.get_gold_context(idx, return_list=True)
        evd_span = {}
        for sent in gold_sents:
            start = context_text.find(sent)
            end = start + len(sent)
            evd_span[sent] = np.array([[start, end]])
        return evd_span

    def get_cqas(self, idx, use_gold=True, norm=True, verbose=False):
        context, question = self.get_context_question(idx, use_gold, norm)
        answers = self.get_answers(idx)
        answer_char_spans = self.get_answer_char_spans(idx)
        if verbose:
            print(
                f"Context: {context}\nQuestion: {question}\n"
                f"Answer: {answers}\nAnswer char span: {answer_char_spans}\n"
                f"Answer extracted by span: {context[answer_char_spans[0][0]:answer_char_spans[1][0]]}"
            )
        return context, question, answers, answer_char_spans

    def get_context_question_answer(self, idx, use_gold=True, norm=True):
        context, question = self.get_context_question(idx, use_gold, norm)
        answers = self.get_answers(idx)
        return context, question, answers

    def get_setting_description(self):
        name = self.HF_DATASET.split("/")[-1].upper()
        return f"{name}_len{len(self.dataset)}_shuffle{self.shuffle}_seed{self.random_state}"


class MRQA:
    """
    A class for loading and processing datasets under the MRQA framework.

    Parameters
    ----------
    subset : str
        Subset of the MRQA dataset to load (e.g., "NewsQA", "TriviaQA-web").

    n_samples : int, optional
        Number of samples to load.

    shuffle : bool, optional (default=True)
        Whether to shuffle the dataset.

    random_state : int, optional (default=42)
        Random seed for shuffling.

    verbose : bool, optional (default=True)
        Whether to print progress messages.
    """

    HF_DATASET_NAME = "mrqa-workshop/mrqa"
    SUBSETS = [
        "NewsQA",
        "TriviaQA-web",
        "NaturalQuestionsShort",
    ]

    def __init__(
        self,
        subset,
        n_samples=None,
        shuffle=True,
        random_state=42,
        verbose=True,
    ):
        assert subset in self.SUBSETS, f"subset must be one of {self.SUBSETS}"
        if n_samples is not None:
            assert n_samples > 0, "n_samples must be a positive integer"
            assert (
                shuffle == True
            ), "shuffle must be True when n_samples and subset are specified"

        self.subset = subset
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

        # Load the dataset
        if verbose:
            print(f"Loading the {subset} dataset ...", end=" ")
        dataset = load_dataset(self.HF_DATASET_NAME, trust_remote_code=True)[
            "validation"
        ]
        # Filter the dataset by subset
        if subset is not None:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            dataset = dataset.filter(
                lambda example: example["subset"] == subset,
                num_proc=min(32, cpu_count),
            )

        # Get the length of the dataset
        dataset_length = len(dataset)

        # Shuffle the dataset
        if shuffle:
            dataset = dataset.shuffle(seed=random_state)

        # Check if the dataset has fewer samples than requested
        if n_samples is None:
            dataset = dataset
        elif dataset_length < n_samples:
            warnings.warn(
                f"The dataset only has {dataset_length} samples that satisfy the filtering criteria."
            )
            dataset = dataset
        elif dataset_length >= n_samples:
            # Create a subset
            dataset = dataset.select(range(n_samples))

        if verbose:
            print("Success!")
            # print(
            #     f"Loaded {len(dataset)} samples from '{self.HF_DATASET_NAME}' (random_state={random_state}, subset={subset}, shuffle={shuffle}, n_samples={n_samples})\n"
            # )

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_context_question(self, idx, norm=False):
        """
        Get the context and question for a specific index.

        Parameters
        ----------
        idx : int
            Index of the sample.
        norm : bool, optional (default=False)
            Whether to normalize the context text.

        Returns
        -------
        context : str
            The context text.
        question : str
            The question text.
        """
        example = self[idx]
        context = example["context"]
        if norm:
            context = norm_text(context)
        question = example["question"]
        return context, question

    def get_answers(self, idx):
        example = self[idx]
        answers = example["detected_answers"]["text"]
        return answers

    def get_answer_char_spans(self, idx):
        example = self[idx]
        char_spans = example["detected_answers"]["char_spans"]
        ans_text = example["detected_answers"]["text"]
        all_char_spans = {}
        for i, span in enumerate(char_spans):
            all_char_spans[ans_text[i]] = np.array(
                [[span["start"][j], span["end"][j]] for j in range(len(span["start"]))]
            )
        return all_char_spans

    def get_cqas(self, idx, verbose=False, norm=True):
        context, question = self.get_context_question(idx, norm)
        answers = self.get_answers(idx)
        answer_char_spans = self.get_answer_char_spans(idx)
        if verbose:
            print(
                f"Context: {context}\nQuestion: {question}\n"
                f"Answer: {answers}\nAnswer char span: {answer_char_spans}\n"
                f"Answer extracted by span: {context[answer_char_spans[0][0]:answer_char_spans[1][0]]}"
            )
        return context, question, answers, answer_char_spans

    def get_context_question_answer(self, idx, norm=True):
        context, question = self.get_context_question(idx, norm)
        answers = self.get_answers(idx)
        return context, question, answers

    def get_setting_description(self):
        name = self.HF_DATASET_NAME.split("/")[-1].upper()
        return f"{name}_subset{self.subset}_len{len(self.dataset)}_shuffle{self.shuffle}_seed{self.random_state}"


class NewsQA(MRQA):

    def __init__(
        self,
        n_samples=None,
        shuffle=True,
        random_state=42,
        verbose=True,
    ):
        super().__init__(
            subset="NewsQA",
            n_samples=n_samples,
            shuffle=shuffle,
            random_state=random_state,
            verbose=verbose,
        )


class TQA(MRQA):

    def __init__(
        self,
        n_samples=None,
        shuffle=True,
        random_state=42,
        verbose=True,
    ):
        super().__init__(
            subset="TriviaQA-web",
            n_samples=n_samples,
            shuffle=shuffle,
            random_state=random_state,
            verbose=verbose,
        )


class NQ(MRQA):

    def __init__(
        self,
        n_samples=None,
        shuffle=True,
        random_state=42,
        verbose=True,
    ):
        super().__init__(
            subset="NaturalQuestionsShort",
            n_samples=n_samples,
            shuffle=shuffle,
            random_state=random_state,
            verbose=verbose,
        )
