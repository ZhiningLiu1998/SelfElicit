import re
import torch
import argparse


def get_agents_dict(model, tokenizer, device, args):
    """
    Initialize a dictionary of agents for different tasks (QA, CoT, SE, PE).

    Parameters
    ----------
    model : torch.nn.Module
        The language model used for question answering.

    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer corresponding to the model.

    device : torch.device
        Device on which the model computations will be performed.

    args : argparse.Namespace
        Arguments object containing the configuration for agents. Must include:
        - `qa_inst`: Instruction for QA agent.
        - `se_inst`: Instruction for self-elicit agent.
        - `cot_inst`: Instruction for chain-of-thought agent.
        - `pe_inst`: Instruction for prompt-elicit agent.
        - `max_ans_tokens`: Maximum tokens for generated answers.

    Returns
    -------
    dict
        Dictionary of agents with keys 'qa', 'se', 'cot', and 'pe'.
    """
    # Ensure the input arguments are correctly structured
    assert (
        type(args) == argparse.Namespace
    ), "args should be an argparse.Namespace object"
    assert hasattr(args, "qa_inst"), "args should have 'qa_inst' attribute"
    assert hasattr(args, "se_inst"), "args should have 'se_inst' attribute"
    assert hasattr(args, "cot_inst"), "args should have 'cot_inst' attribute"
    assert hasattr(args, "pe_inst"), "args should have 'pe_inst' attribute"
    # Prepare shared arguments for all agents
    agent_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "max_ans_tokens": args.max_ans_tokens,
    }
    # Initialize and return a dictionary of agents
    return {
        "qa": ContextQuestionAnsweringAgent(instruction=args.qa_inst, **agent_kwargs),
        "se": ContextQuestionAnsweringAgent(instruction=args.se_inst, **agent_kwargs),
        "cot": ContextQuestionAnsweringAgent(instruction=args.cot_inst, **agent_kwargs),
        "pe": ContextQuestionAnsweringAgent(instruction=args.pe_inst, **agent_kwargs),
    }


class ContextQuestionAnsweringAgent:
    """
    A class for handling context-based question answering with various strategies.

    Parameters
    ----------
    model : torch.nn.Module
        The language model used for question answering.

    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer corresponding to the model.

    device : torch.device
        Device on which the model computations will be performed.

    instruction : str
        Instruction used to guide the model's behavior during question answering.

    max_ans_tokens : int
        Maximum tokens for generated answers.

    Attributes
    ----------
    model : torch.nn.Module
        The initialized language model.

    tokenizer : transformers.PreTrainedTokenizer
        The initialized tokenizer.

    device : torch.device
        The device for computations.

    instruction : str
        The instruction for guiding the agent.

    max_ans_tokens : int
        Maximum tokens for the generated answers.
    """

    def __init__(self, model, tokenizer, device, instruction, max_ans_tokens):
        # Initialize the agent with the given model, tokenizer, and parameters
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.instruction = instruction
        self.max_ans_tokens = max_ans_tokens

        try:
            # Set the model to generate end-of-sequence tokens
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        except AttributeError:
            pass

    def get_chat_template_input_ids(
        self,
        context,
        question,
        return_tensors=None,
    ):
        """
        Prepare input IDs for the model using a chat-based template.

        Parameters
        ----------
        context : str
            The context passage for answering the question.

        question : str
            The question to answer.

        return_tensors : str or None, optional
            Format of returned tensors ('pt' for PyTorch), default is None.

        Returns
        -------
        input_ids : torch.Tensor or list
            Input IDs for the model based on the context and question.
        """
        # Construct a message combining the instruction, context, and question
        instruction = self.instruction
        msg = f"Instruction: {instruction} Context: {context} Question: {question}"
        # Tokenize the message using a chat-based template
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}],
            add_generation_prompt=True,
            return_tensors=return_tensors,
        )
        return input_ids

    def get_answer(
        self,
        context,
        question,
        max_ans_tokens=None,
        verbose=False,
        return_n_tokens=False,
    ):
        """
        Generate an answer to the given question based on the context.

        Parameters
        ----------
        context : str
            The context passage for answering the question.

        question : str
            The question to answer.

        max_ans_tokens : int or None, optional
            Maximum number of tokens for the generated answer. Defaults to `self.max_ans_tokens`.

        verbose : bool, optional
            If True, print the context, question, and answer. Default is False.

        return_n_tokens : bool, optional
            If True, return the number of tokens in the answer along with the answer.

        Returns
        -------
        answer : str
            The generated answer.

        n_tokens : int, optional
            Number of tokens in the generated answer, if `return_n_tokens` is True.
        """
        # Retrieve the model, tokenizer, and device
        model, tokenizer, device = self.model, self.tokenizer, self.device

        # Set the maximum answer tokens if not explicitly provided
        if max_ans_tokens is None:
            max_ans_tokens = self.max_ans_tokens
        else:
            assert type(max_ans_tokens) == int, "max_ans_tokens should be an integer"

        # Tokenize the input context and question
        input_ids = self.get_chat_template_input_ids(
            context, question, return_tensors="pt"
        ).to(device)
        len_input = input_ids.shape[-1]

        # Generate the answer using the model
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_ans_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
            )

        # Extract the answer from the generated output
        answer_ids = outputs[0][len_input:]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

        # Optionally print the context, question, and answer for debugging
        if verbose:
            print(f"Context: {context}\nQuestion: {question}\nAnswer: {answer}")

        # Optionally return the number of tokens in the answer
        if return_n_tokens:
            n_tokens = len(answer_ids)
            return answer, n_tokens

        return answer

    @staticmethod
    def get_n_match(string, substring):
        """
        Count the number of occurrences of a substring within a string.

        Parameters
        ----------
        string : str
            The string to search within.

        substring : str
            The substring to count occurrences of.

        Returns
        -------
        count : int
            Number of occurrences of the substring.
        """
        all_starts = []
        start = 0
        while True:
            start = string.find(substring, start)
            if start == -1:
                break
            all_starts.append(start)
            start += 1  # Increment start to avoid overlapping matches
        return len(all_starts)

    def find_text_token_spans(self, input_ids, target_text, raise_if_not_found=True):
        """
        Locate spans of a target text within tokenized input.

        Parameters
        ----------
        input_ids : list of int
            Tokenized input as a 1D list of token IDs.

        target_text : str
            Target text to find within the tokenized input.

        raise_if_not_found : bool, optional
            If True, raise an error if the target text is not found.

        Returns
        -------
        spans : list of tuple
            List of (start, end) indices for each occurrence of the target text.
        """
        # Ensure input_ids is a list of integers
        assert (type(input_ids) == list) and (
            type(input_ids[0]) == int
        ), "input_ids should be a 1-d list, make sure it's not a tensor."

        # Decode input tokens to text and encode the target text into tokens
        tokenizer = self.tokenizer
        source = tokenizer.decode(input_ids)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        target = tokenizer.decode(target_ids)
        # Raise an error if the target text is not found
        if raise_if_not_found:
            assert target in source, f"'{target}' not found in input"
        # Initialize variables for finding spans
        n_match_left = self.get_n_match(source, target)
        spans = []
        start = 0

        while True:
            start += 1
            source_seg = tokenizer.decode(input_ids[start:])
            n_match_cur = self.get_n_match(source_seg, target)

            # If the number of matches decreases, start of a match is found
            if n_match_cur < n_match_left:
                assert (
                    n_match_left - n_match_cur == 1
                ), f"{n_match_left - n_match_cur} matches in a same token"
                n_match_left = n_match_cur
                start -= 1
                # Find the end of the match
                end = max(start + len(target_ids) - 5, start)
                while True:
                    end += 1
                    seg_text = tokenizer.decode(input_ids[start:end])
                    if target in seg_text:
                        break
                # Save the span and update the start position
                spans.append((start, end))
                start = end

            # Exit condition
            if n_match_left == 0 or start >= len(input_ids):
                break

        return spans

    def get_context_token_span(
        self,
        context,
        question,
    ):
        """
        Identify the token span of the context within the tokenized input.

        Parameters
        ----------
        context : str
            The context passage for answering the question.

        question : str
            The question to answer.

        Returns
        -------
        context_span : tuple of int
            A tuple (start, end) representing the token span of the context.
        """
        input_ids = self.get_chat_template_input_ids(
            context, question, return_tensors=None
        )
        context_spans = self.find_text_token_spans(input_ids, context)
        assert (
            len(context_spans) == 1
        ), f"Multiple/no context spans found: {context_spans}"
        return context_spans[0]

    def get_context_evidence_token_spans(
        self,
        context,
        question,
        evidence_list,
        allow_fuzzy=True,
    ):
        """
        Identify token spans for context and evidence sentences in the tokenized input.

        Parameters
        ----------
        context : str
            The context passage for answering the question.

        question : str
            The question to answer.

        evidence_list : list of str
            List of evidence sentences to locate in the tokenized input.

        allow_fuzzy : bool, optional
            If True, allow fuzzy matching of evidence sentences. Default is True.

        Returns
        -------
        context_span : tuple of int
            A tuple (start, end) representing the token span of the context.

        evidence_spans : list of tuple
            List of (start, end) indices for each evidence sentence found in the tokenized input.
        """
        # Tokenize the input context and question
        input_ids = self.get_chat_template_input_ids(
            context, question, return_tensors=None
        )

        # Retrieve the context span
        context_spans = self.find_text_token_spans(input_ids, context)
        assert (
            len(context_spans) == 1
        ), f"Multiple/no context spans found: {context_spans}"
        context_span = context_spans[0]

        # Retrieve the spans for each evidence sentence
        evidence_spans = []
        for evd_text in evidence_list:
            try:
                evidence_spans += self.find_text_token_spans(input_ids, evd_text)
            except AssertionError as e:
                if allow_fuzzy:
                    # Attempt to find a fuzzy match for evidence text in the context
                    evd_text_alt = re.search(f"({evd_text})", context).group(0)
                    if evd_text_alt is None:
                        print(f"No fuzzy match found for '{evd_text}' in '{context}'")
                        raise e
                    evidence_spans += self.find_text_token_spans(
                        input_ids, evd_text_alt
                    )
                else:
                    raise e
        return context_span, evidence_spans
