
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union, Dict

import torch
from transformers import PreTrainedTokenizer, GPT2Tokenizer


from utils import InputExample, get_verbalization_ids

import log
import wrapper as wrp

logger = log.get_logger('root')

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by PET. Each task requires its own
    custom implementation of a PVP.
    """

    def __init__(self, wrapper, pattern_id: int = 0, verbalizer_file: str = None, seed: int = 42):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)

        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)

        if self.wrapper.config.wrapper_type in [wrp.MLM_WRAPPER]:
            self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.wrapper.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true"

        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer
        parts_a, parts_b = self.get_parts(example)
        logger.info("### PVP ### :: parts_a = ", parts_a)
        logger.info("### PVP ### :: parts_b = ", parts_b)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        if priming:
            input_ids = tokens_a
            if tokens_b:
                input_ids += tokens_b
            if labeled:
                mask_idx = input_ids.index(self.mask_id)
                assert mask_idx >= 0, 'sequence of input_ids must contain a mask token'
                assert len(self.verbalize(example.label)) == 1, 'priming only supports one verbalization per label'
                verbalizer = self.verbalize(example.label)[0]
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                input_ids[mask_idx] = verbalizer_id
            return input_ids, []

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # print("in conver_mlm_logits_t_cls_logits", flush=True)
        # print("mlm_labels", mlm_labels, flush=True)
        # print("logits", logits, flush=True)
        # print(logits.size(), flush=True)
        # print("[mlm_labels >= 0]", [mlm_labels >= 0], flush=True)
        masked_logits = logits[mlm_labels >= 0]
        # print("masked_logits.size", masked_logits.size(), flush=True)
        # print("masked_logits", masked_logits)
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        # print("cls_logits here", cls_logits, flush=True)
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # print("m2c", m2c, flush=True)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)
        # print("filler_len", filler_len, flush=True)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        # print("cls_logits", cls_logits, flush=True)
        cls_logits = cls_logits * (m2c > 0).float()
        # print("cls_logits", cls_logits, flush=True)

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        # print("cls_logits", cls_logits, flush=True)
        return cls_logits

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        logger.info("Automatically loaded the following verbalizer: \n {}".format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize


class MnliPVP(PVP):
    VERBALIZER_A = {
        "contradiction": ["Wrong"],
        "entailment": ["Right"],
        "neutral": ["Maybe"]
    }
    VERBALIZER_B = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }
    VERBALIZER_C = {
        "contradiction": ["Yes"],
        "entailment": ["No"],
        "neutral": ["Maybe"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 1:
            return [text_a, "? ", text_b, ", ", self.mask], []
        elif self.pattern_id == 2:
            return [text_a, " entails ", text_b, ", ", self.mask], []
        elif self.pattern_id == 3:
            return ['"', text_a, '" entails "', text_b, '", ', self.mask], []
        elif self.pattern_id == 4:
            return [text_a, " Does it entail ", text_b, "? ", self.mask], []
        elif self.pattern_id == 5:
            return ['"', text_a, '" Does it entail "', text_b, '"? ', self.mask], []
        elif self.pattern_id == 6:
            return ['Does "', text_a, '" entail "', text_b, '"? ', self.mask], []
        elif self.pattern_id == 7:
            return ["Sentence A ", text_a, " Sentence B ", text_b, " Does sentence A entail sentence B ? ", self.mask], []
        elif self.pattern_id == 8:
            return [text_a, " contradicts ", text_b, ', ', self.mask], []
        elif self.pattern_id == 9:
            return ['"', text_a, '" contradicts "', text_b, '", ', self.mask], []
        elif self.pattern_id == 10:
            return ['sentence A: ', text_a, ' sentence B: ', text_b, self.mask], []

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0:
            return MnliPVP.VERBALIZER_A[label]
        elif self.pattern_id == 8 or self.pattern_id == 9:
            return MnliPVP.VERBALIZER_C[label]
        return MnliPVP.VERBALIZER_B[label]


class RtePVP(PVP):
    VERBALIZER = {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        # switch text_a and text_b to get the correct order
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b.rstrip(string.punctuation))

        if self.pattern_id == 0:
            return [text_a, ' ? ', text_b, " , ", self.mask], []
        if self.pattern_id == 1:
            return [text_a, ' ? ', text_b, " , ", self.mask], []
        elif self.pattern_id == 2:
            return [text_a, ' question: ', self.shortenable(example.text_b), ' True or False? answer:', self.mask], []
        elif self.pattern_id == 3:
            return ["sentence A: ", text_a, " sentence B: ", text_b, " Does sentence A entail sentence B ? ", self.mask], []
        elif self.pattern_id == 4:
            return [text_a, " entails ", text_b, " , ", self.mask], []
        elif self.pattern_id == 5:
            return ['"', text_a, '" entails "', text_b, '" , ', self.mask], []
        elif self.pattern_id == 6:
            return ['"', text_a, '" Does it entail "', text_b, '" ? ', self.mask], []
        elif self.pattern_id == 7:
            return ['Does "', text_a, '" entail "', text_b, '" ? ', self.mask], []
        elif self.pattern_id == 8:
            return ['sentence A: ', text_a, ' sentence B: ', text_b, self.mask], []


    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 1 or self.pattern_id == 2:
            return ['True'] if label == 'entailment' else ['False']
        return RtePVP.VERBALIZER[label]

class ColaPVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "cola"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    VERBALIZER_A = {
        "1": ["correct"],
        "0": ["incorrect"]
    }

    VERBALIZER_B = {
        "1": ["acceptable"],
        "0": ["unacceptable"]
    }

    VERBALIZER_C = {
        "1": ["True"],
        "0": ["False"]
    }

    VERBALIZER_D = {
        "1": ["Yes"],
        "0": ["No"]
    }

    def get_parts(self, example: InputExample):
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0 or self.pattern_id == 1:
            # this corresponds to the pattern [MASK]: a b
            return [text, 'This is', self.mask], []
        elif self.pattern_id == 2:
            return [text, 'The sentence is grammatical. True of False ?', self.mask], []
        elif self.pattern_id == 3:
            return [text, self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0:
            return ColaPVP.VERBALIZER_A[label]
        elif self.pattern_id == 1:
            return ColaPVP.VERBALIZER_B[label]
        elif self.pattern_id == 2:
            return ColaPVP.VERBALIZER_C[label]
        return ColaPVP.VERBALIZER_D[label]

class Sst2PVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "sst2"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    VERBALIZER_A = {
        "1": ["positive"],
        "0": ["negative"]
    }

    VERBALIZER_B = {
        "1": ["good"],
        "0": ["bad"]
    }

    VERBALIZER_C = {
        "1": ["great"],
        "0": ["terrible"]
    }

    def get_parts(self, example: InputExample):
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0 or self.pattern_id == 1 or self.pattern_id == 2:
            # this corresponds to the pattern [MASK]: a b
            return [text, ' It was ', self.mask], []
        elif self.pattern_id == 3 or self.pattern_id == 4 or self.pattern_id == 5:
            return [text, ' All in all, it was ', self.mask], []
        elif self.pattern_id == 6 or self.pattern_id == 7 or self.pattern_id == 8:
            return [text, ' The movie was ', self.mask], []
        elif self.pattern_id == 9:
            return [text, self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 3 or self.pattern_id == 6 or self.pattern_id == 9:
            return Sst2PVP.VERBALIZER_A[label]
        elif self.pattern_id == 1 or self.pattern_id == 4 or self.pattern_id == 7:
            return Sst2PVP.VERBALIZER_B[label]
        else:
            return Sst2PVP.VERBALIZER_C[label]


class QnliPVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "qnli"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    VERBALIZER_A = {
        "entailment": ["Yes"],
        "not_entailment": ["No"]
    }

    VERBALIZER_B = {
        "entailment": ["Right"],
        "not_entailment": ["Wrong"]
    }

    
    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 1:
            return [text_a, "? ", text_b, ", " , self.mask], []
        elif self.pattern_id == 2:
            return ["Is ", text_b, " the answer to ", text_a, "? ", self.mask], []
        elif self.pattern_id == 3:
            return ["Does ", text_b, " contain the answer to ", text_a, "? ", self.mask], []
        elif self.pattern_id == 4:
            return ["Question:", text_a, "Answer:", text_b, "Is the question answered? ", self.mask], []
        elif self.pattern_id == 5:
            return ['sentence A: ', text_a, ' sentence B: ', text_b, self.mask], []

            

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 1:
            return QnliPVP.VERBALIZER_B[label]
        else:
            return QnliPVP.VERBALIZER_A[label]


class MrpcPVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "mrpc"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    VERBALIZER_A = {
        "1": ["Yes"],
        "0": ["No"]
    }
    VERBALIZER_B = {
        "1": ["True"],
        "0": ["False"]
    }
    
    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [text_a, " ? ", text_b, " , ", self.mask], []
        elif self.pattern_id == 1:
            return [text_b, " ? ", text_a, " , ", self.mask], []
        elif self.pattern_id == 2:
            return [text_a, " is similar to ", text_b, " , ", self.mask], []  
        elif self.pattern_id == 3:
            return [text_a, " is similar to ", text_b, " True of False ? ", self.mask], []
        elif self.pattern_id == 4:
            return ['"', text_a, '\" \"', text_b, '" Are they semantically equivalent ? ', self.mask], []
        elif self.pattern_id == 5:
            return ['Is "', text_a, '" equivalent to "', text_b, '" ? ', self.mask], []
        elif self.pattern_id == 6:
            return ['Is "', text_b, '" a paraphrase of "', text_a, '" ? ', self.mask], []
        elif self.pattern_id == 7:
            return ['Is "', text_a, '" a paraphrase of "', text_b, '" ? ', self.mask], []
        elif self.pattern_id == 8:
            return ['sentence A: ', text_a, ' sentence B: ', text_b, self.mask], []

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 3:
            return MrpcPVP.VERBALIZER_B[label]
        return MrpcPVP.VERBALIZER_A[label]


class QqpPVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "qqp"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    VERBALIZER_A = {
        "1": ["Yes"],
        "0": ["No"]
    }
    VERBALIZER_B = {
        "1": ["True"],
        "0": ["False"]
    }
    
    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [text_a, " ? ", text_b, " , ", self.mask], []
        elif self.pattern_id == 1:
            return [text_b, " ? ", text_a, " , ", self.mask], []
        elif self.pattern_id == 2:
            return [text_a, " is similar to ", text_b, " , ", self.mask], []  
        elif self.pattern_id == 3:
            return [text_a, " is similar to ", text_b, " True of False ? ", self.mask], []
        elif self.pattern_id == 4:
            return ['"', text_a, '\" \"', text_b, '" Are they semantically equivalent ? ', self.mask], []
        elif self.pattern_id == 5:
            return ['Is "', text_a, '" equivalent to "', text_b, '" ? ', self.mask], []
        elif self.pattern_id == 6:
            return ['Is "', text_b, '" a paraphrase of "', text_a, '" ? ', self.mask], []
        elif self.pattern_id == 7:
            return ['Is "', text_a, '" a paraphrase of "', text_b, '" ? ', self.mask], []
        elif self.pattern_id == 8:
            return ['sentence A: ', text_a, ' sentence B: ', text_b, self.mask], []            

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 3:
            return QqpPVP.VERBALIZER_B[label]
        return QqpPVP.VERBALIZER_A[label]



PVPS = {
    'mnli': MnliPVP,
    'rte': RtePVP,
    'cola': ColaPVP,
    'sst-2': Sst2PVP,
    'qnli': QnliPVP,
    'mrpc': MrpcPVP,
    'qqp': QqpPVP,
}
