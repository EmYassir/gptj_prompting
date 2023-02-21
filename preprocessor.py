# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy

import numpy as np
from transformers import GPT2Tokenizer

from utils import InputFeatures, InputExample, random_sampling
from pvp import PVP, PVPS


class Preprocessor(ABC):
    """
    A preprocessor that transforms an :class:`InputExample` into a :class:`InputFeatures` object so that it can be
    processed by the model being used.
    """

    def __init__(self, wrapper, task_name, pattern_id: int = 0, verbalizer_file: str = None):
        """
        Create a new preprocessor.

        :param wrapper: the wrapper for the language model to use
        :param task_name: the name of the task
        :param pattern_id: the id of the PVP to be used
        :param verbalizer_file: path to a file containing a verbalizer that overrides the default verbalizer
        """
        self.wrapper = wrapper
        self.pvp = PVPS[task_name](self.wrapper, pattern_id, verbalizer_file)  # type: PVP
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}

    @abstractmethod
    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass


class MLMPreprocessor(Preprocessor):
    """Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT)."""

    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False, priming_num: int = 0, priming_description: str = "",
                           **kwargs) -> InputFeatures:

        selected_priming_data_count = None

        if priming:
            input_ids, token_type_ids = self.pvp.encode(example, priming=True)
            input_des_ids = []
            
            # if priming_description is provided, use a priming_description in front of examples
            if len(priming_description) > 0:
                tokenizer = self.wrapper.tokenizer
                priming_description_parts = priming_description.split(" ")
                kwargs = {'add_prefix_space': True} if isinstance(self.wrapper.tokenizer, GPT2Tokenizer) else {}
                parts_des = [x if isinstance(x, tuple) else (x, False) for x in priming_description_parts]
                parts_des = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_des if x]
                self.pvp.truncate(parts_des, None, max_length=self.wrapper.config.max_seq_length)
                input_des_ids = [token_id for part, _ in parts_des for token_id in part]

            rest_of_input_len = self.wrapper.config.max_seq_length - len(input_ids) - len(input_des_ids)
            priming_data = example.meta['priming_data']  # type: List[InputExample]

            # will use # of examples that is <= required priming examples and can fit into the rest of input len
            assert priming_num > 0

            # selected required number of priming examples randomly
            selected_priming_data = random_sampling(priming_data, priming_num)
            
            pre_priming_input_ids = []
            priming_input_ids = []
            selected_priming_data_count = 0
            for priming_example in selected_priming_data:
                pe_input_ids, _ = self.pvp.encode(priming_example, priming=True, labeled=True)
                pre_priming_input_ids += pe_input_ids
                # if sentences are longer than required_max_length, do not append any more example
                if len(pre_priming_input_ids) <= rest_of_input_len:
                    priming_input_ids = deepcopy(pre_priming_input_ids)
                    selected_priming_data_count += 1
                else:  # if selected priming input ids have passed required sentence length, do not add more priming input
                    break

            input_ids = input_des_ids + priming_input_ids + input_ids
            token_type_ids = self.wrapper.tokenizer.create_token_type_ids_from_sequences(input_ids)
            input_ids = self.wrapper.tokenizer.build_inputs_with_special_tokens(input_ids)
        else:
            input_ids, token_type_ids = self.pvp.encode(example)

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100
        logits = example.logits if example.logits else [-1]

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
            if self.wrapper.config.model_type == 'gpt2' or self.wrapper.config.model_type == 'EleutherAI/gpt-j-6B':
                # shift labels to the left by one
                mlm_labels.append(mlm_labels.pop(0))
        else:
            mlm_labels = [-1] * self.wrapper.config.max_seq_length

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, logits=logits, idx=example.idx, meta={"priming_used_example_num": selected_priming_data_count})


class SequenceClassifierPreprocessor(Preprocessor):
    """Preprocessor for a regular sequence classification model."""

    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:
        inputs = self.wrapper.task_helper.get_sequence_classifier_inputs(example) if self.wrapper.task_helper else None
        if inputs is None:
            inputs = self.wrapper.tokenizer.encode_plus(
                example.text_a if example.text_a else None,
                example.text_b if example.text_b else None,
                add_special_tokens=True,
                max_length=self.wrapper.config.max_seq_length,
            )
        input_ids, token_type_ids = inputs["input_ids"], inputs.get("token_type_ids")

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        if not token_type_ids:
            token_type_ids = [0] * self.wrapper.config.max_seq_length
        else:
            token_type_ids = token_type_ids + ([0] * padding_length)
        mlm_labels = [-1] * len(input_ids)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100
        logits = example.logits if example.logits else [-1]

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, logits=logits, idx=example.idx)
