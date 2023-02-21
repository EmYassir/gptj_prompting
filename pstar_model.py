import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import GPT2Model, GPT2PreTrainedModel, GPTJModel, GPTJPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from prefix_encoder import PrefixEncoder

############### 
# p* model: prefix tuning / p-tuning v2 / p-tuning ... 
############### 

class GPT2PrefixForSequenceClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        for param in self.transformer.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        gpt2_param = 0
        for name, param in self.transformer.named_parameters():
            gpt2_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        prefix_param = all_param - gpt2_param
        print('all param is {}; gpt2 param is {}; prefix param is {}'.format(all_param, gpt2_param, prefix_param), flush=True)

        # # Model parallel
        # self.model_parallel = False
        # self.device_map = None

        # # Initialize weights and apply final processing
        # self.post_init()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.transformer.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # _, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.match_n_layer * 2, 
            self.match_n_head,
            self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)

        # print("past_key_values, ", past_key_values.__sizeof__(), flush=True)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.transformer.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                warnings.warn(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[range(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if not hasattr(self.config, 'problem_type') or self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GPTJPrefixForSequenceClassification(GPTJPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.transformer = GPTJModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        for param in self.transformer.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        gptj_param = 0
        for name, param in self.transformer.named_parameters():
            gptj_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        prefix_param = all_param - gptj_param
        print('all param is {}; gptj param is {}; prefix param is {}'.format(all_param, gptj_param, prefix_param), flush=True)

        # # Model parallel
        self.model_parallel = False
        self.device_map = None

        # # Initialize weights and apply final processing
        # self.post_init()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.transformer.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # _, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.match_n_layer * 2, 
            self.match_n_head,
            self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)

        # print("past_key_values, ", past_key_values.__sizeof__(), flush=True)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.transformer.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                warnings.warn(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[range(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if not hasattr(self.config, 'problem_type') or self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# class BertPrefixForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config
#         self.bert = BertModel(config)
#         self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

#         for param in self.bert.parameters():
#             param.requires_grad = False
        
#         self.pre_seq_len = config.pre_seq_len
#         self.n_layer = config.num_hidden_layers
#         self.n_head = config.num_attention_heads
#         self.n_embd = config.hidden_size // config.num_attention_heads

#         self.prefix_tokens = torch.arange(self.pre_seq_len).long()
#         self.prefix_encoder = PrefixEncoder(config)

#         bert_param = 0
#         for name, param in self.bert.named_parameters():
#             bert_param += param.numel()
#         all_param = 0
#         for name, param in self.named_parameters():
#             all_param += param.numel()
#         total_param = all_param - bert_param
#         print('total param is {}'.format(total_param)) # 9860105
    
#     def get_prompt(self, batch_size):
#         prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
#         past_key_values = self.prefix_encoder(prefix_tokens)
#         # bsz, seqlen, _ = past_key_values.shape
#         past_key_values = past_key_values.view(
#             batch_size,
#             self.pre_seq_len,
#             self.n_layer * 2, 
#             self.n_head,
#             self.n_embd
#         )
#         past_key_values = self.dropout(past_key_values)
#         past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
#         return past_key_values

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         batch_size = input_ids.shape[0]
#         past_key_values = self.get_prompt(batch_size=batch_size)
#         prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
#         attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             past_key_values=past_key_values,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# class RobertaPrefixForSequenceClassification(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config
#         self.roberta = RobertaModel(config)
#         self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()

#         for param in self.roberta.parameters():
#             param.requires_grad = False
        
#         self.pre_seq_len = config.pre_seq_len
#         self.n_layer = config.num_hidden_layers
#         self.n_head = config.num_attention_heads
#         self.n_embd = config.hidden_size // config.num_attention_heads

#         self.prefix_tokens = torch.arange(self.pre_seq_len).long()
#         self.prefix_encoder = PrefixEncoder(config)

#         bert_param = 0
#         for name, param in self.roberta.named_parameters():
#             bert_param += param.numel()
#         all_param = 0
#         for name, param in self.named_parameters():
#             all_param += param.numel()
#         total_param = all_param - bert_param
#         print('total param is {}'.format(total_param)) # 9860105

    
#     def get_prompt(self, batch_size):
#         prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
#         past_key_values = self.prefix_encoder(prefix_tokens)
#         past_key_values = past_key_values.view(
#             batch_size,
#             self.pre_seq_len,
#             self.n_layer * 2, 
#             self.n_head,
#             self.n_embd
#         )
#         past_key_values = self.dropout(past_key_values)
#         past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
#         return past_key_values

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         batch_size = input_ids.shape[0]
#         past_key_values = self.get_prompt(batch_size=batch_size)
#         prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
#         attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             past_key_values=past_key_values,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

