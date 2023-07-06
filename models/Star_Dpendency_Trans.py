import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.nn.parameter import Parameter
from torch.nn import init

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers import BertConfig
from  torch.nn import  functional as F
import copy

from transformers import   BertModel as Base_Bert

from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]




class Edge_Embeddings(nn.Module):
    def __init__(self, other_config):
        super().__init__()
        self.edge_embeddings = nn.Embedding.from_pretrained(torch.tensor(other_config.dependency_matrix, dtype=torch.float), freeze=False, padding_idx=0)
        self.edge_embeddings.weight.requires_grad = True
        self.reset_parameters()

    def forward(self,edge):
        batch,seq,_ = edge.shape
        edge = edge.reshape(batch,-1)
        # print(edge.shape)

        edge_embedding = self.edge_embeddings(edge)
        edge_embedding  =edge_embedding.reshape(batch,seq,seq,-1)
        return edge_embedding

    def reset_parameters(self):
        self.edge_embeddings.weight.data.copy_(torch.zeros(self.edge_embeddings.weight.size(0), self.edge_embeddings.weight.size(1)))

class EdgelizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(1, out_features))


        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, input):
        out_put = input.matmul(self.weight)+ self.bias
        # out_put = input.matmul(self.weight)

        return out_put


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config,asp_config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, \
            inputs_embeds=None, past_key_values_length=0,dependency_position_id =None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # print(input_ids)
            # exit()
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):



    def __init__(self, config, asp_config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.K_alpha = asp_config.K_alpha
        self.V_alpha = asp_config.V_alpha
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.multi_att_query = copy.deepcopy(self.query)
        self.multi_att_key = copy.deepcopy(self.key)
        self.multi_att_value= copy.deepcopy(self.value)


        # edge information
        self.e_k = EdgelizedLinear(asp_config.dependency_embed_dim , self.all_head_size)
        self.e_v = EdgelizedLinear(asp_config.dependency_embed_dim , self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def cycle_shift(self, e: torch.Tensor, forward=True):
        b, l, d = e.size()

        if forward:
            temp = e[:, -1, :]
            for i in range(l - 1):
                e[:, i + 1, :] = e[:, i, :]
            e[:, 0, :] = temp
        else:
            temp = e[:, 0, :]
            for i in range(1, l):
                e[:, i - 1, :] = e[:, i, :]
            e[:, -1, :] = temp

        return e

    def transpose_for_scores(self, x):
        if len(x.shape)==3:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            # print(new_x_shape)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            # return x.permute(0, 2, 1, 3)
            return x.permute(0,3,1,2,4)

    def forward(
        self,
        sentence_id,
        hidden_states,
        dependency_type_embedding,
        lengths=None,
        adj_index_undir_all=None,
        adj_dependency_index_all=None,
        batch_dependency_length=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        adj_dependency_edge_mask=None,
        star_presentation= None,
    ):
        # print("11111111111111111")
        # print(hidden_states.shape)
        #adj_dependency_index_all.shape : B*S*(max_dependency_num)   dependency index
        #adj_index_undir_all.shape : B*S*(max_dependency_num) #获取hidden state的dindex
        #adj_dependency_edge_mask.shape : B*1*S*(max_dependency_num+2)
        #dependency_type_embedding.shape :B*S*(max_dependency_num)*dependency_dim
        # print(adj_dependency_index_all.shape)
        # print(adj_index_undir_all.shape)
        # print(dependency_type_embedding.shape)
        # print(adj_dependency_edge_mask.shape)
        # exit()
        batch, seq, dim = hidden_states.shape








        if adj_dependency_edge_mask is not None :
            max_dependency_number = adj_dependency_edge_mask.shape[-1]

            h = hidden_states.clone()
            h_last, h_next = self.cycle_shift(h.clone(), False), self.cycle_shift(h.clone(), True)

            n_gram_representation = torch.cat(
                [h_last.unsqueeze(-2), h.unsqueeze(-2), h_next.unsqueeze(-2),],
                dim=-2)
            n_gram_representation  = n_gram_representation .view(batch * seq, -1, dim)
            h = h.unsqueeze(-2).view(batch * seq, -1, dim)

            n_gram_Q = self.query(h)

            n_gram_K = self.key(n_gram_representation )
            n_gram_V = self.value(n_gram_representation )
            n_gram_Q_layer = self.transpose_for_scores(n_gram_Q)
            n_gram_K_layer = self.transpose_for_scores(n_gram_K)
            n_gram_V_layer = self.transpose_for_scores(n_gram_V)
            n_gram_attention_scores = torch.matmul(n_gram_Q_layer, n_gram_K_layer.transpose(-1, -2))
            n_gram_attention_scores = n_gram_attention_scores / math.sqrt(self.attention_head_size)

            n_gram_attention_probs = nn.Softmax(dim=-1)(n_gram_attention_scores)

            n_gram_attention_probs = self.dropout(n_gram_attention_probs)


            n_gram_context_layer = torch.matmul(n_gram_attention_probs, n_gram_V_layer)
            n_gram_context_layer = n_gram_context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = n_gram_context_layer.size()[:-2] + (self.all_head_size,)
            n_gram_context_layer = n_gram_context_layer.view(*new_context_layer_shape)
            n_gram_context_representation = n_gram_context_layer.squeeze(-2).view(batch, seq, -1)




            h_repeat = torch.repeat_interleave(n_gram_context_representation.unsqueeze(1), dim=1, repeats=seq)
            # print(h_repeat.shape)
            gather_mask = torch.repeat_interleave(adj_index_undir_all.unsqueeze(-1), dim=-1, repeats=dim)
            gathered_data = torch.gather(h_repeat, 2, gather_mask)


            star_cat = torch.cat(
                [gathered_data, hidden_states.unsqueeze(-2),],
                dim=-2)
            star_cat = star_cat.view(batch * seq, -1, dim)
            n_gram_context_representation = n_gram_context_representation.unsqueeze(-2).view(batch * seq, -1, dim)

            # n_gram_context_representation
            # mixed_query_layer = self.multi_att_query(n_gram_context_representation)
            mixed_query_layer = n_gram_context_representation
            mixed_key_layer = star_cat
            mixed_value_layer = star_cat
            # mixed_key_layer = self.multi_att_key(star_cat)
            # mixed_value_layer = self.multi_att_value(star_cat)

            # print("-{0}-edge_states is not None-{1}-".format("-"*20,"-"*20))
            # print("duan dian 2")


            edge_k = self.e_k(dependency_type_embedding)
            edge_v = self.e_v(dependency_type_embedding)
            edge_k = edge_k.view(batch*seq,-1,dim)
            edge_v = edge_v.view(batch*seq,-1,dim)

            mixed_key_layer = mixed_key_layer +self.K_alpha*edge_k
            mixed_value_layer = mixed_value_layer +self.V_alpha*edge_v
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
            query_layer = self.transpose_for_scores(mixed_query_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            adj_dependency_edge_mask =adj_dependency_edge_mask.view(batch*seq,1,1,max_dependency_number)
            # print(attention_scores.shape)
            # print(adj_dependency_edge_mask.shape)
            attention_scores = attention_scores+ adj_dependency_edge_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)


            attention_probs = self.dropout(attention_probs)
            if sentence_id[0] =="33069925#747092#3":
                # print(n_gram_attention_probs)

                # n_gram_attention_mean = torch.mean(n_gram_attention_probs,dim=1).squeeze().cpu().numpy().tolist()
                # for i in n_gram_attention_mean:
                #     print(i)
                # print(n_gram_attention_probs.shape)
                # print(attention_probs.shape)

                attention_probs_mean = torch.mean(attention_probs,dim=1).squeeze().cpu().numpy().tolist()
                for i in attention_probs_mean:
                    print(i[0:-1])




            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            context_layer = context_layer.squeeze(-2).view(batch, seq, -1)

            star_presentation = None

            # star_presentation = star_presentation.unsqueeze(1)
            # m_c = torch.cat([star_presentation, context_layer], dim=1)
            #
            # star_que_layer = self.multi_att_query(star_presentation)
            #
            # star_key_layer = self.key(m_c)
            # star_value_layer = self.value(m_c)
            # star_que_layer = self.transpose_for_scores(star_que_layer)
            # star_key_layer = self.transpose_for_scores(star_key_layer)
            # star_value_layer = self.transpose_for_scores(star_value_layer)
            # star_attention_scores = torch.matmul(star_que_layer, star_key_layer.transpose(-1, -2))
            # star_attention_scores = star_attention_scores / math.sqrt(self.attention_head_size)
            # star_attention_probs = nn.Softmax(dim=-1)(star_attention_scores)
            #
            # star_attention_probs = self.dropout(star_attention_probs)
            # star_context_layer = torch.matmul(star_attention_probs, star_value_layer)
            # star_context_layer = star_context_layer.permute(0, 2, 1, 3).contiguous()
            # star_new_context_layer_shape = star_context_layer.size()[:-2] + (self.all_head_size,)
            # star_context_layer = star_context_layer.view(*star_new_context_layer_shape)
            # star_presentation = star_context_layer.squeeze(1)


        else:
            mixed_query_layer = self.query(hidden_states)

            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            mixed_query_layer = mixed_query_layer
            mixed_key_layer = mixed_key_layer
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
            query_layer = self.transpose_for_scores(mixed_query_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_scores = attention_scores.squeeze(-2) + attention_mask
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
                attention_probs = self.dropout(attention_probs)
                if head_mask is not None:
                    attention_probs = attention_probs * head_mask
                # print(attention_probs.shape)
                # print(value_layer.shape)
                # print(attention_probs.shape)
                # print(attention_probs.shape)
                # print(value_layer.shape)
                context_layer = torch.matmul(attention_probs, value_layer).squeeze(-2)
                # context_layer = torch.matmul(attention_probs, value_layer)
                # print(context_layer.shape)
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                # print(context_layer.shape)
                # context_layer = context_layer.permute(0,2,1, 3).contiguous()
                # print(context_layer.shape)
                # exit()

                new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
                context_layer = context_layer.view(*new_context_layer_shape)
                star_presentation = None

        # print(query_layer.shape)
        # print(key_layer.shape)



        # Take the dot product between "query" and "key" to get the raw attention scores.


        # print(attention_scores.shape)
        # print("*"*50)

        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        #     seq_length = hidden_states.size()[1]
        #     position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        #     position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        #     distance = position_ids_l - position_ids_r
        #     positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        #     positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
        #
        #     if self.position_embedding_type == "relative_key":
        #         relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #         attention_scores = attention_scores + relative_position_scores
        #     elif self.position_embedding_type == "relative_key_query":
        #         relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #         relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
        #         attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs,star_presentation


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, asp_config):
        super().__init__()
        self.self = BertSelfAttention(config, asp_config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        sentence_id,
        hidden_states,
        dependency_type_embedding=None,
        lengths=None,
        adj_index_undir_all=None,
        adj_dependency_index_all=None,
        batch_dependency_length=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        adj_dependency_edge_mask=None,
        star_presentation = None,
    ):
        self_outputs,star_presentation_out = self.self(
            sentence_id,
            hidden_states,
            dependency_type_embedding,
            lengths,
            adj_index_undir_all,
            adj_dependency_index_all,
            batch_dependency_length,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            adj_dependency_edge_mask=adj_dependency_edge_mask,
            star_presentation=star_presentation,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        if star_presentation_out is not None:
            star_presentation_out= self.output(star_presentation_out,star_presentation)
        else:
            star_presentation_out = None
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs,star_presentation_out


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, asp_config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config, asp_config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config, asp_config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        sentence_id=None,
        hidden_states=None,
        dependency_type_embedding=None,
        lengths=None,
        adj_index_undir_all=None,
        adj_dependency_index_all=None,
        batch_dependency_length=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        adj_dependency_edge_mask=None,
        star_presentation=None,

    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        self_attention_outputs,star_presentation_out = self.attention(
            sentence_id,
            hidden_states,
            dependency_type_embedding,
            lengths,
            adj_index_undir_all,
            adj_dependency_index_all,
            batch_dependency_length,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            adj_dependency_edge_mask=adj_dependency_edge_mask,
            star_presentation =star_presentation
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights


        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs,star_presentation_out

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, dependency_emb,pos_emb,config, asp_config):
        super().__init__()
        self.dependency_embedding = \
            torch.nn.Embedding(dependency_emb.shape[0], dependency_emb.shape[1], padding_idx=0, )
        # self.dependency_embedding.weight.data.copy_(torch.rand([dependency_emb.shape[0], dependency_emb.shape[1]]))
        self.dependency_embedding.weight.data.copy_(torch.from_numpy(dependency_emb))
        self.dependency_embedding.weight.requires_grad = True


        self.dropout_output = torch.nn.Dropout(0.1)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout_ops = torch.nn.Dropout(0.5)
        self.dropout_edge = torch.nn.Dropout(0.5)
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, asp_config) for _ in range(config.num_hidden_layers)])
        self.asp_config =asp_config


    def forward(
        self,
        sentence_id,
        hidden_states,
        lengths=None,
        adj_index_undir_all=None,
        adj_dependency_index_all=None,
        batch_dependency_length=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        adj_dependency_edge_mask=None
    ):
        dependency_type_embedding = self.dependency_embedding(adj_dependency_index_all)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        h = hidden_states.clone()
        star_presentation = F.avg_pool2d(h, (h.shape[1], 1)).squeeze(1)
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                #获得初始化 star_representaiton

                if i<=self.asp_config.SD_begin_layer:

                    layer_outputs,star_representation = layer_module(
                        sentence_id,
                        hidden_states,
                        dependency_type_embedding,
                        lengths,
                        adj_index_undir_all,
                        adj_dependency_index_all,
                        batch_dependency_length,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        adj_dependency_edge_mask=None,
                        star_presentation=None,
                    )

                else:
                    # print("111111111111111---------------------")
                    layer_outputs,star_presentation = layer_module(
                        sentence_id,
                        hidden_states,
                        dependency_type_embedding,
                        lengths,
                        adj_index_undir_all,
                        adj_dependency_index_all,
                        batch_dependency_length,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        adj_dependency_edge_mask=adj_dependency_edge_mask,
                        star_presentation=star_presentation,
                    )



            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)





BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, dependency_emb,pos_emb, config, asp_config, add_pooling_layer=True):
        super().__init__(config,asp_config)
        self.config = config
        self.embeddings = BertEmbeddings(config, asp_config)
        self.encoder = BertEncoder(dependency_emb,pos_emb,config, asp_config)


        self.init_weights()

    def init_personalized(self):
        self.edge_position_embeddings.reset_parameters()

        for layer in self.encoder.layer:
            layer.attention.self.e_q.reset_parameters()
            layer.attention.self.e_k.reset_parameters()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        sentence_id = None,
        input_ids=None,
        lengths=None,
        adj_index_undir_all=None,
        adj_dependency_index_all=None,
        adj_dependency_mask_all=None,
        batch_dependency_length=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # print(attention_mask[0])
        # print(attention_mask[0].shape)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        adj_dependency_edge_mask: torch.Tensor = self.get_extended_attention_mask(adj_dependency_mask_all, input_shape,device)




        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            sentence_id,
            embedding_output,
            lengths,
            adj_index_undir_all,
            adj_dependency_index_all,
            batch_dependency_length,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adj_dependency_edge_mask = adj_dependency_edge_mask,
        )
        sequence_output = encoder_outputs[0]   # 这里是取CLS



        if not return_dict:
            return (sequence_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )




@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):



    def __init__(self,config, **kwargs):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.args = kwargs['cus_config']
        dependency_emb, pos_emb = self.args.dependency_embedding, self.args.position_embedding
        self.position_embedding = \
            torch.nn.Embedding(pos_emb.shape[0], pos_emb.shape[1], padding_idx=0)
        self.position_embedding.weight.data.copy_(torch.from_numpy(pos_emb))
        self.position_embedding.weight.data.copy_(torch.rand([pos_emb.shape[0], pos_emb.shape[1]]))
        self.position_embedding.weight.requires_grad = True



        self.bert = BertModel(dependency_emb,pos_emb,config, asp_config=kwargs['cus_config'])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = torch.nn.Dropout(0.1)
        # self.bert.weight=False
        # self.bert.param.requires_grad=False
        # print(args.lstm_dim)
        self.feature_linear = torch.nn.Linear(self.args.bert_feature_dim * 2 +self.args.position_embed_dim + self.args.class_num * 3,
                                              self.args.bert_feature_dim * 2 + self.args.position_embed_dim)
        # self.feature_linear.weight=False
        self.cls_linear = torch.nn.Linear(self.args.bert_feature_dim * 2 + self. args.position_embed_dim, self.args.class_num)
        # print(args.bert_feature_dim * 2 + args.position_embed_dim)
        self.alpha_adjacent = self.args.alpha_adjacent
        self.trans_linear = torch.nn.Linear(self.args.class_num * 3, self.args.class_num)
        # self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)


        self.init_weights()

    def _cls_logits(self, features):
        features = self.dropout2(features)
        # exit()
        tags = self.cls_linear(features)
        return tags


    def multi_hops(self, features, mask, k):
        '''generate mask'''
        max_length = features.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = mask.unsqueeze(3).expand([-1, -1, -1, self.args.class_num])
        # mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])
        # print(features.shape)
        logits = self._cls_logits(features)
        left_logits = torch.zeros(logits.shape).to(self.args.device)
        up_logits = torch.zeros(logits.shape).to(self.args.device)
        dia_logits = torch.zeros(logits.shape).to(self.args.device)
        # print("1222222222222222222222")
        for i in range(k):
            probs = logits
            probs_T = probs.transpose(1, 2)
            logits = probs * mask
            # left_logits[:, 1:, :, :] = logits[:, :-1, :, :]
            # up_logits[:, :, 1:, :] = logits[:, :, :-1, :]
            # dia_logits[:, 1:, 1:, :] = logits[:, :-1, :-1, :]

            logits_a = torch.max(logits, dim=1)[0]
            logits_b = torch.max(logits, dim=2)[0]
            logits = torch.cat([logits_a.unsqueeze(3), logits_b.unsqueeze(3)], dim=3)
            logits = torch.max(logits, dim=3)[0]

            logits = logits.unsqueeze(2).expand([-1, -1, max_length, -1])
            logits_T = logits.transpose(1, 2)
            logits = torch.cat([logits, logits_T], dim=3)

            new_features = torch.cat([features, logits, probs], dim=3)
            features = self.feature_linear(new_features)
            logits = self.cls_linear(features)

            # print(left_logits.shape)
            # print(up_logits)
            # print(dia_logits)
            other_logits = self.trans_linear(torch.cat([left_logits, up_logits, dia_logits], dim=-1))
            # other_logits = other_logits*mask
            logits_output = (1 - self.alpha_adjacent) * logits + self.alpha_adjacent * other_logits
            logits = logits_output
        return logits

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        sentence_id=None,
        input_ids=None,
        lengths=None,
        syntactic_position_datas=None,
        adj_index_undir_all =None,
        adj_dependency_index_all=None,
        adj_dependency_mask_all=None,
        batch_dependency_length=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):


        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        """
        # print(adj_dependency_mask_all)
        # print("111111111111111111")
        # exit()

        #mask和dependency都需要扩展维度一个eh和s_m
        # #其中1为UNK
        # adj_dependency_index_all = F.pad(adj_dependency_index_all,(0,2),'constant',value=1)
        # # mask和dependency都需要扩展维度一个eh和s_m
        # adj_dependency_mask_all = F.pad(adj_dependency_mask_all,(0,2),'constant',value=1)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            sentence_id = sentence_id,
            input_ids = input_ids,
            lengths=lengths,
            adj_index_undir_all=adj_index_undir_all,
            adj_dependency_index_all=adj_dependency_index_all,
            adj_dependency_mask_all=adj_dependency_mask_all,
            batch_dependency_length=batch_dependency_length,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        seq_output = outputs[0]
        batch,seq,dim = seq_output.shape
        seq_output = self.dropout(seq_output)
        seq_output = seq_output.unsqueeze(2).expand([-1, -1,seq , -1])
        seq_output_T = seq_output.transpose(1,2)

        features = torch.cat([seq_output, seq_output_T], dim=-1)
        word_syntactic_position=self.position_embedding(syntactic_position_datas)
        concat_features = torch.cat([features, word_syntactic_position], dim=3)
        logits = self.multi_hops(concat_features, attention_mask, self.args.nhops)



        # print(aspect_output.shape)

        # aspect_output = self.dropout(aspect_output)

        # logits = self.classifier(torch.cat([aspect_output,pooled_output],dim=-1))


        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        loss = 0
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


