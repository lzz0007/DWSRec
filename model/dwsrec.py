import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
import math
from recbole.model.layers import VanillaAttention, MLPLayers, FeedForward
import numpy as np
import random

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class DWSRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.temperature = config['temperature']

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.fusion_type = config['fusion_type']
        self.trm_encoder = DIFTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=[self.hidden_size],
            feat_num=1,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length
        )

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

        # whitening
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        dim = self.plm_embedding.embedding_dim

        if config['layer_choice'] == 'linear':
            self.white_linear = nn.Linear(dim, self.hidden_size)
        elif config['layer_choice'] == 'mlp1':
            self.white_linear = MLPLayers([dim, self.hidden_size], 0.2, 'relu', init_method='norm')
        elif config['layer_choice'] == 'mlp3':
            self.white_linear = MLPLayers([dim, self.hidden_size, self.hidden_size, self.hidden_size], 0.2, 'relu', init_method='norm')
        elif config['layer_choice'] == 'moe':
            self.white_linear = MoEAdaptorLayer(8, [dim, self.hidden_size], 0.2)
        else:
            self.white_linear = MLPLayers([dim, self.hidden_size, self.hidden_size], 0.2, 'relu', init_method='norm')

        self.group, self.engine = config['group'], config['engine']
        if self.engine in ['svd', 'cholesky', 'pca_lowrank']:
            self.whiten_embeddings = self.batch_whitening(self.plm_embedding.weight, 1).to(self.device)
            self.whiten_embeddings_g1 = self.batch_whitening(self.plm_embedding.weight, self.group).to(self.device)
        elif self.engine == 'pw':
            self.moe_adaptor = MoEAdaptorLayer(2, [dim, self.hidden_size], 0.2)
        elif self.engine == 'bn':
            self.bn = nn.BatchNorm1d(dim, affine=False)
            self.whiten_embeddings = self.bn(self.plm_embedding.weight).to(self.device)
            self.whiten_embeddings_g1 = self.bn(self.plm_embedding.weight).to(self.device)

        # Fusion Attn
        self.attn_weights = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)

        self.attn_weights_seq = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn_seq = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.attn_seq, std=0.02)
        nn.init.normal_(self.attn_weights_seq, std=0.02)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_emb, feat_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        feature_table = feat_emb

        feature_emb = feature_table
        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, feature_emb, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # trm encoder
        if self.engine in ['svd', 'cholesky', 'pca_lowrank', 'bn']:
            whiten_embeddings = self.white_linear(self.whiten_embeddings)
            whiten_embeddings_g1 = self.white_linear(self.whiten_embeddings_g1)
            # whiten_embeddings_g1 = self.white_linear(self.plm_embedding.weight)
        elif self.engine == 'pw':
            whiten_embeddings = self.moe_adaptor(self.plm_embedding.weight)
            whiten_embeddings_g1 = self.moe_adaptor(self.plm_embedding.weight)
        seq_output = self.forward(
            item_seq,
            whiten_embeddings[item_seq],
            [whiten_embeddings_g1[item_seq]],
            item_seq_len)
        seq_output_whiten = self.forward(
            item_seq,
            whiten_embeddings_g1[item_seq],
            [whiten_embeddings[item_seq]],
            item_seq_len)

        # fuse seq emb
        mixed_x = torch.stack(
            (seq_output, seq_output_whiten), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights_seq.unsqueeze(0)) * self.attn_seq).sum(-1)
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_fused_emb = (mixed_x * score).sum(0)

        # fuse item emb
        mixed_x = torch.stack(
            (whiten_embeddings, whiten_embeddings_g1), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights.unsqueeze(0)) * self.attn).sum(-1)
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        fused_emb = (mixed_x * score).sum(0)

        # score
        seq_fused_emb = F.normalize(seq_fused_emb, dim=1)
        fused_emb = F.normalize(fused_emb, dim=1)

        logits = torch.matmul(seq_fused_emb, fused_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)

        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        if self.engine in ['svd', 'cholesky', 'pca_lowrank', 'bn']:
            whiten_embeddings = self.white_linear(self.whiten_embeddings)
            whiten_embeddings_g1 = self.white_linear(self.whiten_embeddings_g1)
            # whiten_embeddings_g1 = self.white_linear(self.plm_embedding.weight)
        elif self.engine == 'pw':
            whiten_embeddings = self.moe_adaptor(self.plm_embedding.weight)
            whiten_embeddings_g1 = self.moe_adaptor(self.plm_embedding.weight)

        # trm encoder
        seq_output = self.forward(
            item_seq,
            whiten_embeddings[item_seq],
            [whiten_embeddings_g1[item_seq]],
            item_seq_len)
        seq_output_whiten = self.forward(
            item_seq,
            whiten_embeddings_g1[item_seq],
            [whiten_embeddings[item_seq]],
            item_seq_len)

        # fuse seq emb
        mixed_x = torch.stack(
            (seq_output, seq_output_whiten), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights_seq.unsqueeze(0)) * self.attn_seq).sum(-1)
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_fused_emb = (mixed_x * score).sum(0)

        # fuse item emb
        mixed_x = torch.stack(
            (whiten_embeddings, whiten_embeddings_g1), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights.unsqueeze(0)) * self.attn).sum(-1)
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        fused_emb = (mixed_x * score).sum(0)

        seq_fused_emb = F.normalize(seq_fused_emb, dim=-1)
        fused_emb = F.normalize(fused_emb, dim=-1)

        scores = torch.matmul(seq_fused_emb, fused_emb.transpose(0, 1))  # [B n_items]
        return scores


    def get_item_embeddings(self): # to calculate conditioning per epoch
        if self.engine in ['svd', 'cholesky', 'pca_lowrank', 'bn']:
            whiten_embeddings = self.white_linear(self.whiten_embeddings)
            whiten_embeddings_g1 = self.white_linear(self.whiten_embeddings_g1)
        elif self.engine == 'pw':
            whiten_embeddings = self.moe_adaptor(self.plm_embedding.weight)
            whiten_embeddings_g1 = self.moe_adaptor(self.plm_embedding.weight)

        # fuse item emb
        mixed_x = torch.stack(
            (whiten_embeddings, whiten_embeddings_g1), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights.unsqueeze(0)) * self.attn).sum(-1)
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        fused_emb = (mixed_x * score).sum(0)
        return fused_emb[1:], F.normalize(fused_emb[1:], dim=-1)

    def batch_whitening(self, x, group, assignment=None):
        # code from "On feature decorrelation in self-supervised learning"
        N, D = x.shape
        # G = math.ceil(2 * D / N)
        G = group
        if assignment is not None:
            new_idx = assignment
        else:
            new_idx = torch.arange(D)
        x = x.t()[new_idx].t() # order the columns based on new_idx
        x = x.view(N, G, D // G)
        x = (x - x.mean(dim=0, keepdim=True)).transpose(0, 1)  # G, N, D//G
        covs = x.transpose(1, 2).bmm(x) / N
        W = transformation(covs, x.device, engine=self.engine)
        x = x.bmm(W)
        output = x.transpose(1, 2).flatten(0, 1)[torch.argsort(new_idx)].t()
        return output


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        # self.dropout = nn.Dropout(p=dropout)
        # self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(x)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


def transformation(covs, device, engine='symeig'):
    covs = covs.to(device)
    if engine == 'cholesky':
        C = torch.cholesky(covs.to(device))
        W = torch.triangular_solve(torch.eye(C.size(-1)).expand_as(C).to(C), C, upper=False)[0].transpose(1, 2).to(device)
    else:
        if engine == 'symeig':
            S, U = torch.symeig(covs.to(device), eigenvectors=True, upper=True)
        elif engine == 'svd':
            U, S, _ = torch.svd(covs.to(device))
        elif engine == 'svd_lowrank':
            U, S, _ = torch.svd_lowrank(covs.to(device))
        elif engine == 'pca_lowrank':
            U, S, _ = torch.pca_lowrank(covs.to(device), center=False)
        S, U = S.to(device), U.to(device)
        W = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1, 2))
    return W


class DIFTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(DIFTransformerEncoder, self).__init__()
        layer = DIFTransformerLayer(
            n_heads, hidden_size,attribute_hidden_size,feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,fusion_type,max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,attribute_hidden_states,position_embedding, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attribute_hidden_states,
                                                                  position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class DIFTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(DIFTransformerLayer, self).__init__()
        self.multi_head_attention = DIFMultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states,attribute_embed,position_embedding, attention_mask):
        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class DIFMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(DIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len*(2+self.feat_num), self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len,self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x,i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor,attribute_table,position_embedding, attention_mask):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))

        attribute_attention_table = []

        for i, (attribute_query, attribute_key) in enumerate(
                zip(self.query_layers, self.key_layers)):
            attribute_tensor = attribute_table[i].squeeze(-2)
            attribute_query_layer = self.transpose_for_scores_attribute(attribute_query(attribute_tensor),i)
            attribute_key_layer = self.transpose_for_scores_attribute(attribute_key(attribute_tensor),i)
            attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))
            attribute_attention_table.append(attribute_attention_scores.unsqueeze(-2))
        attribute_attention_table = torch.cat(attribute_attention_table,dim=-2)
        table_shape = attribute_attention_table.shape
        feat_atten_num, attention_size = table_shape[-2], table_shape[-1]
        if self.fusion_type == 'sum':
            attention_scores = torch.sum(attribute_attention_table, dim=-2)
            attention_scores = attention_scores + item_attention_scores + pos_scores
        elif self.fusion_type == 'concat':
            attention_scores = attribute_attention_table.view(table_shape[:-2] + (feat_atten_num * attention_size,))
            attention_scores = torch.cat([attention_scores, item_attention_scores, pos_scores], dim=-1)
            attention_scores = self.fusion_layer(attention_scores)
        elif self.fusion_type == 'gate':
            attention_scores = torch.cat(
                [attribute_attention_table, item_attention_scores.unsqueeze(-2), pos_scores.unsqueeze(-2)], dim=-2)
            attention_scores,_ = self.fusion_layer(attention_scores)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states