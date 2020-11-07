import torch
import torch.nn as nn
from transformers import RobertaConfig, BertPreTrainedModel, RobertaModel, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_bert import (BertConfig, BertEmbeddings,
                                                BertLayerNorm, BertModel,
                                                BertPreTrainedModel, gelu, ACT2FN)
from transformers.modeling_roberta import RobertaLMHead
from nn_utils.general import transform_edges_to_mask, mask_2d_to_1d, zero_mask, exp_mask, prime_to_attn_2d_mask,\
    masked_pool, slice_tensor, reversed_slice_tensor, len_mask
from nn_utils.graph import get_node_rep_from_context, combine_two_sequence, gather_masked_node_reps
from src_lm.data.dataset_line import MAX_NODE_CTX_LEN


class RobertaForGLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    MAX_SEQ_LENGTH = 512
    MAX_NODE_LENGTH = MAX_NODE_CTX_LEN

    def __init__(self, config):
        super(RobertaForGLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.bilinear = nn.Bilinear(config.hidden_size, config.hidden_size, 1)

        self.loss_lambda = getattr(config, "loss_lambda", 1.)
        self.disable_rev_pos = getattr(config, "disable_rev_pos", False)
        self.padding_idx = self.roberta.embeddings.padding_idx

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head.decoder, self.roberta.embeddings.word_embeddings)

    def forward(
            self, input_ids, node_ctx_pos_ids, node_ctx_neg_ids, node_ses, node_masked_flag,
            token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            position_ids=None,
            head_mask=None, **kwargs):
        # print(input_ids.shape, node_ctx_pos_ids.shape,
        #       node_ses.shape[:-1], node_masked_flag.shape)

        bs, sl = input_ids.shape

        # main roberta
        outputs = self.roberta(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask, head_mask=head_mask, attention_mask_2d=None)
        sequence_output = outputs[0]  # [bs,sl,hn]

        # graph loss
        # 0) node_ses <- [bs,nctk,nl,2]; node_masked_flag <- [bs,nctk];

        # 1) get node representaitons [bs,nctk,hn]
        ctk_rep, ctk_mask = get_node_rep_from_context(sequence_output, node_ses)
        # 2) ADD EXTRA LAYER HERE

        # 3) node input: node_ctx_ids [bs,ml2,ncsl], where ml2 = 2 x ml
        node_masked_flag = (node_masked_flag > 0).to(torch.long)  # [bs,nctk] in {-1,0,1} -> {0,1}
        node_masked_lens = node_masked_flag.sum(-1)  # [bs]
        new_ml = node_masked_lens.max().item()

        # # X) generate node_ctx_ids <-[ns,ml2,ncsl]
        node_ctx_ids = combine_two_sequence(node_ctx_pos_ids, node_ctx_neg_ids, new_ml, self.padding_idx)

        bs, ml2, ncsl = node_ctx_ids.shape
        assert ml2 % 2 == 0
        ml = ml2//2

        assert new_ml == ml or new_ml == 0, "{}/{}".format(new_ml, ml)   # if there is no node appearing in
        node_masked_mask, _ = len_mask(node_masked_lens, ml)  # [bs,ml]

        node_ctx_ids_rsp = node_ctx_ids.contiguous().view(-1, ncsl)  # [bs*ml2, ncsl]
        node_ctx_mask_rsp = (node_ctx_ids_rsp != self.padding_idx).to(torch.long)  # [bs*ml2, ncsl]
        if self.base_model_prefix == "roberta":  # roberta compatible
            node_ctx_ids_rsp[:, 0] = 0
        if self.disable_rev_pos:
            node_ctx_pos_ids = None
        else:
            node_ctx_pos_ids = self.generate_reversed_positional_ids(node_ctx_ids_rsp)
        node_ctx_outputs = self.roberta(
            node_ctx_ids_rsp,
            position_ids=node_ctx_pos_ids,
            attention_mask=node_ctx_mask_rsp)
        node_ctx_rep_rsp = node_ctx_outputs[1]  # [bs*ml2,hn] get the pooled output
        node_ctx_rep = node_ctx_rep_rsp.view(bs, ml2, node_ctx_rep_rsp.shape[-1])  # [bs,ml2,hn]

        # 4) get target ctk_rep from ctk_rep: [bs,nctk,hn]&[bs,nctk] -> [bs,ml,hn]
        node_pool_rep = gather_masked_node_reps(node_masked_lens, ml, node_masked_flag, node_masked_mask, ctk_rep)

        # 6) similarity between `node_ctx_rep` and `node_pool_rep`
        similarity_scores = self.bilinear(
            torch.cat([node_pool_rep, node_pool_rep], dim=1), node_ctx_rep).squeeze(-1)  # bs,ml2
        pos_similarity_scores, neg_similarity_scores = similarity_scores[:, :ml], similarity_scores[:, ml:]

        margin_val = 1.
        gpp_losses_list = torch.relu(margin_val - pos_similarity_scores + neg_similarity_scores)  # [bs,ml]
        gpp_losses, gpp_mask = masked_pool(  # [bs] - [bs]
            gpp_losses_list, node_masked_mask, high_rank=False, method="mean", return_new_mask=True)
        gpp_loss = masked_pool(gpp_losses, gpp_mask, high_rank=False, method="mean")

        # MLM loss
        mlm_score_bert = self.lm_head(sequence_output)
        loss_fct_xentropy = nn.CrossEntropyLoss(ignore_index=-1)
        mlm_loss = loss_fct_xentropy(
            mlm_score_bert.view(-1, self.config.vocab_size),  #
            masked_lm_labels.view(-1)
        )
        total_loss = mlm_loss + self.loss_lambda * gpp_loss

        # store
        res_dict = {
            "loss_mlm": mlm_loss,
            "loss_gpp": gpp_loss,
            "loss": total_loss,
        }

        return total_loss, res_dict

    def generate_reversed_positional_ids(self, node_ids):
        start_index = self.MAX_SEQ_LENGTH - self.MAX_NODE_LENGTH
        seq_length = node_ids.size(1)
        assert seq_length <= self.MAX_NODE_LENGTH

        position_ids = torch.arange(self.padding_idx + 1 + start_index,
                                    seq_length + self.padding_idx + 1 + start_index,
                                    dtype=torch.long,
                                    device=node_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(node_ids)
        return position_ids
