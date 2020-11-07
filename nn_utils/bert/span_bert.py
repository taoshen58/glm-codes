from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertLMPredictionHead, \
    BertForPreTraining, gelu, BertLayerNorm, BertOnlyMLMHead
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch


class SpanBertSboHead(nn.Module):
    def __init__(self, config):
        super(SpanBertSboHead, self).__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Linear 1
        self.linear1 = nn.Linear(3*config.hidden_size, config.hidden_size)
        self.layer_norm1 = BertLayerNorm(config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm2 = BertLayerNorm(config.hidden_size)

        # Prediction
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states, span_mask, position_ids=None):
        seq_length = span_mask.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=span_mask.device)
            position_ids = position_ids.unsqueeze(0).expand_as(span_mask)
        position_embeddings = self.position_embeddings(position_ids)  # bs,sl,hn

        # get span representation
        span_mask = (span_mask > -1)  # bs,sl

        fw_idxs = torch.zeros_like(span_mask, dtype=torch.long)  # bs,sl
        for _idx_col in range(1, span_mask.size()[1]):
            fw_idxs[:, _idx_col] = torch.where(
                span_mask[:, _idx_col] & (~ span_mask[:, _idx_col-1]),
                torch.full_like(fw_idxs[:, _idx_col-1], _idx_col - 1), fw_idxs[:, _idx_col-1],
            )

        bw_idxs = torch.full_like(span_mask, span_mask.size(1)-1, dtype=torch.long)  # bs,sl
        for _idx_col in range(span_mask.size(1) - 2, -1, -1):
            bw_idxs[:, _idx_col] = torch.where(
                span_mask[:, _idx_col] & ~ span_mask[:, _idx_col + 1],
                torch.full_like(bw_idxs[:, _idx_col+1], _idx_col + 1), bw_idxs[:, _idx_col+1],
            )

        fw_idxs = fw_idxs.unsqueeze(-1).expand_as(hidden_states)  # bs,sl,hn
        bw_idxs = bw_idxs.unsqueeze(-1).expand_as(hidden_states)  # bs,sl,hn

        fw_hidden_states = torch.gather(hidden_states, 1, fw_idxs)  # bs,sl,hn
        bw_hidden_states = torch.gather(hidden_states, 1, bw_idxs)  # bs,sl,hn

        sbo_rep = torch.cat([fw_hidden_states, bw_hidden_states, position_embeddings], dim=-1)
        sbo_rep = sbo_rep * span_mask.to(dtype=sbo_rep.dtype).unsqueeze(-1)  # bs,sl,3*hn

        mid_rep = self.layer_norm1(gelu(self.linear1(sbo_rep)).to(torch.float32))
        pre_logits = self.layer_norm2(gelu(self.linear2(mid_rep)).to(torch.float32))
        logits = self.decoder(pre_logits) + self.bias
        return logits


class SpanBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(SpanBertForPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        # self.mlm = BertLMPredictionHead(config)
        self.cls = BertOnlyMLMHead(config)
        self.sbo = SpanBertSboHead(config)

        self.apply(self.init_weights)

        # tie the weights of input and output
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)
        self._tie_or_clone_weights(self.sbo.decoder, self.bert.embeddings.word_embeddings)
        self._tie_or_clone_weights(self.sbo.position_embeddings, self.bert.embeddings.position_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                position_ids=None, head_mask=None, **kwargs):  # w/o the NSP loss
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)
        returns = (prediction_scores,)

        # For normal MLM
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        # For span boundary objective
        sbo_scores = self.sbo(sequence_output, span_mask=masked_lm_labels)  # bs,sl,vocab
        sbo_lm_loss = loss_fct(sbo_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        total_loss = masked_lm_loss + sbo_lm_loss
        # returns = (total_loss, masked_lm_loss, sbo_lm_loss, )
        res_dict = {
            "loss_mlm": masked_lm_loss,
            "loss_sbo": sbo_lm_loss,
            "loss": total_loss,
        }
        return total_loss, res_dict
