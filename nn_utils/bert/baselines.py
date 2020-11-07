from transformers.modeling_bert import BertForMaskedLM


class BertForOurMaskedLM(BertForMaskedLM):
    def forward(self, *args, **kwargs):
        outputs = super(BertForOurMaskedLM, self).forward(
            *args, **kwargs
        )
        masked_lm_loss = outputs[0]
        res_dict = {
            "loss_mlm": masked_lm_loss,
            "loss": masked_lm_loss,
        }
        return masked_lm_loss, res_dict

