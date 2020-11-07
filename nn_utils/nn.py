import torch
import torch.nn as nn
from nn_utils.general import exp_mask, zero_mask, mask_2d_to_1d, mask_1d_to_2d


class BilinearCompFn(nn.Module):
    def __init__(self, in1_features, in2_features, rescale=True):
        super(BilinearCompFn, self).__init__()
        self._rescale = rescale
        self._linear = nn.Linear(in1_features, in2_features, bias=False)

    def forward(self, input1, input2):
        trans_input1 = self._linear(input1)  # [bs,sl,hn2]
        attn_scores = torch.bmm(  # bs,sl,sl
            trans_input1, torch.transpose(input2, -2, -1))
        if self._rescale:
            average_len = 0.5 * (input1.size(-2) + input2.size(-2))
            rescale_factor = average_len ** 0.5
            attn_scores /= rescale_factor
        return attn_scores


class BilinearAttn(nn.Module):
    def __init__(self, hn_q, hn_k):
        super(BilinearAttn, self).__init__()

        self._attn_comp = BilinearCompFn(hn_q, hn_k, rescale=True)
        self._attn_softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, q, k=None, v=None, attn_mask=None, **kwargs):
        assert attn_mask is not None
        k = q if k is None else k
        v = q if v is None else v

        g_attn_scores = self._attn_comp(q, k)  # bs,sl1,sl2
        g_attn_prob = self._attn_softmax(exp_mask(attn_mask, g_attn_scores))  # bs,sl1,sl2
        g_attn_res = torch.bmm(g_attn_prob, v)  # [bs,sl1,sl2]x[bs,sl2,hn] ==> [bs,sl,hn]
        g_attn_res = zero_mask(mask_2d_to_1d(attn_mask), g_attn_res, high_rank=True)
        return g_attn_res
