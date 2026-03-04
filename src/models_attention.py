import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_basic import BasicModel

class DotProductAttentionLayer(nn.Module):
    def __init__(self, name, enc_size, dec_size, hid_size):
        super().__init__()
        self.name = name
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.hid_size = hid_size

        self.W_q = nn.Linear(dec_size, hid_size)
        self.W_k = nn.Linear(enc_size, hid_size)
        self.W_v = nn.Linear(enc_size, enc_size)

    def forward(self, enc, dec, inp_mask):
        batch_size, ninp, _ = enc.shape

        Q = self.W_q(dec).unsqueeze(1)
        K = self.W_k(enc)
        V = self.W_v(enc)

        logits = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hid_size)

        if inp_mask is not None:
            mask = inp_mask.unsqueeze(1).to(dtype=torch.bool)
            logits = logits.masked_fill(~mask, float("-inf"))

        probs = F.softmax(logits, dim=-1).squeeze(1)
        attn = torch.bmm(probs.unsqueeze(1), V).squeeze(1)
        return attn, probs

class AttentiveModel(BasicModel):
    def __init__(self, name, inp_voc, out_voc, emb_size=64, hid_size=128, attn_size=128):
        nn.Module.__init__(self)
        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.hid_size = hid_size

        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)

        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True)
        self.dec_start = nn.Linear(hid_size, hid_size)
        self.attention = DotProductAttentionLayer("attn", hid_size, hid_size, attn_size)
        self.dec0 = nn.GRUCell(emb_size + hid_size, hid_size)
        self.logits = nn.Linear(hid_size, len(out_voc))

    def encode(self, inp, **flags):
        inp_emb = self.emb_inp(inp)
        enc_seq, last_state = self.enc0(inp_emb)
        dec_init = self.dec_start(last_state.squeeze(0))

        mask = self.inp_voc.compute_mask(inp)

        first_attn, first_attn_probs = self.attention(enc_seq, dec_init, mask)

        first_state = [dec_init, enc_seq, mask, first_attn_probs]
        return first_state

    def decode_step(self, prev_state, prev_tokens, **flags):
        prev_dec_state, enc_seq, mask, prev_attn_probs = prev_state
        emb = self.emb_out(prev_tokens)
        attn_vec, attn_probs = self.attention(enc_seq, prev_dec_state, mask)

        dec_inp = torch.cat([emb, attn_vec], dim=-1)
        new_dec_state = self.dec0(dec_inp, prev_dec_state)

        output_logits = self.logits(new_dec_state)

        new_state = [new_dec_state, enc_seq, mask, attn_probs]
        return new_state, output_logits