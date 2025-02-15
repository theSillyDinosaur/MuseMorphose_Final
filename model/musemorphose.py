import torch
from torch import nn
import torch.nn.functional as F
from transformer_encoder import VAETransformerEncoder
from transformer_helpers import (
  weights_init, PositionalEncoding, TokenEmbedding, generate_causal_mask
)

class VAETransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation='relu', cond_mode='in-attn'):
    super(VAETransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_seg_emb = d_seg_emb
    self.dropout = dropout
    self.activation = activation
    self.cond_mode = cond_mode

    if cond_mode == 'in-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
    elif cond_mode == 'pre-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb + d_model, d_model, bias=False)

    self.decoder_layers = nn.ModuleList()
    for i in range(n_layer):
      self.decoder_layers.append(
        nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
      )

  def forward(self, x, seg_emb):
    if not hasattr(self, 'cond_mode'):
      self.cond_mode = 'in-attn'
    attn_mask = generate_causal_mask(x.size(0)).to(x.device)
    # print (attn_mask.size())

    if self.cond_mode == 'in-attn':
      seg_emb = self.seg_emb_proj(seg_emb)
    elif self.cond_mode == 'pre-attn':
      x = torch.cat([x, seg_emb], dim=-1)
      x = self.seg_emb_proj(x)

    out = x
    for i in range(self.n_layer):
      if self.cond_mode == 'in-attn':
        out += seg_emb
      out = self.decoder_layers[i](out, src_mask=attn_mask)

    return out

class MuseMorphose(nn.Module):
  def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, 
    dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
    d_vae_latent, d_embed, n_token,
    enc_dropout=0.1, enc_activation='relu',
    dec_dropout=0.1, dec_activation='relu',
    d_composer_emb=32,
    n_composer_cls=13,
    is_training=True, use_attr_cls=True,
    cond_mode='in-attn',
    compound=False,
  ):
    print(n_token)
    super(MuseMorphose, self).__init__()
    self.enc_n_layer = enc_n_layer
    self.enc_n_head = enc_n_head
    self.enc_d_model = enc_d_model
    self.enc_d_ff = enc_d_ff
    self.enc_dropout = enc_dropout
    self.enc_activation = enc_activation

    self.dec_n_layer = dec_n_layer
    self.dec_n_head = dec_n_head
    self.dec_d_model = dec_d_model
    self.dec_d_ff = dec_d_ff
    self.dec_dropout = dec_dropout
    self.dec_activation = dec_activation  

    self.d_vae_latent = d_vae_latent
    self.n_token = n_token
    self.is_training = is_training

    self.cond_mode = cond_mode
    self.compound = compound
    if compound:
      assert d_embed % 4 == 0 and enc_d_model % 4 == 0 and dec_d_model % 4 == 0
      self.first_embs = TokenEmbedding(len(n_token)+1, d_embed//4, enc_d_model//4)
      self.seq_embs = nn.ModuleList([nn.ModuleList([TokenEmbedding(n_token[i][j], d_embed//4, enc_d_model//4)
                                                    for j in range(3)]) for i in range(len(n_token))])
      def token_emb_func(inp_tokens):
        first_emb = self.first_embs(inp_tokens[..., 0])
        seq_embs = [(inp_tokens[..., 0] == i).unsqueeze(-1)
                    * torch.cat([self.seq_embs[i][j](inp_tokens[..., i]) for j in range(3)], dim=-1) for i in range(len(n_token))]
        seq_emb = seq_embs[0]
        for i in range(len(n_token)):
          seq_emb += seq_embs[i]
        return torch.cat([first_emb, seq_emb], dim=-1)
      self.token_emb = token_emb_func
      self.first_out_proj = nn.Linear(dec_d_model//4, len(n_token)+1)
      self.seq_out_proj = nn.ModuleList([nn.ModuleList([nn.Linear(dec_d_model//4, n_token[i][j])
                                                        for j in range(3)]) for i in range(len(n_token))])
      def dec_out_proj_func(latent):
        split_latent = torch.split(latent, dec_d_model // 4, dim=-1)
        first_logits = self.first_out_proj(split_latent[0])
        seq_logits = [[self.seq_out_proj[i][j](split_latent[j+1]) for j in range(3)] for i in range(len(n_token))]
        return first_logits, seq_logits
      self.dec_out_proj = dec_out_proj_func
        
    else:
      self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
      self.dec_out_proj = nn.Linear(dec_d_model, n_token)
    self.d_embed = d_embed
    self.pe = PositionalEncoding(d_embed)
    self.encoder = VAETransformerEncoder(
      enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation
    )

    self.use_attr_cls = use_attr_cls
    if use_attr_cls:
      self.decoder = VAETransformerDecoder(
        dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_composer_emb,
        dropout=dec_dropout, activation=dec_activation,
        cond_mode=cond_mode
      )
    else:
      self.decoder = VAETransformerDecoder(
        dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent,
        dropout=dec_dropout, activation=dec_activation,
        cond_mode=cond_mode
      )

    if use_attr_cls:
      self.d_composer_emb = d_composer_emb
      self.composer_attr_emb = TokenEmbedding(n_composer_cls, d_composer_emb, d_composer_emb)
    else:
      self.composer_attr_emb = None

    self.emb_dropout = nn.Dropout(self.enc_dropout)
    self.apply(weights_init)
    

  def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.):
    std = torch.exp(0.5 * logvar).to(mu.device)
    if use_sampling:
      eps = torch.randn_like(std).to(mu.device) * sampling_var
    else:
      eps = torch.zeros_like(std).to(mu.device)

    return eps * std + mu

  def get_sampled_latent(self, inp, padding_mask=None, use_sampling=False, sampling_var=0.):
    token_emb = self.token_emb(inp)
    enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

    _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
    mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
    vae_latent = self.reparameterize(mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var)

    return vae_latent

  def generate(self, inp, dec_seg_emb, composer_cls=None, keep_last_only=True):
    token_emb = self.token_emb(inp)
    dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

    if composer_cls is not None:
      dec_composer_cls = self.composer_attr_emb(composer_cls)
      dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_composer_cls], dim=-1)
    else:
      dec_seg_emb_cat = dec_seg_emb

    out = self.decoder(dec_inp, dec_seg_emb_cat)
    out = self.dec_out_proj(out)

    if keep_last_only:
      if self.compound:
        for i in range(4):
          out[i] = out[i][-1, ...]
      else:
        out = out[-1, ...]

    return out


  def forward(self, enc_inp, dec_inp, dec_inp_bar_pos, composer_cls=None, padding_mask=None):
    # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
    enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
    enc_token_emb = self.token_emb(enc_inp)

    dec_token_emb = self.token_emb(dec_inp)

    enc_token_emb = enc_token_emb.reshape(
      enc_inp.size(0), -1, enc_token_emb.size(-1)
    )
    enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))
    dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))

    # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
    # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
    if padding_mask is not None:
      padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

    _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
    vae_latent = self.reparameterize(mu, logvar)
    vae_latent_reshaped = vae_latent.reshape(enc_bt_size, enc_n_bars, -1)

    dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(vae_latent.device)
    for n in range(dec_inp.size(1)):
      # [shape of dec_inp_bar_pos] (bsize, n_bars_per_sample + 1)
      # -- stores [[start idx of bar #1, sample #1, ..., start idx of bar #K, sample #1, seqlen of sample #1], [same for another sample], ...]
      for b, (st, ed) in enumerate(zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])):
        dec_seg_emb[st:ed, n, :] = vae_latent_reshaped[n, b, :]

    if composer_cls is not None:
      dec_composer_cls = self.composer_attr_emb(composer_cls)
      dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_composer_cls], dim=-1)
    else:
      dec_seg_emb_cat = dec_seg_emb

    dec_out = self.decoder(dec_inp, dec_seg_emb_cat)
    dec_logits = self.dec_out_proj(dec_out)

    return mu, logvar, dec_logits

  def compute_loss(self, mu, logvar, beta, fb_lambda, dec_logits, dec_tgt):
    if self.compound:
      first_logits, seq_logits = dec_logits
      first_recons = F.cross_entropy(
        first_logits.view(-1, first_logits.size(-1)), dec_tgt[..., 0].contiguous().view(-1), 
        ignore_index=0, reduction='mean'
      ).float()
      seq_recons = [[F.cross_entropy(
        seq_logits[i][j].view(-1, seq_logits[i][j].size(-1)),
        ((dec_tgt[..., 0] == i+1)*(dec_tgt[..., j+1])).contiguous().view(-1), 
        ignore_index=0, reduction='mean'
      ).float() for j in range(3)] for i in range(len(self.n_token))]
      recons_loss = first_recons
      for i in seq_recons:
        for j in i:
          recons_loss += torch.nan_to_num(j, nan=0)
    else:
      recons_loss = F.cross_entropy(
        dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
        ignore_index=self.n_token, reduction='mean'
      ).float()

    kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
    kl_before_free_bits = kl_raw.mean()
    kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
    kldiv_loss = kl_after_free_bits.mean()

    return {
      'beta': beta,
      'total_loss': recons_loss + beta * kldiv_loss,
      'kldiv_loss': kldiv_loss,
      'kldiv_raw': kl_before_free_bits,
      'recons_loss': recons_loss
    }