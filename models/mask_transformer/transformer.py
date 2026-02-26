import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.tools import *
from models.mask_transformer.conditioning import ConditioningProvider
from torch.distributions.categorical import Categorical


# ──────────────────── Helper Modules ────────────────────

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2))  # [seqlen, bs, input_feats]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output


class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


# ──────────────────── Shared Base Class ────────────────────

class BaseCondTransformer(nn.Module):
    """Shared base for MaskTransformer and ResidualTransformer.

    Consolidates:
      - CLIP loading, freezing, encoding (with built-in finetune guard)
      - Condition encoding (text / action / uncond)
      - Condition masking (classifier-free guidance)
      - Weight initialization
      - Frame conditioning modules
      - Core transformer encoder
    """

    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, use_frames=False, frame_dim=512,
                 finetune_clip=False, finetune_clip_layers=2,
                 conditioning_mode='clip', num_id_samples=50,
                 t5_model_name='t5-base', **kargs):
        super().__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.use_frames = use_frames
        self.finetune_clip = finetune_clip
        self.finetune_clip_layers = finetune_clip_layers
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        # ── New: conditioning mode (clip / t5 / id_embedding) ──
        self.conditioning_mode = conditioning_mode
        self.num_id_samples = num_id_samples
        self.t5_model_name = t5_model_name
        self._use_new_provider = conditioning_mode in ('t5', 'id_embedding')
        print(f'Conditioning mode: {conditioning_mode}')

        if self._use_new_provider:
            self.cond_provider = ConditioningProvider(
                mode=conditioning_mode,
                latent_dim=latent_dim,
                clip_dim=clip_dim,
                num_samples=num_id_samples,
                t5_model_name=t5_model_name,
                cond_drop_prob=cond_drop_prob,
                device=getattr(opt, 'device', None),
            )

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        print(f'Frame conditioning: {use_frames}')
        if use_frames:
            print(f'  Frame dim: {frame_dim} -> latent_dim: {latent_dim}')

        # ── Core network layers ──
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, nhead=num_heads,
            dim_feedforward=ff_size, dropout=dropout, activation='gelu')
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # ── Condition embedding (legacy CLIP / action / uncond path) ──
        if not self._use_new_provider:
            if self.cond_mode == 'text':
                self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
            elif self.cond_mode == 'action':
                self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
            elif self.cond_mode == 'uncond':
                self.cond_emb = nn.Identity()
            else:
                raise KeyError("Unsupported condition mode!")
        else:
            # For new provider modes, cond_emb is inside ConditioningProvider
            # but we still keep a dummy for action/uncond fallback
            if self.cond_mode == 'action':
                self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)

        # ── Sparse keyframe conditioning ──
        if self.use_frames:
            self.modality_emb_frame = nn.Parameter(torch.randn(1, 1, self.latent_dim))
            print(f'  Sparse keyframe fusion: enabled')

    # ── CLIP ────────────────────────────────────────────────

    def _init_clip(self, clip_version):
        """Initialize CLIP model. Call AFTER self.apply(_init_weights)."""
        if self.cond_mode == 'text' and self.conditioning_mode == 'clip':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)
            if self.finetune_clip:
                print(f'CLIP Fine-tuning: ENABLED (last {self.finetune_clip_layers} layers)')
            else:
                print('CLIP Fine-tuning: DISABLED (fully frozen)')
        elif self._use_new_provider and self.conditioning_mode == 'clip':
            # Initialise CLIP via the ConditioningProvider
            self.cond_provider.init_clip(clip_version, device=getattr(self.opt, 'device', None))

    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(clip_version, device='cpu', jit=False)
        # Convert to FP16 only if NOT fine-tuning (fine-tuning requires FP32 for stability)
        if str(self.opt.device) != "cpu" and not self.finetune_clip:
            clip.model.convert_weights(clip_model)

        if self.finetune_clip:
            clip_model.eval()
            for p in clip_model.parameters():
                p.requires_grad = False
            total_layers = len(clip_model.transformer.resblocks)
            for i in range(total_layers - self.finetune_clip_layers, total_layers):
                for p in clip_model.transformer.resblocks[i].parameters():
                    p.requires_grad = True
            for p in clip_model.ln_final.parameters():
                p.requires_grad = True
            if hasattr(clip_model, 'text_projection') and clip_model.text_projection is not None:
                clip_model.text_projection.requires_grad = True
            print(f'Unfroze last {self.finetune_clip_layers} transformer layers + final LN + projection')
        else:
            clip_model.eval()
            for p in clip_model.parameters():
                p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        """Encode text via CLIP. Handles finetune_clip gradient guard internally."""
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        if self.finetune_clip:
            feat_clip_text = self.clip_model.encode_text(text).float()
        else:
            with torch.no_grad():
                feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def encode_condition(self, y, bs, device):
        """Encode condition vector from text / action / uncond.

        For new conditioning modes (t5, id_embedding), delegates to
        ConditioningProvider and returns a tuple:
            (cond_vector, force_mask)              — for clip / action / uncond
            (cond_vector, force_mask, cond_mask)   — for t5 (sequence condition)

        Returns:
            cond_vector: (B, clip_dim) or (B, T, t5_dim) or (B, num_actions) or (B, latent_dim)
            force_mask: bool -- True for uncond mode
            cond_mask: Optional (B, T) bool -- only for t5 mode
        """
        # ── New provider modes ──
        if self._use_new_provider:
            cond, cond_mask, force_mask = self.cond_provider.encode(y, bs, device)
            return cond, force_mask, cond_mask

        # ── Legacy clip / action / uncond path ──
        force_mask = False
        if self.cond_mode == 'text':
            cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.encode_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!")
        return cond_vector, force_mask, None

    # ── Shared utilities ───────────────────────────────────

    def mask_cond(self, cond, force_mask=False):
        """Apply CFG dropout. Works for (B, D) single-vector conditions."""
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        """Return parameters excluding frozen encoders (CLIP / T5)."""
        if self._use_new_provider:
            return self.cond_provider.parameters_wo_clip() + [
                p for n, p in self.named_parameters()
                if not n.startswith('cond_provider.')]
        if self.finetune_clip:
            return list(self.parameters())
        else:
            return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]


# ──────────────────── Mask Transformer ────────────────────

class MaskTransformer(BaseCondTransformer):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, use_frames=False, frame_dim=512,
                 finetune_clip=False, finetune_clip_layers=2,
                 conditioning_mode='clip', num_id_samples=50,
                 t5_model_name='t5-base', **kargs):
        super().__init__(
            code_dim, cond_mode, latent_dim=latent_dim, ff_size=ff_size,
            num_layers=num_layers, num_heads=num_heads, dropout=dropout,
            clip_dim=clip_dim, cond_drop_prob=cond_drop_prob,
            clip_version=clip_version, opt=opt, use_frames=use_frames,
            frame_dim=frame_dim, finetune_clip=finetune_clip,
            finetune_clip_layers=finetune_clip_layers,
            conditioning_mode=conditioning_mode,
            num_id_samples=num_id_samples,
            t5_model_name=t5_model_name, **kargs)

        # ── Mask-specific layers ──
        _num_tokens = opt.num_tokens + 2  # mask + pad dummies
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1

        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)
        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.apply(self._init_weights)
        self._init_clip(clip_version)

        self.noise_schedule = cosine_schedule

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(
            torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0))
        self.token_emb.requires_grad_(False)
        print("Token embedding initialized!")

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False,
                      frame_emb=None, has_frames=False, cond_mask=None):
        '''
        :param motion_ids: (b, seqlen)
        :param cond: (b, embed_dim)  OR  (b, T_text, t5_dim)  for t5 mode
        :param padding_mask: (b, seqlen), pad positions are TRUE
        :param frame_emb: (seqlen, b, latent_dim) optional
        :param cond_mask: (b, T_text) optional — T5 attention mask
        :return: logits (b, num_token, seqlen)
        '''
        x = self.token_emb(motion_ids)   # (b, seqlen, code_dim)
        x = self.input_process(x)        # (seqlen, b, latent_dim)
        x = self.position_enc(x)

        # Sparse keyframe fusion
        if self.use_frames and frame_emb is not None and has_frames:
            f = frame_emb + self.modality_emb_frame
            f = self.position_enc(f)
            x = x + f

        # ── Build prefix conditioning ──
        if self._use_new_provider:
            # ConditioningProvider handles projection + CFG masking
            prefix, prefix_kp, n_prefix = self.cond_provider.project_and_prepare(
                cond, cond_mask, force_mask=force_mask)
            # prefix: (N_prefix, B, D),  prefix_kp: (B, N_prefix) or None
        else:
            # Legacy CLIP path: project + CFG mask + unsqueeze
            cond = self.mask_cond(cond, force_mask=force_mask)
            prefix = self.cond_emb(cond).unsqueeze(0)    # (1, B, D)
            prefix_kp = None
            n_prefix = 1

        xseq = torch.cat([prefix, x], dim=0)             # (n_prefix+S, B, D)

        # Build key-padding mask for the full sequence
        if prefix_kp is not None:
            padding_mask = torch.cat([prefix_kp, padding_mask], dim=1)
        else:
            padding_mask = torch.cat(
                [torch.zeros_like(padding_mask[:, :1]).expand(-1, n_prefix), padding_mask], dim=1)

        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[n_prefix:]
        logits = self.output_process(output)  # (b, ntoken, seqlen)
        return logits

    def forward(self, ids, y, m_lens, frame_emb=None, has_frames=False, return_logits=False):
        '''
        :param ids: (b, n)
        :param y: raw text for text, (b,) for action, LongTensor for id_embedding
        :param m_lens: (b,)
        :param return_logits: if True, also returns logits tensor
        '''
        bs, ntokens = ids.shape
        device = ids.device

        non_pad_mask = lengths_to_mask(m_lens, ntokens)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        cond_vector, force_mask, cond_mask = self.encode_condition(y, bs, device)

        # BERT-style masking
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask &= non_pad_mask

        labels = torch.where(mask, ids, self.mask_id)
        x_ids = ids.clone()

        # 10% random replace
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        # 79.2% mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask,
                                    frame_emb=frame_emb, has_frames=has_frames,
                                    cond_mask=cond_mask)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        if return_logits:
            return ce_loss, pred_id, acc, logits
        return ce_loss, pred_id, acc

    def forward_with_cond_scale(self, motion_ids, cond_vector, padding_mask,
                                cond_scale=3, force_mask=False, cond_mask=None):
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask,
                                      force_mask=True, cond_mask=cond_mask)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask,
                                    cond_mask=cond_mask)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask,
                                        force_mask=True, cond_mask=cond_mask)
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self, conds, m_lens, timesteps: int, cond_scale: int,
                 temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False):

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        cond_vector, _, cond_mask = self.encode_condition(conds, batch_size, device)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device),
                                            reversed(range(timesteps))):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)

            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  cond_mask=cond_mask)
            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            temperature = starting_temperature
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                pred_ids = Categorical(probs).sample()

            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit(self, conds, tokens, m_lens, timesteps: int, cond_scale: int,
             temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False,
             edit_mask=None, padding_mask=None):

        assert edit_mask.shape == tokens.shape if edit_mask is not None else True
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        cond_vector, _, cond_mask = self.encode_condition(conds, tokens.shape[0], device)

        if padding_mask is None:
            padding_mask = ~lengths_to_mask(m_lens, seq_len)

        if edit_mask is None:
            mask_free = True
            ids = torch.where(padding_mask, self.pad_id, tokens)
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            scores = torch.where(edit_mask, 0., 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            ids = torch.where(edit_mask, self.mask_id, tokens)
            scores = torch.where(edit_mask, 0., 1e5)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device),
                                            reversed(range(timesteps))):
            rand_mask_prob = 0.16 if mask_free else self.noise_schedule(timestep)
            num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)

            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  cond_mask=cond_mask)
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            temperature = starting_temperature
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                pred_ids = Categorical(probs).sample()

            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~edit_mask, 1e5) if mask_free else scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit_beta(self, conds, conds_og, tokens, m_lens, cond_scale: int, force_mask=False):
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        cond_vector, _, cond_mask = self.encode_condition(conds, tokens.shape[0], device)
        if conds_og is not None:
            cond_vector_og, _, _ = self.encode_condition(conds_og, tokens.shape[0], device)
        else:
            cond_vector_og = None

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        ids = torch.where(padding_mask, self.pad_id, tokens)

        # NOTE: cond_vector_neg is passed but forward_with_cond_scale does not accept it.
        # This is preserved from the original code for future extension.
        logits = self.forward_with_cond_scale(ids,
                                              cond_vector=cond_vector,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)
        logits = logits.permute(0, 2, 1)

        probs_without_temperature = logits.softmax(dim=-1)
        tokens[tokens == -1] = 0
        og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1)).squeeze(-1)
        return og_tokens_scores


# ──────────────────── Residual Transformer ────────────────────

class ResidualTransformer(BaseCondTransformer):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 cond_drop_prob=0.1, num_heads=4, dropout=0.1, clip_dim=512,
                 shared_codebook=False, share_weight=False,
                 clip_version=None, opt=None, use_frames=False, finetune_clip=False,
                 finetune_clip_layers=2,
                 conditioning_mode='clip', num_id_samples=50,
                 t5_model_name='t5-base', **kargs):
        super().__init__(
            code_dim, cond_mode, latent_dim=latent_dim, ff_size=ff_size,
            num_layers=num_layers, num_heads=num_heads, dropout=dropout,
            clip_dim=clip_dim, cond_drop_prob=cond_drop_prob,
            clip_version=clip_version, opt=opt, use_frames=use_frames,
            finetune_clip=finetune_clip, finetune_clip_layers=finetune_clip_layers,
            conditioning_mode=conditioning_mode,
            num_id_samples=num_id_samples,
            t5_model_name=t5_model_name, **kargs)

        # ── Residual-specific layers ──
        self.encode_quant = partial(F.one_hot, num_classes=self.opt.num_quantizers)
        self.quant_emb = nn.Linear(self.opt.num_quantizers, self.latent_dim)

        _num_tokens = opt.num_tokens + 1  # one pad dummy
        self.pad_id = opt.num_tokens

        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)

        # ── Codebook weight schemes ──
        if shared_codebook:
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
            self.token_embed_weight = token_embed.expand(opt.num_quantizers - 1, _num_tokens, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(_num_tokens,)))
                self.output_proj_weight = output_proj.expand(opt.num_quantizers - 1, _num_tokens, code_dim)
                self.output_proj_bias = output_bias.expand(opt.num_quantizers - 1, _num_tokens)
        else:
            if share_weight:
                self.embed_proj_shared_weight = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, _num_tokens, code_dim)))
                self.token_embed_weight_ = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_weight_ = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_bias = None
                self.registered = False
            else:
                self.output_proj_weight = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 1, _num_tokens, code_dim)))
                self.output_proj_bias = nn.Parameter(
                    torch.zeros(size=(opt.num_quantizers, _num_tokens)))
                self.token_embed_weight = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 1, _num_tokens, code_dim)))

        self.shared_codebook = shared_codebook
        self.share_weight = share_weight

        self.apply(self._init_weights)
        self._init_clip(clip_version)

    def q_schedule(self, bs, low, high):
        noise = uniform((bs,), device=self.opt.device)
        schedule = 1 - cosine_schedule(noise)
        return torch.round(schedule * (high - low)) + low

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            device = next(self.parameters()).device
            self.output_proj_weight = torch.cat(
                [self.embed_proj_shared_weight, self.output_proj_weight_], dim=0).to(device)
            self.token_embed_weight = torch.cat(
                [self.token_embed_weight_, self.embed_proj_shared_weight], dim=0).to(device)

    def output_project(self, logits, qids):
        '''
        :logits: (bs, code_dim, seqlen)
        :qids: (bs)
        :return: logits (bs, ntoken, seqlen)
        '''
        output_proj_weight = self.output_proj_weight[qids]
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]

        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)
        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(-1)
        return output

    def trans_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False,
                      frame_emb=None, has_frames=False, cond_mask=None):
        '''
        :param motion_codes: (b, seqlen, d)
        :param qids: (b), quantizer layer ids
        :param cond: (b, embed_dim) or (b, T_text, t5_dim)
        :param padding_mask: (b, seqlen), pad positions are TRUE
        :param cond_mask: (b, T_text) optional — T5 attention mask
        :return: logits (b, code_dim, seqlen)
        '''
        x = self.input_process(motion_codes)  # (seqlen, b, latent_dim)
        q_onehot = self.encode_quant(qids).float().to(x.device)
        q_emb = self.quant_emb(q_onehot).unsqueeze(0)  # (1, b, latent_dim)

        x = self.position_enc(x)

        # Sparse keyframe fusion
        if self.use_frames and frame_emb is not None and has_frames:
            f = frame_emb + self.modality_emb_frame
            f = self.position_enc(f)
            x = x + f

        # ── Build prefix conditioning ──
        if self._use_new_provider:
            prefix, prefix_kp, n_prefix = self.cond_provider.project_and_prepare(
                cond, cond_mask, force_mask=force_mask)
        else:
            cond = self.mask_cond(cond, force_mask=force_mask)
            prefix = self.cond_emb(cond).unsqueeze(0)  # (1, b, latent_dim)
            prefix_kp = None
            n_prefix = 1

        xseq = torch.cat([prefix, q_emb, x], dim=0)  # (n_prefix+1+seqlen, b, latent_dim)
        n_strip = n_prefix + 1  # strip prefix + q_emb from output

        if prefix_kp is not None:
            padding_mask = torch.cat([
                prefix_kp,
                torch.zeros_like(padding_mask[:, :1]),  # q_emb is never padded
                padding_mask
            ], dim=1)
        else:
            padding_mask = torch.cat([
                torch.zeros_like(padding_mask[:, :1]).expand(-1, n_prefix + 1),
                padding_mask
            ], dim=1)

        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[n_strip:]
        logits = self.output_process(output)
        return logits

    def forward_with_cond_scale(self, motion_codes, q_id, cond_vector, padding_mask,
                                cond_scale=3, force_mask=False, cond_mask=None):
        bs = motion_codes.shape[0]
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                        force_mask=True, cond_mask=cond_mask)
            return self.output_project(logits, qids - 1)

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                    cond_mask=cond_mask)
        logits = self.output_project(logits, qids - 1)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                        force_mask=True, cond_mask=cond_mask)
        aux_logits = self.output_project(aux_logits, qids - 1)
        return aux_logits + (logits - aux_logits) * cond_scale

    def forward(self, all_indices, y, m_lens, frame_emb=None, has_frames=False):
        '''
        :param all_indices: (b, n, q)
        :param y: raw text or action labels
        :param m_lens: (b,)
        '''
        self.process_embed_proj_weight()

        bs, ntokens, num_quant_layers = all_indices.shape
        device = all_indices.device

        non_pad_mask = lengths_to_mask(m_lens, ntokens)
        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id)

        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
        gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
        all_codes = token_embed.gather(1, gather_indices)
        cumsum_codes = torch.cumsum(all_codes, dim=-1)

        active_indices = all_indices[torch.arange(bs), :, active_q_layers]
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        cond_vector, force_mask, cond_mask = self.encode_condition(y, bs, device)

        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask,
                                    force_mask, frame_emb=frame_emb, has_frames=has_frames,
                                    cond_mask=cond_mask)
        logits = self.output_project(logits, active_q_layers - 1)
        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)
        return ce_loss, pred_id, acc

    @torch.no_grad()
    @eval_decorator
    def generate(self, motion_ids, conds, m_lens, temperature=1,
                 topk_filter_thres=0.9, cond_scale=2, num_res_layers=-1):

        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        cond_vector, _, cond_mask = self.encode_condition(conds, batch_size, device)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0
        num_quant_layers = self.opt.num_quantizers if num_res_layers == -1 else num_res_layers + 1

        for i in range(1, num_quant_layers):
            token_embed = self.token_embed_weight[i - 1].to(device)
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask,
                                                  cond_scale=cond_scale, cond_mask=cond_mask)
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)
            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = torch.where(all_indices == self.pad_id, -1, all_indices)
        return all_indices

    @torch.no_grad()
    @eval_decorator
    def edit(self, motion_ids, conds, m_lens, temperature=1,
             topk_filter_thres=0.9, cond_scale=2):

        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        cond_vector, _, cond_mask = self.encode_condition(conds, batch_size, device)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0

        for i in range(1, self.opt.num_quantizers):
            token_embed = self.token_embed_weight[i - 1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask,
                                                  cond_scale=cond_scale, cond_mask=cond_mask)
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)
            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = torch.where(all_indices == self.pad_id, -1, all_indices)
        return all_indices