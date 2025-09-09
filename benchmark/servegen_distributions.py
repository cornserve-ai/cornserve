#!/usr/bin/env python3
"""
Example: Generate workloads for multimodal and reasoning models.

This script demonstrates how to:
1. Generate multimodal workloads with image/audio/video tokens
2. Generate reasoning workloads with reason_ratio field
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from servegen.construct import Request

# Video and Audio sampling functions
def _trunc(draw_fn, size: int, lo: float, hi: float) -> np.ndarray:
    out = np.empty(size)
    i = 0
    while i < size:
        x = draw_fn(size - i)
        x = x[(x >= lo) & (x <= hi)]
        if x.size == 0:
            continue
        k = min(x.size, size - i)
        out[i:i+k] = x[:k]
        i += k
    return out

def _trunc_normal(rng: np.random.Generator, loc: float, scale: float, size: int, lo: float, hi: float) -> np.ndarray:
    return _trunc(lambda n: rng.normal(loc, scale, n), size, lo, hi)

def _trunc_lognormal(rng: np.random.Generator, median: float, sigma: float, size: int, lo: float, hi: float) -> np.ndarray:
    mu = np.log(median)
    return _trunc(lambda n: rng.lognormal(mu, sigma, n), size, lo, hi)

@dataclass
class MixComp:
    kind: str          # "normal" or "lognormal"
    w: float
    p1: float          # loc or median
    p2: float          # scale or sigma

@dataclass
class MixPreset:
    comps: List[MixComp]
    clip: Tuple[float, float]

def sample_mixture(n: int, preset: MixPreset, seed: int):
    rng = np.random.default_rng(seed)
    ws = np.array([c.w for c in preset.comps], float)
    ws /= ws.sum()
    idx = rng.choice(len(preset.comps), size=n, p=ws)
    lo, hi = preset.clip
    out = np.empty(n)
    for k, c in enumerate(preset.comps):
        m = (idx == k)
        sz = int(m.sum())
        if sz == 0: continue
        if c.kind == "normal":
            out[m] = _trunc_normal(rng, c.p1, c.p2, sz, lo, hi)
        elif c.kind == "lognormal":
            out[m] = _trunc_lognormal(rng, c.p1, c.p2, sz, lo, hi)
        else:
            raise ValueError(f"Unknown mixture component kind: {c.kind}")
    return out.astype(int)

# Video
BIMODAL_OVERALL = MixPreset(
    comps=[
        MixComp("lognormal", 0.35, 3000, 0.27),
        MixComp("lognormal", 0.65, 10000, 0.28),
    ],
    clip=(1200, 26000),
)

def sample_video_tokens_bimodal(n: int, seed: int):
    return sample_mixture(n, BIMODAL_OVERALL, seed)

def tokens_to_video_resolution2(n_tokens, aspect_ratio=16/9, max_frames_pref=(4,6,8,10,12,16,20,24,28,32)):
    token_frames, token_pix = 2, 28
    cand_t = []
    for F in max_frames_pref:
        t = F // token_frames
        if t and n_tokens % t == 0:
            cand_t.append(t)
    if not cand_t:
        for t in range(1, int(n_tokens**(1/3))*2 + 3):
            if n_tokens % t == 0:
                cand_t.append(t)
        if not cand_t: cand_t = [1]

    cands = []
    best, best_err = None, 1e9
    for t in cand_t:
        s = n_tokens // t
        r = int(math.isqrt(s))
        for h in range(1, r+1):
            if s % h:
                continue
            w = s // h
            err = abs(w/h - aspect_ratio)
            cands.append((err, max(h, w), t, h, w))
            if err < best_err or (abs(err-best_err)<1e-12 and max(h,w) < max(best[3],best[4])):
                best_err = err
                best = (err, max(h, w), t, h, w)

    # Filter out weird aspect ratios and giant dimensions
    cands.sort(key=lambda x: (x[1], x[0]))

    def sane(t, h, w):
        H = h * token_pix
        W = w * token_pix
        return t > 1 and H <= 8192 and W <= 8192

    # print(cands)

    for _, _, t, h, w in cands:
        if sane(t, h, w):
            return t*token_frames, h*token_pix, w*token_pix

    _, _, t_tok, h_tok, w_tok = best
    return t_tok*token_frames, h_tok*token_pix, w_tok*token_pix


def tokens_to_video_resolution(n_tokens, aspect_ratio=16/9, max_frames_pref=(4,6,8,10,12),
                               *, token_frames=2, token_pix=28,
                               min_dim=224, max_dim=4096,      # pixel bounds for sanity
                               ar_min=0.5, ar_max=2.0,         # ~9:16..16:9-ish
                               max_rel_tok_err=0.20,           # allow up to 20% token mismatch
                               neighbor_h=12,
                               max_frames=32):                 # hard cap on frames
    assert n_tokens > 0
    min_tok_dim = max(1, math.ceil(min_dim / token_pix))
    max_tok_dim = max(1, max_dim // token_pix)

    # candidate temporal tokens (t_tok): from frame prefs and around cube root
    T=set()
    for F in max_frames_pref:
        t=max(1, F//token_frames)
        T.update({t, max(1,t-1), t+1})
    t0=max(1, round(n_tokens ** (1/3)))
    T.update({max(1,t0+d) for d in range(-3,4)})
    T=sorted(T)

    def sane(H,W,frames):
        ar=W/H
        return (ar_min<=ar<=ar_max
                and min_dim<=H<=max_dim
                and min_dim<=W<=max_dim
                and frames<=max_frames)

    cands=[]
    for t_tok in T:
        s_target = n_tokens / t_tok
        h0 = int(round((s_target / aspect_ratio) ** 0.5))
        h0 = min(max(h0, min_tok_dim), max_tok_dim)
        for dh in range(-neighbor_h, neighbor_h+1):
            h = h0 + dh
            if h < min_tok_dim or h > max_tok_dim: continue
            ws = {
                max(min_tok_dim, min(max_tok_dim, int(round(h * aspect_ratio)))),
                max(min_tok_dim, min(max_tok_dim, int(round(s_target / h))))
            }
            for w in ws:
                H, W = h*token_pix, w*token_pix
                frames = t_tok * token_frames
                if not sane(H,W,frames): continue
                tok = t_tok * h * w
                rel_err = abs(tok - n_tokens) / n_tokens
                if rel_err > max_rel_tok_err: continue
                ar_err = abs((W/H) - aspect_ratio) / aspect_ratio
                frame_pref_err = min((abs(frames-F) for F in max_frames_pref), default=0)
                score = (rel_err, ar_err, max(W,H), frame_pref_err)
                cands.append((score, (frames, H, W)))

    if cands:
        cands.sort(key=lambda x: x[0])
        print(f"Video tokens {n_tokens} -> {cands[0][1]}")
        return cands[0][1]

    # fallback
    print(f"[Fallback] Video tokens {n_tokens} -> {min_dim}x{int(min_dim*aspect_ratio)}")
    return token_frames, min_dim, int(min_dim*aspect_ratio)


# Audio
AUDIO_MIXTURE = MixPreset(
    comps=[
        MixComp("lognormal", 0.486, 205.0, 0.18),
        MixComp("normal",   0.433, 290.0, 6.2),
        MixComp("lognormal", 0.081, 680.0, 0.12),
    ],
    clip=(120, 900),
)

def sample_audio_tokens(n: int, seed: int):
    return sample_mixture(n, AUDIO_MIXTURE, seed)

def tokens_to_audio_sec(n_tokens: int) -> float:
    return n_tokens / 25  # Alibaba docs say 25 tokens/sec


def add_update_multimodal_content(
    requests: list[Request],
    no_image_prob: float,
    video_prob: float,
    audio_prob: float,
    seed: int,
) -> list[Request]:
    """Add video and audio data to requests since ServeGen doesn't have real video/audio traces."""
    rng = np.random.default_rng(seed)

    # Remove images based on no_image_prob
    for req in requests:
        if rng.random() < no_image_prob:
            req.data['image_tokens'] = []
            req.data['image_resolution'] = []
    
    for req in requests:
        num_videos = rng.choice([0, 1], p=[1 - video_prob, video_prob])
        if num_videos > 0:
            video_tokens = sample_video_tokens_bimodal(num_videos, rng.integers(0, 2**31-1))
            req.data['video_tokens'] = [int(token) for token in video_tokens]
            req.data['video_resolution'] = [
                tuple(int(x) for x in tokens_to_video_resolution(token))
                for token in video_tokens
            ]
        else:
            req.data['video_tokens'] = []
            req.data['video_resolution'] = []
            
        num_audios = rng.choice([0, 2], p=[1 - audio_prob, audio_prob])
        if num_audios > 0:
            audio_tokens = sample_audio_tokens(num_audios, rng.integers(0, 2**31-1))
            req.data['audio_tokens'] = [int(token) for token in audio_tokens]
            req.data['audio_duration'] = [
                float(tokens_to_audio_sec(token))
                for token in audio_tokens
            ]
        else:
            req.data['audio_tokens'] = []
            req.data['audio_duration'] = []
    
    return requests
