"""Critic loading + gradient guidance for the pi0.5 flow policy.

This is the critic side of the critic-guided flow sampling in
``critic_ensemble_toy_example/tiny_flow_policy.py::TinyFlowPolicy.sample_action``:
during flow-matching denoising the sampler estimates the clean action chunk, asks
the critic for ``dValue/d(action)``, rescales that gradient to the velocity norm,
and nudges the velocity toward higher critic value. The sampling loop lives in
``openpi.models.pi0.Pi0.sample_actions``; this module provides the loadable critic
(``action_gradient``, numpy in/out) that the JAX sampler calls via
``jax.pure_callback``.

The critic (``/home/natasha/openpi/critic/model.py``) conditions on the pi0.5 head
SigLIP patch features ``(b, 256, 1152)``, a **raw** 14-d robot state and a **raw**
14-d action *chunk* ``(b, horizon, 14)``, and regresses a scalar value. The flow
sampler, however, works in the
model's *normalized*, action-dim-padded (32-d) space over a 50-step chunk. This
wrapper bridges the two:

  * the head SigLIP features are supplied by ``sample_actions`` (it re-runs the image
    tower to grab ``aux["encoded"]``);
  * the padded, normalized state/action are un-normalized back to the raw 14-d space
    the critic expects (``adapt_to_pi=False`` for RoboTwin, so the only transforms are
    Normalize + padding, both diagonal-affine);
  * the whole (horizon, 14) action chunk is scored in one critic call (its action encoder
    treats the chunk as a single-channel 2-D image) with the current-frame obs/state, and
    the gradient is taken w.r.t. the whole chunk, then scattered back into the first 14 of
    the 32 action dims (padding dims get zero gradient).

Because the un-normalization ``raw = norm * scale + loc`` is applied as a differentiable
torch op with the *normalized* action as the autograd leaf, ``torch.autograd`` returns
the gradient directly in the normalized space the sampler needs (chain rule handled for
free).
"""

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Critic architecture (vendored from openpi/critic/model.py, SigLIP-embedding variant).
# Module names match the checkpoint's state_dict so it loads as-is.
# --------------------------------------------------------------------------------------
class SigLIPEmbEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1152, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ffn = nn.Linear(128, 128)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.ffn(x)


class RobotStateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(14, 64), nn.ReLU(), nn.Linear(64, 128))

    def forward(self, x):
        return self.fc(x)


class RobotActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ffn = nn.Linear(128, 128)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv2d
        x = self.conv_block(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)  # Flatten to (b, 128)
        x = self.ffn(x)
        return x


class CriticMLP(nn.Module):
    """Scalar value from head SigLIP embedding(s), robot state and action (raw 14-d)."""

    def __init__(self, num_cameras: int = 1):
        super().__init__()
        self.siglip_emb_encoders = nn.ModuleList([SigLIPEmbEncoder() for _ in range(num_cameras)])
        fused_dim = 128 * (num_cameras + 2)  # per-camera + robot state + action
        self.robot_state_encoder = RobotStateEncoder()
        self.action_encoder = RobotActionEncoder()
        self.out = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, robot_state, action):
        robot_state_emb = self.robot_state_encoder(robot_state)
        action_emb = self.action_encoder(action)
        # obs: (b, num_cameras, 256, 1152) siglip embedding; reshape to (b, 1152, 16, 16) for Conv2d.
        if obs.ndim < 4:
            obs = obs.unsqueeze(1)
        siglip_embs = [
            encoder(obs[:, i, :, :].view(obs.size(0), 16, 16, 1152).permute(0, 3, 1, 2))
            for i, encoder in enumerate(self.siglip_emb_encoders)
        ]
        combined = torch.cat((*siglip_embs, robot_state_emb, action_emb), dim=1)
        return self.out(combined)


# --------------------------------------------------------------------------------------
# Guidance wrapper
# --------------------------------------------------------------------------------------
class SigLIPCritic:
    """Loads the SigLIP critic and exposes ``action_gradient`` for flow guidance.

    ``action_gradient(siglip, state, action)`` takes (all numpy, from the JAX sampler):
      * ``siglip``: head camera patch features, ``(b, 256, 1152)``;
      * ``state``:  normalized, padded state, ``(b, action_dim)``;
      * ``action``: normalized, padded action chunk, ``(b, horizon, action_dim)``;
    and returns ``d(sum_t value_t) / d(action)`` in the *same* normalized/padded space
    (``(b, horizon, action_dim)``), nonzero only on the first 14 (real) action dims.
    """

    CRITIC_DIM = 14  # the raw state/action dimensionality the critic was trained on

    def __init__(self, state_dict, num_cameras, state_loc, state_scale, action_loc, action_scale, device="cpu"):
        self.device = torch.device(device)
        self.model = CriticMLP(num_cameras=num_cameras).to(self.device).eval()
        self.model.load_state_dict(state_dict)
        for p in self.model.parameters():
            p.requires_grad_(False)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)
        self.state_loc, self.state_scale = t(state_loc), t(state_scale)
        self.action_loc, self.action_scale = t(action_loc), t(action_scale)

    def action_gradient(self, siglip, state, action):
        d = self.CRITIC_DIM
        siglip_t = torch.as_tensor(np.asarray(siglip, dtype=np.float32), device=self.device)
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)
        action_np = np.asarray(action, dtype=np.float32)
        action_t = torch.as_tensor(action_np, device=self.device)
        if action_t.ndim == 2:  # (horizon, dim) -> add batch
            action_t = action_t[None]
            state_t = state_t[None] if state_t.ndim == 1 else state_t

        # Un-normalize state -> raw 14-d (padding dims 14: are dropped).
        state_raw = state_t[:, :d] * self.state_scale + self.state_loc  # (b, 14)

        # Normalized action (first 14 dims) is the autograd leaf; un-normalize is a
        # differentiable affine op, so grad comes back in the normalized space directly.
        a_norm = action_t[..., :d].detach().requires_grad_(True)  # (b, horizon, 14)
        a_raw = a_norm * self.action_scale + self.action_loc

        # The critic scores the whole action chunk at once: its RobotActionEncoder treats
        # the (horizon, 14) chunk as a single-channel 2-D image (unsqueeze -> Conv2d, then
        # AdaptiveAvgPool2d over the horizon), so it takes the full (b, horizon, 14) chunk
        # with the shared current-frame obs/state and returns one value per batch element.
        # The gradient then flows back to every step of the chunk.
        with torch.enable_grad():
            value = self.model(siglip_t, state_raw, a_raw)           # (b, 1)
            grad_norm = torch.autograd.grad(value.sum(), a_norm)[0]  # (b, horizon, 14)

        # Scatter back into the padded action dims (padding gets zero gradient).
        grad_full = np.zeros(action_np.shape, dtype=np.float32)
        grad_full[..., :d] = grad_norm.detach().cpu().numpy()
        return grad_full


def _loc_scale(stats, use_quantile_norm):
    """(loc, scale) such that ``raw = normalized * scale + loc`` inverts openpi's Normalize."""
    if use_quantile_norm:
        q01 = np.asarray(stats.q01, dtype=np.float32)
        q99 = np.asarray(stats.q99, dtype=np.float32)
        scale = (q99 - q01 + 1e-6) / 2.0
        loc = q01 + scale
    else:
        mean = np.asarray(stats.mean, dtype=np.float32)
        std = np.asarray(stats.std, dtype=np.float32)
        scale = std + 1e-6
        loc = mean
    return loc, scale


def load_critic(checkpoint_path, norm_stats, use_quantile_norm, device="cpu"):
    """Load the SigLIP critic checkpoint and build a guidance wrapper.

    ``norm_stats`` is the openpi ``{"state": NormStats, "actions": NormStats}`` dict for
    this checkpoint; it is used to un-normalize actions/state into the critic's raw space.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # num_cameras is encoded in the fused input width: 128 * (num_cameras + 2).
    num_cameras = state_dict["out.0.weight"].shape[1] // 128 - 2
    state_loc, state_scale = _loc_scale(norm_stats["state"], use_quantile_norm)
    action_loc, action_scale = _loc_scale(norm_stats["actions"], use_quantile_norm)
    return SigLIPCritic(
        state_dict, num_cameras, state_loc, state_scale, action_loc, action_scale, device=device
    )
