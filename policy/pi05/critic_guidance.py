"""Critic loading + gradient guidance for the pi0 flow policy.

This is the critic side of the critic-guided flow sampling used in
``critic_ensemble_toy_example/tiny_flow_policy.py::TinyFlowPolicy.sample_action``:
during flow-matching denoising the sampler estimates the clean action, asks the
critic for ``dQ/d(action)``, rescales that gradient to the velocity norm, and
nudges the velocity toward higher Q. The sampling loop itself lives in
``openpi.models.pi0.Pi0.sample_actions``; this module only provides a loadable
critic exposing ``action_gradient`` (numpy in, numpy out) so the JAX sampler can
call it through ``jax.pure_callback``.

The weight layout matches ``PessimisticCriticEnsemble`` from the toy example
(``critic_ensemble_toy_example/critics.py``): each layer stores stacked per-critic
weights ``W*`` of shape ``(num_critics, in, out)`` and biases ``b*`` of shape
``(num_critics, 1, out)``, run as a single batched einsum. Action chunks are
flattened before being concatenated with the state, so the same code works for
any state / action-chunk dimensionality.
"""

import numpy as np
import torch


class GuidanceCritic:
    """Pessimistic ensemble of Q(state, action) critics, restored from a checkpoint."""

    def __init__(self, weights, device="cpu"):
        self.device = torch.device(device)
        (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3) = (
            w.to(self.device).float() for w in weights
        )
        self.num_critics = int(self.W1.shape[0])

    def _q_values(self, inp):
        """Batched forward for all critics. inp: (C, B, in) -> (C, B)."""
        x = torch.relu(torch.einsum("cbi,cio->cbo", inp, self.W1) + self.b1)
        x = torch.relu(torch.einsum("cbi,cio->cbo", x, self.W2) + self.b2)
        x = torch.einsum("cbi,cio->cbo", x, self.W3) + self.b3
        return x.squeeze(-1)

    def action_gradient(self, state, action):
        """Gradient of the pessimistic Q w.r.t. the action.

        ``state`` is ``(B, state_dim)`` and ``action`` is ``(B, *action_shape)``
        (e.g. ``(B, action_horizon, action_dim)``). Returns a numpy array with the
        same shape as ``action``. Mirrors
        ``PessimisticCriticEnsemble.action_gradient`` in the toy example.
        """
        action = np.asarray(action, dtype=np.float32)
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)
        action_t = torch.as_tensor(action, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t[None]
        if action_t.ndim == 1:
            action_t = action_t[None]
        batch = action_t.shape[0]

        action_flat = action_t.reshape(batch, -1).detach().requires_grad_(True)
        inputs = torch.cat([state_t.reshape(batch, -1), action_flat], dim=1)  # (B, in)
        stacked = inputs[None].expand(self.num_critics, -1, -1)  # (C, B, in)
        with torch.enable_grad():
            q_values = self._q_values(stacked)  # (C, B)
            q_mean = q_values.mean(dim=0)
            q_std = q_values.std(dim=0, unbiased=False)
            pessimistic_q = q_mean - 0.5 * q_std / np.sqrt(self.num_critics)
            grad = torch.autograd.grad(pessimistic_q.sum(), action_flat)[0]
        return grad.reshape(action.shape).detach().cpu().numpy().astype(np.float32)


def load_critic(checkpoint_path, device="cpu"):
    """Load a critic checkpoint saved by the toy example's ``save_sarsa_checkpoint``.

    The checkpoint is a dict with a ``critic_state_dict`` holding the stacked
    ensemble weights (``W1/b1/W2/b2/W3/b3``); a bare state dict is also accepted.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("critic_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    weights = (
        state_dict["W1"],
        state_dict["b1"],
        state_dict["W2"],
        state_dict["b2"],
        state_dict["W3"],
        state_dict["b3"],
    )
    return GuidanceCritic(weights, device=device)
