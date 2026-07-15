"""Online QMFM ``Value`` critic for pi0.5 flow-matching guidance.

This replaces the frozen PyTorch ``SigLIPCritic`` (``critic_guidance.py``) with QMFM's
**exact** ``Value`` critic (an ensemble Q), trained **online** during eval rollouts, and
steers the frozen pi0.5 with QMFM's **exact denoised-estimate gradient guidance** (see
``Pi0.sample_actions``).

What is reproduced from QMFM (https://.../QMFM, ``agents/mfm.py`` + ``utils/networks.py``):

* ``Value`` / ``MLP`` / ``ensemblize`` / ``default_init`` are **copied verbatim** from
  ``QMFM/utils/networks.py`` (that module imports ``distrax`` + ``tensorflow_probability``
  at top, neither of which is in the pi05 venv, so we vendor the dependency-free classes).
* ``ReplayBuffer`` is **imported directly** from ``QMFM/utils/datasets.py`` (dependency-clean).
* The TD critic loss mirrors ``MFMAgent.critic_loss`` — ``target = r + gamma^H * mask *
  (mean_k - rho*std_k) target_Q(s', a')``; ``loss = mean((Q - target)^2)``; EMA target update.
  The one necessary deviation: the next action ``a'`` is **SARSA** (the guided-pi0.5's own next
  chunk, taken from the buffer) rather than QMFM's policy-resampled action, because pi0.5 is a
  frozen *external* actor that cannot be cheaply re-queried per batch element.

QMFM itself runs its ``Value`` with ``encoder=None`` on flat *state* observations (its tasks
are state-based). pi0.5's observation is visual, so — per the reproduction request — we supply
a CNN through ``Value``'s ``encoder`` hook (``SiglipCNNEncoder``), modelled on the SigLIP-feature
CNN of the critic being replaced (``critic_guidance.py::SigLIPEmbEncoder``). The critic observation
is ``CNN(head-SigLIP feature map) (+) proprio state``.
"""

import importlib.util
import os
import pickle
from functools import partial
from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

# --------------------------------------------------------------------------------------
# QMFM ReplayBuffer, imported by explicit file path (avoids any top-level ``utils`` clash;
# the eval driver runs from the RoboTwin repo root).
# --------------------------------------------------------------------------------------
_QMFM_ROOT = os.environ.get("QMFM_ROOT", "/home/natasha/QMFM")


def _load_qmfm_module(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_QMFM_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_qmfm_datasets = _load_qmfm_module("_qmfm_datasets", "utils/datasets.py")
ReplayBuffer = _qmfm_datasets.ReplayBuffer


# --------------------------------------------------------------------------------------
# Vendored VERBATIM from QMFM/utils/networks.py (Value / MLP / ensemblize / default_init).
# Byte-identical bodies so this is "the exact same critic class".
# --------------------------------------------------------------------------------------
default_init = nn.initializers.xavier_uniform


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, y=None):
        if y is not None:
            x = jnp.concatenate([x, y], axis=-1)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)
        if self.num_ensembles == 1:
            v = v[None, ...]

        return v


# --------------------------------------------------------------------------------------
# CNN observation encoder plugged into Value.encoder.
# Modelled on critic_guidance.py::SigLIPEmbEncoder (Conv->act->Conv->act->global-pool->Dense),
# operating on the head-camera SigLIP feature map reshaped to (b, 16, 16, 1152). BatchNorm is
# replaced by LayerNorm so ``apply({"params": ...})`` needs no mutable batch-stat collection.
# --------------------------------------------------------------------------------------
class SiglipCNNEncoder(nn.Module):
    """Encode ``obs = (siglip_map, state)`` -> ``concat(CNN(siglip_map), state)``.

    ``siglip_map``: ``(b, 16, 16, 1152)`` head-camera SigLIP patch features.
    ``state``:      ``(b, state_dim)`` proprioceptive state.
    """

    features: Sequence[int] = (128, 128)
    out_dim: int = 128

    @nn.compact
    def __call__(self, observations):
        siglip_map, state = observations
        x = siglip_map
        if x.ndim == 3:  # (256, 1152) -> (16, 16, 1152)
            x = x.reshape(16, 16, x.shape[-1])
        elif x.shape[-3:-1] != (16, 16):  # (b, 256, 1152) -> (b, 16, 16, 1152)
            x = x.reshape(*x.shape[:-2], 16, 16, x.shape[-1])
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            x = nn.gelu(x)
            x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=(-3, -2))  # global average pool -> (..., feat)
        x = nn.gelu(nn.Dense(self.out_dim)(x))
        return jnp.concatenate([x, state], axis=-1)


def build_critic_def(config: dict) -> Value:
    """Construct the QMFM ``Value`` critic (with CNN encoder) from a config dict."""
    encoder = SiglipCNNEncoder(
        features=tuple(config.get("cnn_features", (128, 128))),
        out_dim=int(config.get("cnn_out_dim", 128)),
    )
    return Value(
        hidden_dims=tuple(config["value_hidden_dims"]),
        layer_norm=bool(config["value_layer_norm"]),
        num_ensembles=int(config["num_qs"]),
        encoder=encoder,
    )


# --------------------------------------------------------------------------------------
# Online critic manager: holds the Value params/target/optimizer + a QMFM ReplayBuffer and a
# per-control-step transition state machine. Guidance reads ``.params`` (traced) each infer;
# ``.critic_apply`` (stable identity) is passed to sample_actions as a static arg.
# --------------------------------------------------------------------------------------
class OnlineValueCritic:
    def __init__(self, seed: int, config: dict):
        self.config = dict(config)
        self.critic_def = build_critic_def(self.config)

        action_dim_flat = int(config["action_dim_flat"])
        state_dim = int(config["state_dim"])
        h = w = int(config.get("siglip_grid", 16))
        c = int(config.get("siglip_channels", 1152))

        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        self.rng = rng

        ex_obs = (jnp.zeros((1, h, w, c), jnp.float32), jnp.zeros((1, state_dim), jnp.float32))
        ex_action = jnp.zeros((1, action_dim_flat), jnp.float32)
        self.params = self.critic_def.init(init_rng, ex_obs, ex_action)["params"]
        self.target_params = self.params

        if bool(config.get("clip_grad", True)):
            self.tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(float(config["lr"])))
        else:
            self.tx = optax.adam(float(config["lr"]))
        self.opt_state = self.tx.init(self.params)

        # Guidance apply fn (stable identity -> safe as a jit static arg).
        _cdef = self.critic_def

        def _critic_apply(params, observations, actions):
            return _cdef.apply({"params": params}, observations, actions)

        self.critic_apply = _critic_apply
        self._jit_update = jax.jit(self._make_update_fn())

        # Replay buffer + transition state machine.
        self.buffer = None
        self._prev = None  # (img, state, action_flat, reward) of the last non-terminal step
        self._cur = None   # (img, state, action_flat) stashed by the most recent sample_actions
        self.num_updates = 0
        self.siglip_grid = h

    # ---- network update (SARSA TD, mirrors MFMAgent.critic_loss) ----------------------
    def _make_update_fn(self):
        cdef = self.critic_def
        discount = float(self.config["discount"])
        horizon = int(self.config["horizon"])  # primitive sim steps executed per chunk (pi0_step)
        rho = float(self.config["rho"])
        tau = float(self.config["tau"])
        tx = self.tx
        gamma_h = discount ** horizon

        def update_fn(params, target_params, opt_state, batch):
            obs = (batch["obs_img"].astype(jnp.float32), batch["obs_state"].astype(jnp.float32))
            next_obs = (batch["next_obs_img"].astype(jnp.float32), batch["next_obs_state"].astype(jnp.float32))
            actions = batch["actions"].astype(jnp.float32)
            next_actions = batch["next_actions"].astype(jnp.float32)
            rewards = batch["rewards"].astype(jnp.float32)
            masks = batch["masks"].astype(jnp.float32)

            next_qs = cdef.apply({"params": target_params}, next_obs, next_actions)  # (num_qs, B)
            next_q = next_qs.mean(axis=0) - rho * next_qs.std(axis=0)
            target_q = rewards + gamma_h * masks * next_q

            def loss_fn(p):
                q = cdef.apply({"params": p}, obs, actions)  # (num_qs, B)
                loss = jnp.mean((q - target_q[None]) ** 2)
                return loss, {
                    "critic_loss": loss,
                    "q_mean": q.mean(),
                    "q_max": q.max(),
                    "q_min": q.min(),
                    "target_q_mean": target_q.mean(),
                    "reward_mean": rewards.mean(),
                }

            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_target = jax.tree_util.tree_map(lambda pn, tp: tau * pn + (1.0 - tau) * tp, new_params, target_params)
            return new_params, new_target, new_opt_state, info

        return update_fn

    def update(self):
        batch = self._sample(int(self.config["batch_size"]))
        self.params, self.target_params, self.opt_state, info = self._jit_update(
            self.params, self.target_params, self.opt_state, batch
        )
        self.num_updates += 1
        return info

    def train_step(self):
        """Run ``utd_ratio`` gradient updates; returns the last info dict (or None if not ready)."""
        if not self.ready():
            return None
        info = None
        for _ in range(int(self.config.get("utd_ratio", 1))):
            info = self.update()
        return info

    def ready(self):
        return self.buffer is not None and self.buffer.size >= int(self.config["start_training"])

    def _sample(self, batch_size):
        idxs = np.random.randint(self.buffer.size, size=batch_size)
        return {k: jnp.asarray(self.buffer[k][idxs]) for k in self.buffer.keys()}

    # ---- transition collection --------------------------------------------------------
    def stash(self, siglip_map, state, action):
        """Record the critic's (obs, action) for the control step just sampled."""
        img = np.asarray(siglip_map, dtype=np.float16)
        if img.ndim == 2:  # (256, 1152) -> (16, 16, 1152)
            img = img.reshape(self.siglip_grid, self.siglip_grid, img.shape[-1])
        self._cur = (
            img,
            np.asarray(state, dtype=np.float32).reshape(-1),
            np.asarray(action, dtype=np.float32).reshape(-1),
        )

    def commit(self, reward, done):
        """Close the pending transition(s) once the chunk's reward/termination is known.

        Forms ``(obs_k, a_k, r_k, obs_{k+1}, a_{k+1}, mask=1-done_k)`` (SARSA), bridging the
        previous non-terminal step to the current one, and finalising a terminal step against
        itself with ``mask=0``.
        """
        if self._cur is None:
            return
        cur_img, cur_state, cur_act = self._cur
        if self._prev is not None:
            p_img, p_state, p_act, p_r = self._prev
            self._add(p_img, p_state, p_act, p_r, cur_img, cur_state, cur_act, mask=1.0, terminal=0.0)
            self._prev = None
        if done:
            self._add(cur_img, cur_state, cur_act, float(reward), cur_img, cur_state, cur_act, mask=0.0, terminal=1.0)
            self._prev = None
            self._cur = None
        else:
            self._prev = (cur_img, cur_state, cur_act, float(reward))

    def reset_episode(self):
        self._prev = None
        self._cur = None

    def _add(self, img, state, act, r, n_img, n_state, n_act, mask, terminal):
        transition = dict(
            obs_img=img,
            obs_state=state,
            actions=act,
            rewards=np.float32(r),
            masks=np.float32(mask),
            terminals=np.float32(terminal),
            next_obs_img=n_img,
            next_obs_state=n_state,
            next_actions=n_act,
        )
        if self.buffer is None:
            self.buffer = ReplayBuffer.create(transition, size=int(self.config["buffer_size"]))
        self.buffer.add_transition(transition)

    # ---- persistence ------------------------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "params": jax.device_get(self.params),
                    "target_params": jax.device_get(self.target_params),
                    "config": self.config,
                    "num_updates": self.num_updates,
                },
                f,
            )

    def load(self, path):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        self.params = jax.device_put(blob["params"])
        self.target_params = jax.device_put(blob["target_params"])
        self.num_updates = blob.get("num_updates", 0)
