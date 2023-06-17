# # Score-Based Generative Modeling
# 
# 
# ### Goals
# This is a hitchhiker's guide to score-based generative models, a family of approaches based on [estimating gradients of the data distribution](https://arxiv.org/abs/1907.05600). They have obtained high-quality samples comparable to GANs (like below, figure from [this paper](https://arxiv.org/abs/2006.09011)) without requiring adversarial training, and are considered by some to be [the new contender to GANs](https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/).
# 
# ![ncsnv2](https://github.com/ermongroup/ncsnv2/blob/master/assets/samples.jpg?raw=true)
# 
# The contents of this notebook are mainly based on the following paper: 
# 
# Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. "[Score-Based Generative Modeling through Stochastic Differential Equations.](https://arxiv.org/pdf/2011.13456.pdf)" Internation Conference on Learning Representations, 2021

# ## Introduction
# 
# ### Score and Score-Based Models
# Given a probablity density function $p(\mathbf{x})$, we define the *score* as $$\nabla_\mathbf{x} \log p(\mathbf{x}).$$ As you might guess, score-based generative models are trained to estimate $\nabla_\mathbf{x} \log p(\mathbf{x})$. Unlike likelihood-based models such as flow models or autoregressive models, score-based models do not have to be normalized and are easier to parameterize. For example, consider a non-normalized statistical model $p_\theta(\mathbf{x}) = \frac{e^{-E_\theta(\mathbf{x})}}{Z_\theta}$, where $E_\theta(\mathbf{x}) \in \mathbb{R}$ is called the energy function and $Z_\theta$ is an unknown normalizing constant that makes $p_\theta(\mathbf{x})$ a proper probability density function. The energy function is typically parameterized by a flexible neural network. When training it as a likelihood model, we need to know the normalizing constant $Z_\theta$ by computing complex high-dimensional integrals, which is typically intractable. In constrast, when computing its score, we obtain $\nabla_\mathbf{x} \log p_\theta(\mathbf{x}) = -\nabla_\mathbf{x} E_\theta(\mathbf{x})$ which does not require computing the normalizing constant $Z_\theta$.
# 
# In fact, any neural network that maps an input vector $\mathbf{x} \in \mathbb{R}^d$ to an output vector $\mathbf{y} \in \mathbb{R}^d$ can be used as a score-based model, as long as the output and input have the same dimensionality. This yields huge flexibility in choosing model architectures.
# 
# ### Perturbing Data with a Diffusion Process
# 
# In order to generate samples with score-based models, we need to consider a [diffusion process](https://en.wikipedia.org/wiki/Diffusion_process) that corrupts data slowly into random noise. Scores will arise when we reverse this diffusion process for sample generation. You will see this later in the notebook.
# 
# A diffusion process is a [stochastic process](https://en.wikipedia.org/wiki/Stochastic_process#:~:text=A%20stochastic%20or%20random%20process%20can%20be%20defined%20as%20a,an%20element%20in%20the%20set.) similar to [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion). Their paths are like the trajectory of a particle submerged in a flowing fluid, which moves randomly due to unpredictable collisions with other particles. Let $\{\mathbf{x}(t) \in \mathbb{R}^d \}_{t=0}^T$ be a diffusion process, indexed by the continuous time variable $t\in [0,T]$. A diffusion process is governed by a stochastic differential equation (SDE), in the following form
# 
# \begin{align*}
# d \mathbf{x} = \mathbf{f}(\mathbf{x}, t) d t + g(t) d \mathbf{w},
# \end{align*}
# 
# where $\mathbf{f}(\cdot, t): \mathbb{R}^d \to \mathbb{R}^d$ is called the *drift coefficient* of the SDE, $g(t) \in \mathbb{R}$ is called the *diffusion coefficient*, and $\mathbf{w}$ represents the standard Brownian motion. You can understand an SDE as a stochastic generalization to ordinary differential equations (ODEs). Particles moving according to an SDE not only follows the deterministic drift $\mathbf{f}(\mathbf{x}, t)$, but are also affected by the random noise coming from $g(t) d\mathbf{w}$. From now on, we use $p_t(\mathbf{x})$ to denote the distribution of $\mathbf{x}(t)$. 
# 
# For score-based generative modeling, we will choose a diffusion process such that $\mathbf{x}(0) \sim p_0$, and $\mathbf{x}(T) \sim p_T$. Here $p_0$ is the data distribution where we have a dataset of i.i.d. samples, and $p_T$ is the prior distribution that has a tractable form and easy to sample from. The noise perturbation by the diffusion process is large enough to ensure $p_T$ does not depend on $p_0$.
# 
# ### Reversing the Diffusion Process Yields Score-Based Generative Models
# By starting from a sample from the prior distribution $p_T$ and reversing the diffusion process, we will be able to obtain a sample from the data distribution $p_0$. Crucially, the reverse process is a diffusion process running backwards in time. It is given by the following reverse-time SDE
# 
# \begin{align}
#   d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}},
# \end{align}
# 
# where $\bar{\mathbf{w}}$ is a Brownian motion in the reverse time direction, and $dt$ represents an infinitesimal negative time step. This reverse SDE can be computed once we know the drift and diffusion coefficients of the forward SDE, as well as the score of $p_t(\mathbf{x})$ for each $t\in[0, T]$.
# 
# The overall intuition of score-based generative modeling with SDEs can be summarized in the illustration below
# 
# ![sde schematic](https://drive.google.com/uc?id=1Ptvb790eQRYMHLnDGBeYZK9A2cF-JMEP)
# 
# 
# ### Score Estimation
# 
# Based on the above intuition, we can use the time-dependent score function $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ to construct the reverse-time SDE, and then solve it numerically to obtain samples from $p_0$ using samples from a prior distribution $p_T$. We can train a time-dependent score-based model $s_\theta(\mathbf{x}, t)$ to approximate $\nabla_\mathbf{x} \log p_t(\mathbf{x})$, using the following weighted sum of [denoising score matching](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) objectives.
# 
# \begin{align}
# \min_\theta \mathbb{E}_{t\sim \mathcal{U}(0, T)} [\lambda(t) \mathbb{E}_{\mathbf{x}(0) \sim p_0(\mathbf{x})}\mathbf{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))}[ \|s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))\|_2^2]],
# \end{align}
# where $\mathcal{U}(0,T)$ is a uniform distribution over $[0, T]$, $p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))$ denotes the transition probability from $\mathbf{x}(0)$ to $\mathbf{x}(t)$, and $\lambda(t) \in \mathbb{R}_{>0}$ denotes a positive weighting function.
# 
# In the objective, the expectation over $\mathbf{x}(0)$ can be estimated with empirical means over data samples from $p_0$. The expectation over $\mathbf{x}(t)$ can be estimated by sampling from $p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))$, which is efficient when the drift coefficient $\mathbf{f}(\mathbf{x}, t)$ is affine. The weight function $\lambda(t)$ is typically chosen to be inverse proportional to $\mathbb{E}[\|\nabla_{\mathbf{x}}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) \|_2^2]$.
# 
# 

# ### Time-Dependent Score-Based Model
# 
# There are no restrictions on the network architecture of time-dependent score-based models, except that their output should have the same dimensionality as the input, and they should be conditioned on time.
# 
# Several useful tips on architecture choice:
# * It usually performs well to use the [U-net](https://arxiv.org/abs/1505.04597) architecture as the backbone of the score network $s_\theta(\mathbf{x}, t)$,
# 
# * We can incorporate the time information via [Gaussian random features](https://arxiv.org/abs/2006.10739). Specifically, we first sample $\omega \sim \mathcal{N}(\mathbf{0}, s^2\mathbf{I})$ which is subsequently fixed for the model (i.e., not learnable). For a time step $t$, the corresponding Gaussian random feature is defined as 
# \begin{align}
#   [\sin(2\pi \omega t) ; \cos(2\pi \omega t)],
# \end{align}
# where $[\vec{a} ; \vec{b}]$ denotes the concatenation of vector $\vec{a}$ and $\vec{b}$. This Gaussian random feature can be used as an encoding for time step $t$ so that the score network can condition on $t$ by incorporating this encoding. We will see this further in the code.
# 
# * We can rescale the output of the U-net by $1/\sqrt{\mathbb{E}[\|\nabla_{\mathbf{x}}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) \|_2^2]}$. This is because the optimal $s_\theta(\mathbf{x}(t), t)$ has an $\ell_2$-norm close to $\mathbb{E}[\|\nabla_{\mathbf{x}}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))]\|_2$, and the rescaling helps capture the norm of the true score. Recall that the training objective contains sums of the form
# \begin{align*}
# \mathbf{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))}[ \|s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))\|_2^2].
# \end{align*}
# Therefore, it is natural to expect that the optimal score model $s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))$.
# 
# * Use [exponential moving average](https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3) (EMA) of weights when sampling. This can greatly improve sample quality, but requires slightly longer training time, and requires more work in implementation. We do not include this in this tutorial, but highly recommend it when you employ score-based generative modeling to tackle more challenging real problems.

# In[ ]:


# @title Defining a time-dependent score-based model (double click to expand or collapse)

import functools
from typing import Any, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import torch
import torchvision.transforms as transforms
# import tqdm
from tqdm import trange, tqdm

from flax.serialization import to_bytes, from_bytes
from scipy import integrate
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
devices = jax.devices()
num_devices = jax.local_device_count()
print(f"Emma's fantastic machine has {num_devices} cores!")
print(f"And JAX calls them {devices}!")

# print number of cpu cores
import multiprocessing

print(f"Emma's fantastic machine has {multiprocessing.cpu_count()} cpu cores!")


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    embed_dim: int
    scale: float = 30.

    @nn.compact
    def __call__(self, x):
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        W = self.param('W', jax.nn.initializers.normal(stddev=self.scale),
                       (self.embed_dim // 2,))
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    output_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_dim)(x)[:, None, None, :]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.

    Args:
        marginal_prob_std: A function that takes time t and gives the standard
          deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    marginal_prob_std: Any
    channels: Tuple[int] = (32, 64, 128, 256)
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x, t):
        # The swish activation function
        act = nn.swish
        # Obtain the Gaussian random feature embedding for t
        embed = act(nn.Dense(self.embed_dim)(
            GaussianFourierProjection(embed_dim=self.embed_dim)(t)))

        # Encoding path
        h1 = nn.Conv(self.channels[0], (3, 3), (1, 1), padding='VALID',
                     use_bias=False)(x)
        ## Incorporate information from t
        h1 += Dense(self.channels[0])(embed)
        ## Group normalization
        h1 = nn.GroupNorm(4)(h1)
        h1 = act(h1)
        h2 = nn.Conv(self.channels[1], (3, 3), (2, 2), padding='VALID',
                     use_bias=False)(h1)
        h2 += Dense(self.channels[1])(embed)
        h2 = nn.GroupNorm(4)(h2)
        h2 = act(h2)
        h3 = nn.Conv(self.channels[2], (3, 3), (2, 2), padding='VALID',
                     use_bias=False)(h2)
        h3 += Dense(self.channels[2])(embed)
        h3 = nn.GroupNorm(4)(h3)
        h3 = act(h3)
        h4 = nn.Conv(self.channels[3], (3, 3), (2, 2), padding='VALID',
                     use_bias=False)(h3)
        h4 += Dense(self.channels[3])(embed)
        h4 = nn.GroupNorm(4)(h4)
        h4 = act(h4)

        # Decoding path
        h = nn.Conv(self.channels[2], (3, 3), (1, 1), padding=((2, 2), (2, 2)),
                    input_dilation=(2, 2), use_bias=False)(h4)
        ## Skip connection from the encoding path
        h += Dense(self.channels[2])(embed)
        h = nn.GroupNorm(4)(h)
        h = act(h)
        h = nn.Conv(self.channels[1], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                    input_dilation=(2, 2), use_bias=False)(
            jnp.concatenate([h, h3], axis=-1)
        )
        h += Dense(self.channels[1])(embed)
        h = nn.GroupNorm(4)(h)
        h = act(h)
        h = nn.Conv(self.channels[0], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                    input_dilation=(2, 2), use_bias=False)(
            jnp.concatenate([h, h2], axis=-1)
        )
        h += Dense(self.channels[0])(embed)
        h = nn.GroupNorm(4)(h)
        h = act(h)
        h = nn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
            jnp.concatenate([h, h1], axis=-1)
        )

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class TinyScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.

    Args:
        marginal_prob_std: A function that takes time t and gives the standard
          deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    marginal_prob_std: Any
    channels: Tuple[int] = (32, 64, 128, 256)
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x, t):
        # The swish activation function
        act = nn.swish
        # Obtain the Gaussian random feature embedding for t
        embed = act(nn.Dense(self.embed_dim)(
            GaussianFourierProjection(embed_dim=self.embed_dim)(t)))

        hs = [None, ] * len(self.channels)
        # Encoding path
        hs[0] = nn.Conv(self.channels[0], (3, 3), (1, 1), padding='VALID',
                        use_bias=False)(x)
        ## Incorporate information from t
        hs[0] += Dense(self.channels[0])(embed)
        ## Group normalization
        hs[0] = nn.GroupNorm(4)(hs[0])
        hs[0] = act(hs[0])

        for c, num_channels in enumerate(self.channels[1:]):
            hs[c + 1] = nn.Conv(num_channels, (3, 3), (2, 2), padding='VALID', use_bias=False)(hs[c])
            hs[c + 1] += Dense(num_channels)(embed)
            hs[c + 1] = nn.GroupNorm(num_groups=None, group_size=4)(hs[c + 1])
            hs[c + 1] = act(hs[c + 1])

        # h2 = nn.Conv(self.channels[1], (3, 3), (2, 2), padding='VALID', use_bias=False)(h1)
        # h2 += Dense(self.channels[1])(embed)
        # h2 = nn.GroupNorm()(h2)
        # h2 = act(h2)
        #
        # h3 = nn.Conv(self.channels[2], (3, 3), (2, 2), padding='VALID', use_bias=False)(h2)
        # h3 += Dense(self.channels[2])(embed)
        # h3 = nn.GroupNorm()(h3)
        # h3 = act(h3)
        #
        # h4 = nn.Conv(self.channels[3], (3, 3), (2, 2), padding='VALID', use_bias=False)(h3)
        # h4 += Dense(self.channels[3])(embed)
        # h4 = nn.GroupNorm()(h4)
        # h4 = act(h4)

        # Decoding path
        # h = nn.Conv(self.channels[-2], (3, 3), (1, 1), padding=((2, 2), (2, 2)),
        #             input_dilation=(2, 2), use_bias=False)(h4)

        channels = list(range(1, len(self.channels)))
        # for c, channel in enumerate(range(len(self.channels)-1, 0, -1)):
        for c, rc in zip(channels, reversed(channels)):
            ## Skip connection from the encoding path
            representation = hs[rc] if c == 1 else jnp.concatenate([h, hs[rc]], axis=-1)

            h = nn.Conv(self.channels[rc - 1], (3, 3), (1, 1), padding=((2, 2), (2, 2)),
                        input_dilation=(2, 2), use_bias=False)(representation)

            h += Dense(self.channels[rc - 1])(embed)
            h = nn.GroupNorm(num_groups=None, group_size=4)(h)
            h = act(h)

        # ## Skip connection from the encoding path
        # h += Dense(self.channels[2])(embed)
        # h = nn.GroupNorm()(h)
        # h = act(h)
        # h = nn.Conv(self.channels[1], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
        #             input_dilation=(2, 2), use_bias=False)(
        #     jnp.concatenate([h, h3], axis=-1)
        # )
        #
        # h += Dense(self.channels[1])(embed)
        # h = nn.GroupNorm()(h)
        # h = act(h)
        # h = nn.Conv(self.channels[0], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
        #             input_dilation=(2, 2), use_bias=False)(
        #     jnp.concatenate([h, h2], axis=-1)
        # )
        #
        # h += Dense(self.channels[0])(embed)
        # h = nn.GroupNorm()(h)
        # h = act(h)
        h = nn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
            jnp.concatenate([h, h1], axis=-1)
        )

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


# ## Training with Weighted Sum of Denoising Score Matching Objectives
# 
# Now let's get our hands dirty on training. First of all, we need to specify an SDE that perturbs the data distribution $p_0$ to a prior distribution $p_T$. We choose the following SDE
# \begin{align*}
# d \mathbf{x} = \sigma^t d\mathbf{w}, \quad t\in[0,1]
# \end{align*}
# In this case,
# \begin{align*}
# p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) = \mathcal{N}\bigg(\mathbf{x}(t); \mathbf{x}(0), \frac{1}{2\log \sigma}(\sigma^{2t} - 1) \mathbf{I}\bigg)
# \end{align*}
# and we can choose the weighting function $\lambda(t) = \frac{1}{2 \log \sigma}(\sigma^{2t} - 1)$.
# 
# When $\sigma$ is large, the prior distribution, $p_{t=1}$ is 
# \begin{align*}
# \int p_0(\mathbf{y})\mathcal{N}\bigg(\mathbf{x}; \mathbf{y}, \frac{1}{2 \log \sigma}(\sigma^2 - 1)\mathbf{I}\bigg) d \mathbf{y} \approx \mathbf{N}\bigg(\mathbf{x}; \mathbf{0}, \frac{1}{2 \log \sigma}(\sigma^2 - 1)\mathbf{I}\bigg),
# \end{align*}
# which is approximately independent of the data distribution and is easy to sample from.
# 
# Intuitively, this SDE captures a continuum of Gaussian perturbations with variance function $\frac{1}{2 \log \sigma}(\sigma^{2t} - 1)$. This continuum of perturbations allows us to gradually transfer samples from a data distribution $p_0$ to a simple Gaussian distribution $p_1$.

# In[ ]:


# @title Set up the SDE

def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    return jnp.sqrt((sigma ** (2 * t) - 1.) / 2. / jnp.log(sigma))


def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return sigma ** t


sigma = 25.0  # @param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


# @title Define the loss function (double click to expand or collapse)

def loss_fn(rng, model, params, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
      model: A `flax.linen.Module` object that represents the structure of
        the score-based model.
      params: A dictionary that contains all trainable parameters.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
    """
    rng, step_rng = jax.random.split(rng)
    random_t = jax.random.uniform(step_rng, (x.shape[0],), minval=eps, maxval=1.)
    rng, step_rng = jax.random.split(rng)
    z = jax.random.normal(step_rng, x.shape)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model.apply(params, perturbed_x, random_t)
    loss = jnp.mean(jnp.sum((score * std[:, None, None, None] + z) ** 2,
                            axis=(1, 2, 3)))
    return loss


def get_train_step_fn(model, optimiser, marginal_prob_std):
    """Create a one-step training function.

    Args:
      model: A `flax.linen.Module` object that represents the structure of
        the score-based model.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    Returns:
      A function that runs one step of training.
    """

    val_and_grad_fn = jax.value_and_grad(loss_fn, argnums=2)

    # def step_fn(rng, x, optimizer):
    def step_fn(rng, x, params, opt_state):
        # params = optimizer.target
        loss, grad = val_and_grad_fn(rng, model, params, x, marginal_prob_std)
        mean_grad = jax.lax.pmean(grad, axis_name='device')
        mean_loss = jax.lax.pmean(loss, axis_name='device')

        # new_optimizer = optimizer.apply_gradient(mean_grad)
        updates, opt_state = optimiser.update(mean_grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        # return mean_loss, new_optimizer
        return params, opt_state, mean_loss

    return jax.pmap(step_fn, axis_name='device')


n_epochs = 50  # @param {'type':'integer'}
## size of a mini-batch
batch_size = 256  # @param {'type':'integer'}
## learning rate
lr = 1e-4  # @param {'type':'number'}

rng = jax.random.PRNGKey(0)
fake_input = jnp.ones((batch_size, 28, 28, 1))
fake_time = jnp.ones(batch_size)
# score_model = ScoreNet(marginal_prob_std_fn)
#score_model = ScoreNet(marginal_prob_std_fn, channels=(8, 16, 32, 64), embed_dim=64)
score_model = ScoreNet(marginal_prob_std_fn, channels=(8, 16, 16, 32), embed_dim=32)
#score_model = TinyScoreNet(marginal_prob_std_fn, channels=(8, 16, 32, 64), embed_dim=64)
params = score_model.init({'params': rng}, fake_input, fake_time)

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
print(f"We have {len(dataset)} images in our pretty dataset!")
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # , num_workers=4)

optimiser = optax.adam(learning_rate=lr)
opt_state = optimiser.init(params)

train_step_fn = get_train_step_fn(score_model, optimiser, marginal_prob_std_fn)
tqdm_dataloader = tqdm(data_loader, position=0, desc="Batch progress")
tqdm_epoch = trange(n_epochs, position=1, desc="Epoch progress")

assert batch_size % jax.local_device_count() == 0
data_shape = (jax.local_device_count(), -1, 28, 28, 1)

# optimiser = flax.jax_utils.replicate(optimiser)
params = flax.jax_utils.replicate(params)
opt_state = flax.jax_utils.replicate(opt_state)

for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    for i, (x, y) in enumerate(tqdm_dataloader):
        x = x.permute(0, 2, 3, 1).numpy().reshape(data_shape)
        rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
        step_rng = jnp.asarray(step_rng)
        # loss, optimiser = train_step_fn(step_rng, x, optimiser)
        params, opt_state, mean_loss = train_step_fn(step_rng, x, params, opt_state)

        mean_loss = flax.jax_utils.unreplicate(mean_loss)
        avg_loss += mean_loss.item() * x.shape[0] * x.shape[1]
        num_items += x.shape[0] * x.shape[1]
        # tqdm_dataloader.set_description("Batch progress")
        if i == 10:
            break

        # Update the checkpoint after each epoch of training.
    with tf.io.gfile.GFile('ckpt.flax', 'wb') as fout:
        # fout.write(to_bytes({"params": flax.jax_utils.unreplicate(params),
        #                      "opt_state": flax.jax_utils.unreplicate(opt_state)}))
        # save opt_state and params to file
        fout.write(to_bytes(flax.jax_utils.unreplicate(params)))
        # fout.write(to_bytes(flax.jax_utils.unreplicate(optimiser)))

    # Print the averaged training loss so far.
    tqdm_epoch.set_postfix({"Avg loss": avg_loss/num_items})
    #tqdm_epoch.set_description(f'Average Loss (for {num_items} images): {avg_loss / num_items:5f}')
    break

params = flax.jax_utils.unreplicate(params)
opt_state = flax.jax_utils.unreplicate(opt_state)

# ## Sampling with Numerical SDE Solvers
# Recall that for any SDE of the form
# \begin{align*}
# d \mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w},
# \end{align*}
# the reverse-time SDE is given by
# \begin{align*}
# d \mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})] dt + g(t) d \bar{\mathbf{w}}.
# \end{align*}
# Since we have chosen the forward SDE to be
# \begin{align*}
# d \mathbf{x} = \sigma^t d\mathbf{w}, \quad t\in[0,1]
# \end{align*}
# The reverse-time SDE is given by
# \begin{align*}
# d\mathbf{x} = -\sigma^{2t} \nabla_\mathbf{x} \log p_t(\mathbf{x}) dt + \sigma^t d \bar{\mathbf{w}}.
# \end{align*}
# To sample from our time-dependent score-based model $s_\theta(\mathbf{x}, t)$, we first draw a sample from the prior distribution $p_1 \approx \mathbf{N}\bigg(\mathbf{x}; \mathbf{0}, \frac{1}{2}(\sigma^{2} - 1) \mathbf{I}\bigg)$, and then solve the reverse-time SDE with numerical methods.
# 
# In particular, using our time-dependent score-based model, the reverse-time SDE can be approximated by
# \begin{align*}
# d\mathbf{x} = -\sigma^{2t} s_\theta(\mathbf{x}, t) dt + \sigma^t d \bar{\mathbf{w}}
# \end{align*}
# 
# Next, one can use numerical methods to solve for the reverse-time SDE, such as the [Euler-Maruyama](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) approach. It is based on a simple discretization to the SDE, replacing $dt$ with $\Delta t$ and $d \mathbf{w}$ with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, g^2(t) \Delta t \mathbf{I})$. When applied to our reverse-time SDE, we can obtain the following iteration rule
# \begin{align}
# \mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \sigma^{2t} s_\theta(\mathbf{x}_t, t)\Delta t + \sigma^t\sqrt{\Delta t} \mathbf{z}_t,
# \end{align}
# where $\mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

# In[ ]:


# @title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.
num_steps = 500  # @param {'type':'integer'}


def score_fn(score_model, params, x, t):
    return score_model.apply(params, x, t)


pmap_score_fn = jax.pmap(score_fn, static_broadcasted_argnums=(0, 1))


def Euler_Maruyama_sampler(rng,
                           score_model,
                           params,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=num_steps,
                           eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
      rng: A JAX random state.
      score_model: A `flax.linen.Module` object that represents the architecture
        of a score-based model.
      params: A dictionary that contains the model parameters.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    rng, step_rng = jax.random.split(rng)
    time_shape = (jax.local_device_count(), batch_size // jax.local_device_count())
    sample_shape = time_shape + (28, 28, 1)
    init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)
    time_steps = jnp.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    for time_step in tqdm(time_steps):
        batch_time_step = jnp.ones(time_shape) * time_step
        g = diffusion_coeff(time_step)
        mean_x = x + (g ** 2) * pmap_score_fn(score_model,
                                              params,
                                              x,
                                              batch_time_step) * step_size
        rng, step_rng = jax.random.split(rng)
        x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(step_rng, x.shape)
        # Do not include any noise in the last sampling step.
    return mean_x


# ## Sampling with Predictor-Corrector Methods
# 
# Aside from generic numerical SDE solvers, we can leverage special properties of our reverse-time SDE for better solutions. Since we have an estimate of the score of $p_t(\mathbf{x}(t))$ via the score-based model, i.e., $s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}(t)} \log p_t(\mathbf{x}(t))$, we can leverage score-based MCMC approaches, such as Langevin MCMC, to correct the solution obtained by numerical SDE solvers.
# 
# Score-based MCMC approaches can produce samples from a distribution $p(\mathbf{x})$ once its score $\nabla_\mathbf{x} \log p(\mathbf{x})$ is known. For example, Langevin MCMC operates by running the following iteration rule for $i=1,2,\cdots, N$:
# \begin{align*}
# \mathbf{x}_{i+1} = \mathbf{x}_{i} + \epsilon \nabla_\mathbf{x} \log p(\mathbf{x}_i) + \sqrt{2\epsilon} \mathbf{z}_i,
# \end{align*}
# where $\mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\epsilon > 0$ is the step size, and $\mathbf{x}_1$ is initialized from any prior distribution $\pi(\mathbf{x}_1)$. When $N\to\infty$ and $\epsilon \to 0$, the final value $\mathbf{x}_{N+1}$ becomes a sample from $p(\mathbf{x})$ under some regularity conditions. Therefore, given $s_\theta(\mathbf{x}, t) \approx \nabla_\mathbf{x} \log p_t(\mathbf{x})$, we can get an approximate sample from $p_t(\mathbf{x})$ by running several steps of Langevin MCMC, replacing $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ with $s_\theta(\mathbf{x}, t)$ in the iteration rule.
# 
# Predictor-Corrector samplers combine both numerical solvers for the reverse-time SDE and the Langevin MCMC approach. In particular, we first apply one step of numerical SDE solver to obtain $\mathbf{x}_{t-\Delta t}$ from $\mathbf{x}_t$, which is called the "predictor" step. Next, we apply several steps of Langevin MCMC to refine $\mathbf{x}_t$, such that $\mathbf{x}_t$ becomes a more accurate sample from $p_{t-\Delta t}(\mathbf{x})$. This is the "corrector" step as the MCMC helps reduce the error of the numerical SDE solver. 

# In[ ]:


# @title Define the Predictor-Corrector sampler (double click to expand or collapse)

signal_to_noise_ratio = 0.16  # @param {'type':'number'}

## The number of sampling steps.
num_steps = 500  # @param {'type':'integer'}


def pc_sampler(rng,
               score_model,
               params,
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64,
               num_steps=num_steps,
               snr=signal_to_noise_ratio,
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
      rng: A JAX random state.
      score_model: A `flax.linen.Module` that represents the
        architecture of the score-based model.
      params: A dictionary that contains the parameters of the score-based model.
      marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    time_shape = (jax.local_device_count(), batch_size // jax.local_device_count())
    sample_shape = time_shape + (28, 28, 1)
    rng, step_rng = jax.random.split(rng)
    init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)
    time_steps = jnp.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    for time_step in tqdm(time_steps):
        batch_time_step = jnp.ones(time_shape) * time_step
        # Corrector step (Langevin MCMC)
        grad = pmap_score_fn(score_model, params, x, batch_time_step)
        grad_norm = jnp.linalg.norm(grad.reshape(sample_shape[0], sample_shape[1], -1),
                                    axis=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        x = x + langevin_step_size * grad + jnp.sqrt(2 * langevin_step_size) * z

        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(time_step)
        score = pmap_score_fn(score_model, params, x, batch_time_step)
        x_mean = x + (g ** 2) * score * step_size
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        x = x_mean + jnp.sqrt(g ** 2 * step_size) * z

        # The last step does not include any noise
    return x_mean


# ## Sampling with Numerical ODE Solvers
# 
# For any SDE of the form
# \begin{align*}
# d \mathbf{x} = \mathbf{f}(\mathbf{x}, t) d t + g(t) d \mathbf{w},
# \end{align*}
# there exists an associated ordinary differential equation (ODE)
# \begin{align*}
# d \mathbf{x} = \bigg[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})\bigg] dt,
# \end{align*}
# such that their trajectories have the same mariginal probability density $p_t(\mathbf{x})$. Therefore, by solving this ODE in the reverse time direction, we can sample from the same distribution as solving the reverse-time SDE.
# We call this ODE the *probability flow ODE*.
# 
# Below is a schematic figure showing how trajectories from this probability flow ODE differ from SDE trajectories, while still sampling from the same distribution.
# ![SDE and ODE](https://drive.google.com/uc?id=1CGFbtY2mCjlIY8pjvoGevfa_32d4b1dj)
# 
# Therefore, we can start from a sample from $p_T$, integrate the ODE in the reverse time direction, and then get a sample from $p_0$. In particular, for the SDE in our running example, we can integrate the following ODE from $t=T$ to $0$ for sample generation
# \begin{align*}
# d\mathbf{x} =  -\frac{1}{2}\sigma^{2t} s_\theta(\mathbf{x}, t) dt.
# \end{align*}
# This can be done using many black-box ODE solvers provided by packages such as `scipy`.

# In[ ]:


# @title Define the ODE sampler (double click to expand or collapse)


## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5  # @param {'type': 'number'}



def ode_sampler(rng,
                score_model,
                params,
                #opt_state,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                z=None,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
      rng: A JAX random state.
      score_model: A `flax.linen.Module` object  that represents architecture
        of the score-based model.
      params: A dictionary that contains model parameters.
      marginal_prob_std: A function that returns the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      atol: Tolerance of absolute errors.
      rtol: Tolerance of relative errors.
      z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
      eps: The smallest time step for numerical stability.
    """

    time_shape = (jax.local_device_count(), batch_size // jax.local_device_count())
    sample_shape = time_shape + (28, 28, 1)
    # Create the latent code
    if z is None:
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, sample_shape)
        init_x = z * marginal_prob_std(1.)
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(sample_shape)
        time_steps = jnp.asarray(time_steps).reshape(time_shape)
        score = pmap_score_fn(score_model, params, sample, time_steps)
        return np.asarray(score).reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones(time_shape) * t
        g = diffusion_coeff(t)
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), np.asarray(init_x).reshape(-1),
                              rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = jnp.asarray(res.y[:, -1]).reshape(shape)

    return x


# In[ ]:


# @title Sampling (double click to expand or collapse)


sample_batch_size = 64  # @param {'type':'integer'}
sampler = ode_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

## Load the pre-trained checkpoint from disk.
score_model = ScoreNet(marginal_prob_std_fn, channels=(8, 16, 16, 32), embed_dim=32)
fake_input = jnp.ones((sample_batch_size, 28, 28, 1))
fake_time = jnp.ones((sample_batch_size,))
rng = jax.random.PRNGKey(0)
#params = score_model.init({'params': rng}, fake_input, fake_time)
_ = score_model.init({'params': rng}, fake_input, fake_time)

#lr = 1e-3
#optimiser = flax.optim.Adam().create(params)
#optimiser = optax.adam(learning_rate=lr)
#opt_state = optimiser.init(params)

# with tf.io.gfile.GFile('ckpt.flax', 'rb') as fin:
#     # read in params from ckpt
#     params = from_bytes({}, fin.read(),)
#
#     par_dict = from_bytes({"params": {}, "opt_state": {}}, fin.read(),)
#     opt_state = par_dict["opt_state"]
#     params = par_dict["params"]
#     #optimiser = from_bytes(optimiser, fin.read())

## Generate samples using the specified sampler.
rng, step_rng = jax.random.split(rng)
samples = sampler(rng,
                  score_model,
                  #optimiser.target,
                  params,
                  #opt_state,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size)

## Sample visualization.
samples = jnp.clip(samples, 0.0, 1.0)
samples = jnp.transpose(samples.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

sample_grid = make_grid(torch.tensor(np.asarray(samples)), nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()


# ## Likelihood Computation
# 
# A by-product of the probability flow ODE formulation is likelihood computation. Suppose we have a differentiable one-to-one mapping $\mathbf{h}$ that transforms a data sample $\mathbf{x} \sim p_0$ to a prior distribution $\mathbf{h}(\mathbf{x}) \sim p_T$. We can compute the likelihood of $p_0(\mathbf{x})$ via the following [change-of-variable formula](https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function)
# \begin{align*}
# p_0(\mathbf{x}) = p_T(\mathbf{h}(\mathbf{x})) |\operatorname{det}(J_\mathbf{h}(\mathbf{x}))|,
# \end{align*}
# where $J_\mathbf{h}(\mathbf{x})$ represents the Jacobian of the mapping $\mathbf{h}$, and we assume it is efficient to evaluate the likelihood of the prior distribution $p_T$. 
# 
# The trajectories of an ODE also define a one-to-one mapping from $\mathbf{x}(0)$ to $\mathbf{x}(T)$. For ODEs of the form
# \begin{align*}
# d \mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt,
# \end{align*}
# there exists an [instantaneous change-of-variable formula](https://arxiv.org/abs/1806.07366) that connects the probability of $p_0(\mathbf{x})$ and $p_1(\mathbf{x})$, given by
# \begin{align*}
# p_0 (\mathbf{x}(0)) = e^{\int_0^1 \operatorname{div} \mathbf{f}(\mathbf{x}(t), t) d t} p_1(\mathbf{x}(1)),
# \end{align*}
# where $\operatorname{div}$ denotes the divergence function (trace of Jacobian). 
# 
# In practice, this divergence function can be hard to evaluate for general vector-valued function $\mathbf{f}$, but we can use an unbiased estimator, named [Skilling-Hutchinson estimator](http://blog.shakirm.com/2015/09/machine-learning-trick-of-the-day-3-hutchinsons-trick/), to approximate the trace. Let $\boldsymbol \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The Skilling-Hutchinson estimator is based on the fact that
# \begin{align*}
# \operatorname{div} \mathbf{f}(\mathbf{x}) = \mathbb{E}_{\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}[\boldsymbol\epsilon^\intercal  J_\mathbf{f}(\mathbf{x}) \boldsymbol\epsilon].
# \end{align*}
# Therefore, we can simply sample a random vector $\boldsymbol \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, and then use $\boldsymbol \epsilon^\intercal J_\mathbf{f}(\mathbf{x}) \boldsymbol \epsilon$ to estimate the divergence of $\mathbf{f}(\mathbf{x})$. This estimator only requires computing the Jacobian-vector product $J_\mathbf{f}(\mathbf{x})\boldsymbol \epsilon$, which is typically efficient.
# 
# As a result, for our probability flow ODE, we can compute the (log) data likelihood with the following
# \begin{align*}
# \log p_0(\mathbf{x}(0)) = \log p_1(\mathbf{x}(1)) -\frac{1}{2}\int_0^1 \frac{d[\sigma^2(t)]}{dt} \operatorname{div} s_\theta(\mathbf{x}(t), t) dt.
# \end{align*}
# With the Skilling-Hutchinson estimator, we can compute the divergence via
# \begin{align*}
# \operatorname{div} s_\theta(\mathbf{x}(t), t) = \mathbb{E}_{\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}[\boldsymbol\epsilon^\intercal  J_{s_\theta}(\mathbf{x}(t), t) \boldsymbol\epsilon].
# \end{align*}
# Afterwards, we can compute the integral with numerical integrators. This gives us an unbiased estimate to the true data likelihood, and we can make it more and more accurate when we run it multiple times and take the average. The numerical integrator requires $\mathbf{x}(t)$ as a function of $t$, which can be obtained by the probability flow ODE sampler.

# In[ ]:


# @title Define the likelihood function (double click to expand or collapse)

def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and
        standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[2:])
    return -N / 2. * jnp.log(2 * np.pi * sigma ** 2) \
        - jnp.sum(z ** 2, axis=(2, 3, 4)) / (2 * sigma ** 2)


def ode_likelihood(rng,
                   x,
                   score_model,
                   params,
                   marginal_prob_std,
                   diffusion_coeff,
                   batch_size=64,
                   eps=1e-5):
    """Compute the likelihood with probability flow ODE.

    Args:
      rng: A JAX random state.
      x: Input data.
      score_model: A `flax.linen.Module` instance that represents the architecture
        of the score-based model.
      params: A dictionary that contains model parameters.
      marginal_prob_std: A function that gives the standard deviation of the
        perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the
        forward SDE.
      batch_size: The batch size. Equals to the leading dimension of `x`.
      eps: A `float` number. The smallest time step for numerical stability.

    Returns:
      z: The latent code for `x`.
      bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    rng, step_rng = jax.random.split(rng)
    epsilon = jax.random.normal(step_rng, x.shape)

    def divergence_eval(sample, time_steps, epsilon):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        score_e_fn = lambda x: \
            jnp.sum(pmap_score_fn(score_model, params, x, time_steps) * epsilon)
        grad_score_e = jax.grad(score_e_fn)(sample)
        return jnp.sum(grad_score_e * epsilon, axis=(2, 3, 4))

    shape = x.shape
    time_shape = (shape[0], shape[1])

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(shape)
        time_steps = jnp.asarray(time_steps, dtype=jnp.float32).reshape(time_shape)
        score = pmap_score_fn(score_model, params, sample, time_steps)
        return np.asarray(score).reshape((-1,)).astype(np.float64)

    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        # Obtain x(t) by solving the probability flow ODE.
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(shape)
        time_steps = jnp.asarray(time_steps, dtype=jnp.float32).reshape(time_shape)
        # Compute likelihood.
        div = divergence_eval(sample, time_steps, epsilon)
        return np.asarray(div).reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = np.ones(time_shape) * t
        sample = x[:-shape[0] * shape[1]]
        logp = x[-shape[0] * shape[1]:]
        g = diffusion_coeff(t)
        sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)

    init = jnp.concatenate([x.reshape((-1,)), jnp.zeros((shape[0] * shape[1],))],
                           axis=0)
    # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.), np.asarray(init),
                              rtol=1e-5, atol=1e-5,
                              method='RK45')
    zp = jnp.asarray(res.y[:, -1])
    z = zp[:-shape[0] * shape[1]].reshape(shape)
    delta_logp = zp[-shape[0] * shape[1]:].reshape((shape[0], shape[1]))
    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[2:])
    bpd = bpd / N + 8.
    return z, bpd


# In[ ]:
"hello"

# @title Compute likelihood on the dataset (double click to expand or collapse)

batch_size = 64  # @param {'type':'integer'}

dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

## Load the pre-trained checkpoint from disk.
score_model = ScoreNet(marginal_prob_std_fn)
fake_input = jnp.ones((sample_batch_size, 28, 28, 1))
fake_time = jnp.ones((sample_batch_size,))
rng = jax.random.PRNGKey(0)
params = score_model.init({'params': rng}, fake_input, fake_time)
optimiser = flax.optim.Adam().create(params)
with tf.io.gfile.GFile('ckpt.flax', 'rb') as fin:
    optimiser = from_bytes(optimiser, fin.read())

all_bpds = 0.
all_items = 0
rng = jax.random.PRNGKey(0)
try:
    tqdm_data = tqdm.notebook.tqdm(data_loader)
    for x, _ in tqdm_data:
        x = x.permute(0, 2, 3, 1).cpu().numpy().reshape((
            jax.local_device_count(), -1, 28, 28, 1
        ))
        # uniform dequantization
        rng, step_rng = jax.random.split(rng)
        z = jax.random.uniform(step_rng, x.shape)
        x = (x * 255. + z) / 256.
        _, bpd = ode_likelihood(step_rng, x, score_model, optimiser.target,
                                marginal_prob_std_fn,
                                diffusion_coeff_fn,
                                batch_size, eps=1e-5)
        all_bpds += bpd.sum()
        all_items += bpd.shape[0] * bpd.shape[1]
        tqdm_data.set_description("Average bits/dim: {:5f}".format(all_bpds / all_items))

except KeyboardInterrupt:
    # Remove the error message when interuptted by keyboard or GUI.
    pass

# ## Further Resources
# 
# If you're interested in learning more about score-based generative models, the following papers would be a good start:
# 
# * Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. "[Score-Based Generative Modeling through Stochastic Differential Equations.](https://arxiv.org/pdf/2011.13456.pdf)" International Conference on Learning Representations, 2021.
# * Jonathan Ho, Ajay Jain, and Pieter Abbeel. "[Denoising diffusion probabilistic models.](https://arxiv.org/pdf/2006.11239.pdf)" Advances in Neural Information Processing Systems. 2020.
# *    Yang Song, and Stefano Ermon. "[Improved Techniques for Training Score-Based Generative Models.](https://arxiv.org/pdf/2006.09011.pdf)" Advances in Neural Information Processing Systems. 2020.
# *   Yang Song, and Stefano Ermon. "[Generative modeling by estimating gradients of the data distribution.](https://arxiv.org/pdf/1907.05600.pdf)" Advances in Neural Information Processing Systems. 2019.
# 
#
