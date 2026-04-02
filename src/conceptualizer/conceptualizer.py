from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Buffer, Parameter


class BlockLinear(nn.Module):
    def __init__(
        self,
        context_dims,
        width,
        bias=True,
        dtype=None,
        *args,
        **kwargs,
    ):
        """
        args:
            context_dims [int]: how many exogenous variables in causal model
            width [int]: how expressive they are
            bias [bool]: whether to include a bias term
            dtype [torch.dtype]: data type to use for tensors
        """
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.block_mask = Buffer(torch.eye(context_dims).kron(torch.ones(width, width)))
        num_features = context_dims * width
        self.weight = Parameter(
            torch.empty((num_features, num_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters(width)

    def reset_parameters(self, in_width):
        """
        Sets the weight and bias parameters to their initial values.
        """
        # Standard linear initialization given the input size of each
        # fully-connected linear block
        bound = 1 / np.sqrt(in_width)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        args:
            input [torch.Tensor]: input tensor of shape (batch_size, context_dims * width)
        returns:
            output [torch.Tensor]: output tensor of shape (batch_size, context_dims * width)
        """
        # masked linear layer
        return nn.functional.linear(input, self.weight * self.block_mask, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class CompressBlockLinear(BlockLinear):
    def __init__(
        self,
        context_dims,
        width_in,
        width_out,
        bias=True,
        dtype=None,
        *args,
        **kwargs,
    ):
        """
        args:
            context_dims [int]: how many exogenous variables in causal model.
            width_in [int]: how expressive they are
            width_out [int]: how expressive they are
            bias [bool]: whether to include a bias term
            dtype [torch.dtype]: data type to use for tensors
        """
        factory_kwargs = {"dtype": dtype}
        super().__init__(1, 1)
        self.block_mask = Buffer(
            torch.eye(context_dims).kron(torch.ones(width_out, width_in))
        )
        # Input is (batch_size, context_dims * width_in)
        # Output is (batch_size, context_dims * width_out)
        self.in_features = context_dims * width_in
        self.out_features = context_dims * width_out
        self.weight = Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters(width_in)


class Intervenable(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        in_width,
        out_width,
        concepts,
        mask=None,
        bias=True,
        dtype=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            in_dim [int]: number of input dimensions
            out_dim [int]: number of output dimensions
            in_width [int]: width of input
            out_width [int]: width of output
            concepts [iterable[str]]: names of concepts
            mask [torch.Tensor]: mask to apply to the weights
            bias [bool]: whether to include a bias term
            dtype [torch.dtype]: data type to use for tensors
        """
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_width = in_width
        self.out_width = out_width
        self.mask = Buffer(data=mask) if mask is not None else None
        self.obs_weight = Parameter(
            torch.empty((out_dim * out_width, in_dim * in_width), **factory_kwargs)
        )
        self.ivn_weight = Parameter(
            torch.empty((out_dim * out_width, in_dim * in_width), **factory_kwargs)
        )
        self.grouper = nn.LPPool2d(2, (out_width, in_width))
        self.pooler = nn.AvgPool2d((out_width, in_width))
        self.concepts = concepts

        if in_dim != out_dim:
            raise ValueError(
                "number exogenous vars differs from endogenous vars, changing causal semantics, so `intervention` is no longer sound"
            )
        num_concepts = in_dim
        self.block_mask = Buffer(
            torch.eye(num_concepts).kron(torch.ones(out_width, in_width)).to(bool)
        )
        self.block_mask_diag = Buffer(torch.eye(in_dim * in_width).to(bool))

        if bias:
            self.bias = Parameter(torch.empty(out_dim * out_width, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(in_dim, in_width)

    def reset_parameters(self, in_dim, in_width):
        bound = 1 / np.sqrt((in_dim - 1) * in_width) if in_dim > 1 else 1
        nn.init.uniform_(self.obs_weight, -bound, bound)
        nn.init.uniform_(self.ivn_weight, -bound, bound)

        # These are both set to 1 in forward method through self.block_mask.
        # This is repetitive but useful to show the effective weight
        self.obs_weight.data[self.block_mask] = self.block_mask_diag[
            self.block_mask
        ].to(self.obs_weight.data.dtype)
        self.ivn_weight.data[self.block_mask] = self.block_mask_diag[
            self.block_mask
        ].to(self.ivn_weight.data.dtype)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, batch_labels):
        """
        Args:
            input [torch.Tensor]: input tensor of shape (batch_size, in_dim * in_width)
        Returns:
            output [torch.Tensor]: output tensor of shape (batch_size, out_dim * out_width)
        """

        # ensures coherent indices between end/exogenous
        obs_weight = self.block_mask_diag.to(int) + self.obs_weight * (
            1 - self.block_mask.to(int)
        )
        ivn_weight = self.block_mask_diag.to(int) + self.ivn_weight * (
            1 - self.block_mask.to(int)
        )
        if batch_labels == ["obs"]:
            weight = obs_weight
        else:
            ivn_concepts = [con for con in self.concepts if con != "obs"]
            ivn_idcs = [ivn_concepts.index(con) for con in batch_labels]
            ivn_mask = torch.zeros(
                self.out_dim, self.in_dim, dtype=bool, device=input.device
            )
            ivn_mask[:, ivn_idcs] = 1  # use ivn epsilon
            obs_mask = ~ivn_mask
            obs_mask[ivn_idcs, :] = (
                0  # zero out incoming eps to ivn Xs other than ivn eps
            )
            wide_block = torch.ones(
                self.out_width, self.in_width, dtype=bool, device=input.device
            )
            ivn_mask = ivn_mask.kron(wide_block)
            obs_mask = obs_mask.kron(wide_block)
            weight = obs_mask * obs_weight + ivn_mask * ivn_weight
        self.grouped = (
            self.grouper(weight.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        )
        self.pooled = (
            self.pooler(weight.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        )
        self.unpooled = weight
        if self.mask is None:
            return nn.functional.linear(input, weight, self.bias)
        else:
            return nn.functional.linear(input, weight * self.mask, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_dim * self.in_width,
            self.out_dim * self.out_width,
            self.bias is not None,
        )


class IntervenableBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        in_width,
        out_width,
        concepts,
        mask=None,
        bias=True,
        dtype=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            in_dim [int]: number of input dimensions. Must be equal to len(concepts) - 1
            out_dim [int]: number of output dimensions. Must be equal to in_dim
            in_width [int]: width of input.
            out_width [int]: width of output. Must be equal to in_wideth.
            concepts [iterable[str]]: names of concepts
            mask [torch.Tensor]: mask to apply to the weights
            bias [bool]: whether to include a bias term
            dtype [torch.dtype]: data type to use for tensors
        """
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_width = in_width
        self.out_width = out_width
        self.mask = Buffer(data=mask) if mask is not None else None
        num_concepts = len(concepts) - 1
        self.block_mask = (
            Buffer(torch.eye(num_concepts).kron(torch.ones(out_width, in_width)))
            if num_concepts > 0
            else 1
        )
        self.obs_weight = Parameter(
            torch.empty((out_dim * out_width, in_dim * in_width), **factory_kwargs)
        )
        self.ivn_weight = Parameter(
            torch.empty((out_dim * out_width, in_dim * in_width), **factory_kwargs)
        )
        self.concepts = concepts
        if bias:
            self.bias = Parameter(torch.empty(out_dim * out_width, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(in_width)

    def reset_parameters(self, in_width):
        """
        Sets the weight and bias parameters to their initial values.
        """
        # Standard linear initialization given the input size of each
        # fully-connected linear block
        bound = 1 / np.sqrt(in_width)
        nn.init.uniform_(self.obs_weight, -bound, bound)
        nn.init.uniform_(self.ivn_weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, batch_labels):
        """
        Args:
            input [torch.Tensor]: input tensor of shape (batch_size, in_features)
        Returns:
            output [torch.Tensor]: output tensor of shape (batch_size, out_features)
        """
        if batch_labels == ["obs"]:
            weight = self.obs_weight
        else:
            weight = self.obs_weight
            ivn_concepts = [con for con in self.concepts if con != "obs"]
            ivn_idcs = [ivn_concepts.index(con) for con in batch_labels]
            ivn_mask = torch.zeros(
                self.out_dim, self.in_dim, dtype=bool, device=input.device
            )
            ivn_mask[:, ivn_idcs] = 1  # use ivn epsilon
            obs_mask = ~ivn_mask
            wide_block = torch.ones(
                self.out_width, self.in_width, dtype=bool, device=input.device
            )
            ivn_mask = ivn_mask.kron(wide_block)
            obs_mask = obs_mask.kron(wide_block)
            weight = obs_mask * self.obs_weight + ivn_mask * self.ivn_weight
            weight = weight * self.block_mask
        if self.mask is None:
            return nn.functional.linear(input, weight, self.bias)
        else:
            return nn.functional.linear(input, weight * self.mask, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_dim * self.in_width,
            self.out_dim * self.out_width,
            self.bias is not None,
        )


def dec_conceptualizer(
    eps_dim,
    eps_in_width,
    eps_out_width,
    eps_depth,
    c_dim,
    c_width,
    concepts,
    decode_dim,
    arch_flag,
    *args,
    **kwargs,
):
    """
    Args:
        eps_dim [int]: how many exogenous variables in causal model. Must equal len(concepts) - 1
        eps_in_width [int]: how expressive they are
        eps_out_width [int]: bottleneck size
        eps_depth [int]: how many layers of expressiveness
        c_dim [int]: how many endogenous variables in causal model
        c_width [int]: how expressive they are
        concepts [iterable[str]]: names of concepts
        decode_dim [int]: how many dimensions to decode to
        arch_flag [str]: architecture flag
            - "concepts": make full use of concept labels
            - "vanilla-obs": ignore conceptualizer module and proceed with base encoder/decoder and only observational setting
            - "vanilla-pooled": ignore conceptualizer module and proceed with base encoder/decoder and pooled concepts and obs settings
            - "single-pooled-concept": ignore concept labels, effectively pooling all concepts and obs settings together (but still pass through conceptualizer)
    Returns:
        ivn_eps [nn.Module]: layer to intervene on eps_dim
        expressive_layer [nn.Module]: layer to make eps_dim more expressive
        causal_layer [nn.Module]: layer to make eps_dim more causal
        unpool [nn.Module]: layer to unpool the output of the causal layer
    """
    # intervenable error layer
    # take inputs of shape (batch_size, eps_dim * eps_in_width)
    # and outputs of shape (batch_size, eps_dim * eps_in_width)
    ivn_eps = IntervenableBlock(
        in_dim=eps_dim,
        in_width=eps_in_width,
        out_dim=eps_dim,
        out_width=eps_in_width,
        concepts=concepts,
    )

    # expressive layers
    widths = np.linspace(eps_in_width, eps_out_width, eps_depth).astype(int)
    w_pairs = zip(widths[:-1], widths[1:])

    def unchained(w_pair):
        return (
            CompressBlockLinear(eps_dim, w_pair[0], w_pair[1]),
            nn.BatchNorm1d(eps_dim * w_pair[1]),
            nn.GELU(),
        )

    # expressive_layer takes inputs of shape (batch_size, eps_dim * eps_in_width)
    # and outputs of shape (batch_size, eps_dim * eps_out_width)
    expressive_layer = nn.Sequential(*chain(*(unchained(w_pair) for w_pair in w_pairs)))

    # causal layer takes inputs of shape (batch_size, eps_dim * eps_out_width)
    # and outputs of shape (batch_size, c_dim * c_width)
    causal_layer = Intervenable(
        in_dim=eps_dim,
        in_width=eps_out_width,
        out_dim=c_dim,
        out_width=c_width,
        concepts=concepts,
    )

    # unpool takes inputs of shape (batch_size, c_dim * c_width)
    # and outputs of shape (batch_size, decode_dim)
    unpool = nn.Linear(c_dim * c_width, decode_dim)

    # If chained together, takes in input of size (batch_size, eps_dim * eps_in_width)
    # and outputs of size (batch_size, decode_dim)
    return ivn_eps, expressive_layer, causal_layer, unpool


def bimodal_regularizer(weights, alpha=0.5, mu0=0.03, mu1=0.7, sigma0=0.01, sigma1=0.1):
    """
    Computes a regularization term that encourages the weights to follow a bimodal distribution.

    Args:
        weights (torch.Tensor): Tensor containing the weights (or activations) to regularize.
        alpha (float): Mixing coefficient for the first Gaussian (mode at mu0). Should be in [0, 1].
        mu0 (float): Mean of the first (narrow) Gaussian mode (typically the 'inactive' mode).
        mu1 (float): Mean of the second (narrow) Gaussian mode (typically the 'active' mode).
        sigma0 (float): Standard deviation of the first Gaussian.
        sigma1 (float): Standard deviation of the second Gaussian.

    Returns:
        torch.Tensor: A scalar tensor representing the regularization loss.
    """
    # extract off-diag elements
    weights = weights[~np.eye(len(weights), dtype=bool)]

    # Calculate the probability density under each Gaussian
    gauss0 = (1.0 / torch.sqrt(2 * torch.pi * torch.tensor(sigma0) ** 2)) * torch.exp(
        -((weights - mu0) ** 2) / (2 * sigma0**2)
    )
    gauss1 = (1.0 / torch.sqrt(2 * torch.pi * torch.tensor(sigma1) ** 2)) * torch.exp(
        -((weights - mu1) ** 2) / (2 * sigma1**2)
    )

    # Compute the mixture density
    p_target = alpha * gauss0 + (1 - alpha) * gauss1

    # Avoid numerical issues with log(0)
    p_target = torch.clamp(p_target, min=1e-10)

    # Compute the negative log-likelihood loss for the bimodal target
    reg_loss = -torch.log(p_target)

    # Return the mean loss over weights
    return reg_loss.mean()
