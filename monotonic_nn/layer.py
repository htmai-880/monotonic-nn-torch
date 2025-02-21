import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional, Callable, Union, Iterable

import numpy as np
from numpy.typing import ArrayLike



class MonoLinear(nn.Linear):
    """MonoDense layer, inspired from the official repository (https://github.com/airtai/monotonic-nn/blob/main/airt/_components/mono_dense_layer.py)
    """
    def __init__(self,
                 in_features : int,
                 out_features : int,
                 bias : bool = True,
                 monotonicity_indicator : Union[int, Iterable[float]] = 1,
                 is_convex : bool = False,
                 is_concave : bool = False,
                 activation_weights : ArrayLike = (7.0, 7.0, 2.0),
                 act : Optional[Union[str, callable]]= None
                ):
        """MonoDense layer, inspired from the official repository. Just like how the original implementation inherits from the Dense layer in Keras,
        this implementation inherits from the Linear layer in PyTorch.

        Args:
            in_features (int): Input dimension.
            out_features (int): Output dimension.
            bias (bool, optional): Whether to use bias. Defaults to True.
            monotonicity_indicator (Union[int, Iterable[float]], optional): Monotonicity indicator. If it is an integer,
                it is used as a constant for all the features. If it is an iterable, it must have the same size as the output features. 
                1 at position i indicates that all output features are increasing with respect to the i-th input feature.
                -1 at position i indicates that all output features are decreasing with respect to the i-th input feature.
                0 at position i indicates that the output features are not expected to depend monotonically on the i-th input feature.
                Defaults to 1.
            is_convex (bool, optional): Whether all the output features are convex with respect to the input features. Defaults to False.
            is_concave (bool, optional): Whether all the output features are concave with respect to the input features. Defaults to False.
            activation_weights (ArrayLike, optional): Ratios of the number of convex, concave, and saturated neurons with respect to inputs. Defaults to (7.0, 7.0, 2.0).
            act (Optional[Union[str, callable]], optional): Base activation function to be found in torch.nn. The paper requires the base activation function to be
                convex, nondecreasing and zero at zero.
                Defaults to None.
        """

        super(MonoLinear, self).__init__(in_features, out_features, bias)

        # Make monotonicity indicator
        if isinstance(monotonicity_indicator, int):
            self.monotonicity_indicator = monotonicity_indicator
        else:
            # it is a tensor of size (in_features,)                
            assert len(monotonicity_indicator) == in_features, "Monotonicity indicator must have the same size as the output features"
            monotonicity_indicator_ = torch.tensor(monotonicity_indicator, dtype=torch.float32)
            self.register_buffer("monotonicity_indicator", monotonicity_indicator_)

        assert not (is_convex and is_concave), "Cannot be both convex and concave"

        self.is_convex = is_convex
        self.is_concave = is_concave
        self.activation_weights = activation_weights
        if act is None:
            self.org_act = None
        else:
            self.org_act = getattr(nn, act)()

    def activation(self, x):
        """Activation function of the layer. It will depend on the base activation function and the s parameter (activation weights).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        assert x.size(-1) == self.out_features, "The input must have the same size as the output features"
        return activation_monolinear(
            x,
            self.activation_weights,
            self.org_act,
            self.is_convex,
            self.is_concave
        )

    def edit_weights(self, W):
        """Helper function to edit the weights of the layer accordingly to the required monotonicity constraints.

        Args:
            W (torch.Tensor): Weights of the layer.

        Returns:
            torch.Tensor: Edited weights.
        """
        assert W.size() == (self.out_features, self.in_features), "Weights must have the same size as the original weights"
        if isinstance(self.monotonicity_indicator, int):
            if self.monotonicity_indicator == 0:
                return W
            else:
                return torch.abs(W) * self.monotonicity_indicator
        # Case where it is a tensor
        ind = self.monotonicity_indicator.unsqueeze(0) # (1, in_features)
        # The columns of W are multiplied by the indicator when it is not zero
        mask = ind == 0 # (in_features, 1)
        new_W = torch.where(
            mask,
            W,
            torch.abs(W) * self.monotonicity_indicator
        )
        return new_W

    def forward(self, x):
        """Forward pass of the layer under monotonicity and convexity constraints.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        W = self.weight
        b = self.bias
        # Transform the weights
        W_ = self.edit_weights(W)
        out = F.linear(x, W_, b)
        if not self.org_act is None:
            out = self.activation(out)
        return out
    

def activation_monolinear(
        x : torch.Tensor,
        activation_weights : ArrayLike,
        org_act = None,
        is_convex : bool = False,
        is_concave : bool = False
        ):
    if org_act is None:
        return x
    if is_convex:
        return org_act(x)
    elif is_concave:
        return -org_act(-x)
    else:
        assert len(activation_weights) == 3
        normalized_activation_weights = np.array(activation_weights) / sum(activation_weights)

    out_features = x.size(-1)
    
    s_convex = round(normalized_activation_weights[0] * out_features)
    s_concave = round(normalized_activation_weights[1] * out_features)
    s_saturated = out_features - s_convex - s_concave

    out = []
    if s_convex > 0:
        x_convex, x_concave, x_saturated = torch.split(
            x,
            (s_convex, s_concave, s_saturated),
            dim=-1
        )
    y_convex = org_act(x_convex)
    y_concave = -org_act(-x_concave)
    act_one = org_act(torch.ones(1, device=x.device, dtype=x.dtype))
    y_saturated = torch.where(
        x_saturated < 0,
        org_act(x_saturated + 1) - act_one,
        act_one - org_act(1 - x_saturated)
    )
    out = torch.cat((y_convex, y_concave, y_saturated), dim=-1)
    return out