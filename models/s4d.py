"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Function aliases
contract = torch.einsum

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

# Try CUDA extension
try:
    from extensions.kernels.vandermonde import log_vandermonde_cuda
    has_cuda_extension = True
    print("CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) found.", flush=True)
except:
    print(
        "CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) not found. Install by going to extensions/kernels/ and running `python setup.py install`, for improved speed and memory efficiency. Note that the kernel changed for state-spaces 4.0 and must be recompiled.", flush=True
    )
    has_cuda_extension = False

# Try pykeops
try:
    import pykeops
    from pykeops.torch import Genred
    has_pykeops = True
    print("Pykeops installation found.", flush=True)

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
        return tensors

    def log_vandermonde_keops(v, x, L):
        expr = 'ComplexMult(v, ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'v = Vj(2)',
                'x = Vj(2)',
                'l = Vi(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        v, x, l = _broadcast_dims(v, x, l)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(v, x, l, backend='GPU')
        return 2*_r2c(r).real

except ImportError:
    has_pykeops = False
    if not has_cuda_extension:
        print(
            "Falling back on slow Cauchy and Vandermonde kernel. Install at least one of pykeops or the CUDA extension for better speed and memory efficiency.", flush=True
        )

def log_vandermonde_naive(v, x, L, conj=True):
    """
    v: (..., N)
    x: (..., N)
    returns: (..., L) \sum v x^l
    """
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = contract('... n, ... n l -> ... l', v, vandermonde_matrix) # (... L)
    return 2*vandermonde_prod.real

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X
    
class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, backend = 'cuda', ssm_init=None, lr=None, train_log_dt=True, train_C=True, train_log_A_real=True, train_A_imag=True):
        super().__init__()

        # Generate dt
        self.H = d_model
        self.N = N
        self.channels = 1
        self.backend = backend
        dt_min = 0.001
        dt_max = 0.1

        if ssm_init is None:
            log_dt = torch.rand(self.H) * (
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
        else:
            log_dt = ssm_init['log_dt']

        if ssm_init is None:
            C = torch.randn(self.channels, self.H, self.N // 2, dtype=torch.cfloat)
        else:
            C = ssm_init['C']
        self.C = nn.Parameter(torch.view_as_real(C), requires_grad=train_C)
        self.register("log_dt", log_dt, lr=lr if train_log_dt else 0.0)

        if ssm_init is None:
            log_A_real = torch.log(0.5 * torch.ones(self.H, self.N//2))
        else:
            log_A_real = ssm_init['log_A_real']
        
        if ssm_init is None:
            A_imag = math.pi * repeat(torch.arange(self.N//2), 'n -> h n', h=self.H)
        else:
            A_imag = ssm_init['A_imag']

        self.register("log_A_real", log_A_real, lr=lr if train_log_A_real else 0.0)
        self.register("A_imag", A_imag, lr=lr if train_A_imag else 0.0)

        # Dispatch which Vandermonde kernel to use
        if has_cuda_extension and C.dtype == torch.cfloat and C.device.type == 'cuda' and self.backend == 'cuda':
            self.log_vandermonde = log_vandermonde_cuda
        elif has_pykeops and self.backend in ['cuda', 'keops']:
            self.log_vandermonde = log_vandermonde_keops
        else:
            self.log_vandermonde = log_vandermonde_naive

    def materialize_parms(self):
        dt = torch.exp(self.log_dt) # (H)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)
        C = torch.view_as_complex(self.C) # (H N)
        return dt, A, C

    def dicretize(self, dt, A, C):
        dtA = A * dt.unsqueeze(-1)  # (H N)

        B = (1. - dtA/2).reciprocal() * dt.unsqueeze(-1) # or * dtA / A
        dC = C * B
        dA = (1. + dtA/2) / (1. - dtA/2)

        return dA, dC
    
    def valdemond_kernel(self, dA, dC, L):
        K = self.log_vandermonde(dC, dA.log(), L)

        K = K.view(-1, self.channels, self.H, L) # (1+B C H L)
        K = K[-1, :, :, :] # (C H L)
        return K

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt, A, C = self.materialize_parms()

        # Discretize
        dA, dC = self.dicretize(dt, A, C)

        # Vandermonde multiplication
        K = self.valdemond_kernel(dA, dC, L)

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, ssm_init=None, train_D=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        if ssm_init is not None:
            self.D = nn.Parameter(ssm_init['D'], requires_grad=train_D)
        else:
            self.D = nn.Parameter(torch.randn(self.h), requires_grad=train_D)

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, ssm_init=ssm_init, **kernel_args)

        # Pointwise
        self.activation = nn.Identity() #ReLU() #GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

        # Output linear mixer
        # self.output_linear = nn.Conv1d(self.h, self.h, kernel_size=1)

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified

    def get_ssm(self):
        # Materialize parameters
        dt, A, C = self.kernel.materialize_parms()

        # Discretize
        dA, dC = self.kernel.dicretize(dt, A, C)

        return [dt.detach().cpu(), A.detach().cpu(), C.detach().cpu(), dA.detach().cpu(), dC.detach().cpu(), self.D.detach().cpu()]
