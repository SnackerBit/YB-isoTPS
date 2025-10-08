from enum import Enum
import numpy as np
import scipy
import scipy.linalg

class Backend(Enum):
    NUMPY = 1
    JAX_CPU = 2
    JAX_GPU = 3

backend = Backend.NUMPY
array_type = np.ndarray
sparse_array_type = scipy.sparse.csr_array
nan = np.nan
inf = np.inf
dtype_complex = np.complex128
newaxis = np.newaxis

def set_backend(backend_enum):
    if backend_enum == Backend.NUMPY:
        backend = Backend.NUMPY
        array_type = np.ndarray
        sparse_array_type = scipy.sparse.csr_array
        nan = np.nan
        inf = np.inf
        dtype_complex = np.complex128
        newaxis = np.newaxis
    elif backend_enum == BACKEND.JAX_CPU:
        raise NotImplementedError("backend \"JAX_CPU\" is not implemented.")
    elif backend_enum == BACKEND.JAX_GPU:
        raise NotImplementedError("backend \"JAX_GPU\" is not implemented.")

log_matrix_ops = False

logged_matrix_ops_qr = {}
logged_matrix_ops_svd = {}
logged_matrix_ops_eigh = {}
logged_matrix_ops_tensordot = {}
logged_matrix_ops_kron = {}
logged_matrix_ops_trace = {}
logged_matrix_ops_dot = {}
logged_matrix_ops_reshape = {}
logged_matrix_ops_transpose = {}

def reset_log_matrix_ops():
    global logged_matrix_ops_qr
    global logged_matrix_ops_svd
    global logged_matrix_ops_eigh
    global logged_matrix_ops_tensordot
    global logged_matrix_ops_kron
    global logged_matrix_ops_trace
    global logged_matrix_ops_dot
    global logged_matrix_ops_reshape
    global logged_matrix_ops_transpose
    logged_matrix_ops_qr = {}
    logged_matrix_ops_svd = {}
    logged_matrix_ops_eigh = {}
    logged_matrix_ops_tensordot = {}
    logged_matrix_ops_kron = {}
    logged_matrix_ops_trace = {}
    logged_matrix_ops_dot = {}
    logged_matrix_ops_reshape = {}
    logged_matrix_ops_transpose = {}


def array(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.array(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"array\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"array\" is not implemented for backend \"JAX_CPU\".")

def safe_svd(A, full_matrices=True):
    """
    Computes the Singular Value Decomposition A = U@S@V. If the numpy svd does not converge,
    scipy's SVD with the less efficient but more involved general rectangular approach is used,
    which is more likely to converge.

    Parameters
    ----------
    A : np.ndarray of shape (n, m)
        The matrix that should be decomposed using SVD.
    full_matrices : bool, optional
        determines the shape of the output matrices. See official
        numpy documentation for more details.
    
    Returns
    -------
    U : np.ndarray of shape (n, chi)
        isometric matrix. A = U@np.diag(S)@V.
    S : np.ndarray of shape (chi, )
        vector containing the real singular values >= 0. A = U@np.diag(S)@V.
    V : np.ndarray of shape (chi, m)
        V.T is an isometric matrix. A = U@np.diag(S)@V.
    """
    if backend == Backend.NUMPY:
        try:
            return np.linalg.svd(A, full_matrices=full_matrices)
        except np.linalg.LinAlgError:
            if np.isnan(A).any() or np.isinf(A).any():
                print("[WARNING]: Trying to perform SVD on a matrix with nan or inf entries!")
            U, S, V = scipy.linalg.svd(A, full_matrices=full_matrices, lapack_driver='gesvd')
            if np.isnan(U).any() or np.isinf(U).any() or np.isnan(S).any() or np.isinf(S).any() or np.isnan(V).any() or np.isinf(V).any():
                print("[WARNING] scipy SVD did not converge!")
                m, n = A.shape
                k = min(m, n)
                return np.zeros(m, k), np.zeros(k), np.zeros(k, n)
            return U, S, V
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"safe_svd\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"safe_svd\" is not implemented for backend \"JAX_CPU\".")

def sum(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.sum(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"sum\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"sum\" is not implemented for backend \"JAX_CPU\".")

def argsort(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.argsort(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"argsort\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"argsort\" is not implemented for backend \"JAX_CPU\".")

def norm(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.linalg.norm(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"norm\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"norm\" is not implemented for backend \"JAX_CPU\".")

def random(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.random.random(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"random\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"random\" is not implemented for backend \"JAX_CPU\".")

def sqrt(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.sqrt(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"sqrt\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"sqrt\" is not implemented for backend \"JAX_CPU\".")

def diag(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.diag(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"diag\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"diag\" is not implemented for backend \"JAX_CPU\".")

def abs(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.abs(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"abs\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"abs\" is not implemented for backend \"JAX_CPU\".")

def conj(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.conj(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"conj\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"conj\" is not implemented for backend \"JAX_CPU\".")

def eye(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.eye(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"eye\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"eye\" is not implemented for backend \"JAX_CPU\".")

def isclose(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.isclose(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"isclose\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"isclose\" is not implemented for backend \"JAX_CPU\".")

def all(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.all(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"all\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"all\" is not implemented for backend \"JAX_CPU\".")

def allclose(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.allclose(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"allclose\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"allclose\" is not implemented for backend \"JAX_CPU\".")

def real_if_close(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.real_if_close(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"real_if_close\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"real_if_close\" is not implemented for backend \"JAX_CPU\".")

def floor(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.floor(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"floor\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"floor\" is not implemented for backend \"JAX_CPU\".")

def ascontiguousarray(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.ascontiguousarray(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"ascontiguousarray\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"ascontiguousarray\" is not implemented for backend \"JAX_CPU\".")

def random_unitary(N):
    """
    Returns a random (N, N) unitary drawn from the Haar measure
    
    Parameters
    ----------
    N: int
        dimension of the unitary
    
    Returns
    -------
    U : backend.array_type of shape (N, N)
        random unitary
    """
    if backend == Backend.NUMPY:
        from scipy.stats import unitary_group
        return unitary_group.rvs(N)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"random_unitary\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"random_unitary\" is not implemented for backend \"JAX_CPU\".")

def expm(*args, **kwargs):
    if backend == Backend.NUMPY:
        return scipy.linalg.expm(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"expm\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"expm\" is not implemented for backend \"JAX_CPU\".")

def flipud(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.flipud(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"flipud\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"flipud\" is not implemented for backend \"JAX_CPU\".")

def sparse_kron(*args, **kwargs):
    if backend == Backend.NUMPY:
        return scipy.sparse.kron(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"sparse_kron\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"sparse_kron\" is not implemented for backend \"JAX_CPU\".")

def min(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.min(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"min\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"min\" is not implemented for backend \"JAX_CPU\".")

def rand(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.random.rand(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"min\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"min\" is not implemented for backend \"JAX_CPU\".")

def isnan(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.isnan(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"isnan\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"isnan\" is not implemented for backend \"JAX_CPU\".")

def isinf(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.isinf(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"isinf\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"isinf\" is not implemented for backend \"JAX_CPU\".")

def sign(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.sign(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"sign\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"sign\" is not implemented for backend \"JAX_CPU\".")

def real(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.real(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"real\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"real\" is not implemented for backend \"JAX_CPU\".")

def where(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.where(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"where\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"where\" is not implemented for backend \"JAX_CPU\".")

def imag(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.imag(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"imag\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"imag\" is not implemented for backend \"JAX_CPU\".")

def zeros(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.zeros(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"zeros\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"zeros\" is not implemented for backend \"JAX_CPU\".")

def ones(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.ones(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"ones\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"ones\" is not implemented for backend \"JAX_CPU\".")

def log(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.log(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"log\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"log\" is not implemented for backend \"JAX_CPU\".")

def arctan(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.arctan(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"arctan\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"arctan\" is not implemented for backend \"JAX_CPU\".")

def arctan2(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.arctan2(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"arctan2\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"arctan2\" is not implemented for backend \"JAX_CPU\".")

def sin(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.sin(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"sin\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"sin\" is not implemented for backend \"JAX_CPU\".")

def cos(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.cos(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"cos\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"cos\" is not implemented for backend \"JAX_CPU\".")

# ==========================================================================
# ========================== Matrix contractions ===========================
# ==========================================================================

def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_trace:
            logged_matrix_ops_trace[a.shape] += 1
        else: 
            logged_matrix_ops_trace[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"trace\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"trace\" is not implemented for backend \"JAX_CPU\".")

def kron(a, b):
    if log_matrix_ops:
        key = np.prod(a.shape)*np.prod(b.shape)
        if key in logged_matrix_ops_kron:
            logged_matrix_ops_kron[key] += 1
        else: 
            logged_matrix_ops_kron[key] = 1
    if backend == Backend.NUMPY:
        return np.kron(a, b)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"kron\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"kron\" is not implemented for backend \"JAX_CPU\".")

def tensordot(a, b, axes=2):
    if log_matrix_ops:
        key = np.prod(a.shape)
        for axis, value in enumerate(b.shape):
            if axis == axes or ((isinstance(axes, tuple) or isinstance(axes, list)) and axis in axes[0]):
                continue
            else:
                key *= value
        if key in logged_matrix_ops_tensordot:
            logged_matrix_ops_tensordot[key] += 1
        else: 
            logged_matrix_ops_tensordot[key] = 1
    if backend == Backend.NUMPY:
        return np.tensordot(a, b, axes=axes)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"tensordot\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"tensordot\" is not implemented for backend \"JAX_CPU\".")

def dot(a, b, out=None):
    if log_matrix_ops:
        key = np.prod(a.shape)*np.prod(b.shape[1:])
        if a.shape in logged_matrix_ops_dot:
            logged_matrix_ops_dot[a.shape] += 1
        else: 
            logged_matrix_ops_dot[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.dot(a, b, out=out)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"dot\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"dot\" is not implemented for backend \"JAX_CPU\".")

# ==========================================================================
# ========================== Matrix decompositions =========================
# ==========================================================================

def qr(a, mode='reduced'):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_qr:
            logged_matrix_ops_qr[a.shape] += 1
        else: 
            logged_matrix_ops_qr[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.linalg.qr(a, mode=mode)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"qr\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"qr\" is not implemented for backend \"JAX_CPU\".")

def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_svd:
            logged_matrix_ops_svd[a.shape] += 1
        else: 
            logged_matrix_ops_svd[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"svd\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"svd\" is not implemented for backend \"JAX_CPU\".")

def eigh(a, UPLO='L'):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_eigh:
            logged_matrix_ops_eigh[a.shape] += 1
        else: 
            logged_matrix_ops_eigh[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.linalg.eigh(a, UPLO=UPLO)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"min\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"min\" is not implemented for backend \"JAX_CPU\".")

# ==========================================================================
# ======================== Other relevant operations =======================
# ==========================================================================

def transpose(a, axes=None):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_transpose:
            logged_matrix_ops_transpose[a.shape] += 1
        else: 
            logged_matrix_ops_transpose[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.transpose(a, axes=axes)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"transpose\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"transpose\" is not implemented for backend \"JAX_CPU\".")

def reshape(a, shape=None, order='C', newshape=None, copy=None):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_reshape:
            logged_matrix_ops_reshape[a.shape] += 1
        else: 
            logged_matrix_ops_reshape[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.reshape(a, shape=shape, order=order, newshape=newshape, copy=copy)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"reshape\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"reshape\" is not implemented for backend \"JAX_CPU\".")