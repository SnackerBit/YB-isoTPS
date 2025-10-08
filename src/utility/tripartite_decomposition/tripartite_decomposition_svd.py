from .. import backend
from .. import utility
from ..disentangle import disentangle as disentangle_lib
from .. import debug_logging

def tripartite_decomposition(T, D1, D2, chi, N_iters_svd=None, eps_svd=0.0, disentangle=False, disentangle_options={}, debug_logger=debug_logging.DebugLogger()):
    """
    Performs the tripartite decomposition via two consecutive SVDs. Optionally a disentangling step is done before the second SVD.

    Parameters
    ----------
    T : backend.array_tpye of shape (chi_1, chi_2, chi_3)
        the tensor that is to be decomposed.
    D1 : int
        bond dimension of the leg connecting the A and B tensor.
    D2 : int
        bond dimension of the leg connecting the A and C tensor.
    chi : int
        bond dimension of the leg connecting the B and C tensor.
    N_iters_svd : int or None, optional
        Number of iterations for approximating the SVD, gets passed into split_matrix_iterate_QR().
        if this is set to None, a full SVD is used instead.
    eps_svd : float, optional
        eps parameter passed into split_matrix_iterate_QR().
    disentangle : bool, optional
        wether a disentangling routine should be executed before the second split. Default: False.
    disentangle_options : dictionary, optional
        options that get passed into the disentangler as kwargs.
        See src/utility/disentangle/disentangle.py for more information
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.

    Returns
    -------
    A : backend.array_tpye of shape (chi_1, D1, D2)
        resulting A tensor, isometry along (chi_1, (D1, D2))
    B : backend.array_tpye of shape (D1, chi_2, chi)
        resulting B tensor, isometry along ((D1, chi_2), chi)
    C : backend.array_tpye of shape (D2, chi, chi_3)
        resulting C tensor, normalized.
    """
    chi_1, chi_2, chi_3 = T.shape
    A = backend.reshape(T, (chi_1, chi_2*chi_3))
    if N_iters_svd is None:
        A, theta = utility.split_matrix_svd(A, D1*D2)
    else:
        A, theta, _, _ = utility.split_matrix_iterate_QR(A, D1*D2, N_iters_svd, eps_svd)
    A = backend.reshape(A, (chi_1, D1, D2))
    theta = backend.reshape(theta, (D1, D2, chi_2, chi_3)) # (D1, D2), (chi_2, chi_3) -> D1, D2, chi_2, chi_3
    # Check if the tripartite decomposition can be performed without error, in which case we don't need to disentangle
    if D1*chi_2 == chi:
        disentangle = False
    # Optionally disentangle
    if disentangle:
        theta = theta.transpose(2, 0, 1, 3) # D1, D2, chi_2, chi_3 -> chi_2, D_1, D_2, chi_3
        U = disentangle_lib.disentangle(theta, debug_logger=debug_logger, chi=chi, **disentangle_options)
        # Apply U to W
        theta = backend.tensordot(U, theta, ([2, 3], [1, 2])) # D1 D2 [D1*] [D2*]; chi_2 [D1] [D2] chi_3 -> D1 D2 chi_2 chi_3
        theta = theta.transpose((0, 2, 1, 3)) # D1, D2, chi_2, chi_3 -> D1, chi_2, D2, chi_3
        # Contract A with U^\dagger
        A = backend.tensordot(A, backend.conj(U), ([1, 2], [2, 3])) # chi_1 [D1] [D2]; D1* D2* [D1] [D2] -> chi_1 D1 D2
    else:
        theta = theta.transpose(0, 2, 1, 3) # D1, D2, chi_2, chi_3 -> D1, chi_2, D2, chi_3
    # Split the theta tensor into B and C
    theta = theta.reshape(D1*chi_2, D2*chi_3) # D1, chi_2, D2, chi_3 -> (D1, chi_2), (D2, chi_3)
    if N_iters_svd is None:
        B, C = utility.split_matrix_svd(theta, chi)
    else:
        B, C, _, _ = utility.split_matrix_iterate_QR(theta, chi, N_iters_svd, eps_svd)
    B = B.reshape(D1, chi_2, chi)
    C = C.reshape(chi, D2, chi_3)
    C = C.transpose(1, 0, 2) # chi, D2, chi_3 -> D2, chi, chi_3
    return A, B, C