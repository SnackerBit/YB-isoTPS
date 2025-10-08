from .. import backend
from .. import utility
from ..disentangle import disentangle
from .. import debug_logging

def tripartite_decomposition(T, D1, D2, chi, N_iters=100, initialize="polar", N_iters_pre_disentangler=200, N_iters_inner=5, debug_logger=debug_logging.DebugLogger()):
    """
    Performs the tripartite decomposition by iterating over two tensors while optimizing them and enforcing a 
    low-rank constraint on the second tensor through truncated SVD.

    Parameters
    ----------
    T : backend.array_type of shape (chi_1, chi_2, chi_3)
        the tensor that is to be decomposed.
    D1 : int
        bond dimension of the leg connecting the A and B tensor.
    D2 : int
        bond dimension of the leg connecting the A and C tensor.
    chi : int
        bond dimension of the leg connecting the B and C tensor.
    N_iters : int, optional
        number of iterations the algorithm is run for. Default: 100.
    initialize : str, one of {"polar", "identity", "qr", "random"}
        string selecting the method for initializing the disentangling unitary U. Default : "polar".
        See function `initialize_disentangle` in file `src/utility/disentangle/disentangle.py`.
    N_iters_pre_disentangler : int, optional 
        number of pre-disentangling iterations done before tha actual call to the disentangler. The pre-disentangler
        uses the fast power method to minimize the renyi-2 entropy. Default : 200.
        See function `initialize_disentangle` in file `src/utility/disentangle/disentangle.py`.
    N_iters_inner : int or None, optional
        number of iterations for approximating the SVD using the QR iteration power method.
        If this is set to None, a full SVD is used instead. Default: 5.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.

    Returns
    -------
    A : backend.array_type of shape (chi_1, D1, D2)
        resulting A tensor, isometry along (chi_1, (D1, D2))
    B : backend.array_type of shape (D1, chi_2, chi)
        resulting B tensor, isometry along ((D1, chi_2), chi)
    C : backend.array_type of shape (D2, chi, chi_3)
        resulting C tensor, normalized.
    """
    def _reduce_rank(X, k, C=None):
        """
        Helper function reducing the rank of matrix X to k using truncated SVD

        Parameters
        ----------
        X : backend.array_type of shape (n, m)
            matrix whose rank is to be reduced
        k : int
            truncation parameter
        C : backend.array_type of shape (k, m) or None, optional
            initialization from previous calls to _reduce_rank(),
            may lead to faster convergence. Default: None

        Returns
        -------
        X' : backend.array_type of shape (n, m)
            rank-reduced matrix
        C : backend.array_type of shape (k, m)
            can be used as initialization for the next call to _reduce_rank().
        """
        if N_iters_inner is None:
            U, V = utility.split_matrix_svd(X, k)
        else:
            U, V, _, _ = utility.split_matrix_iterate_QR(X, k, N_iters=N_iters_inner, C0=C)
        V /= backend.norm(V)
        return U@V, V
    # initialization
    chi_1, chi_2, chi_3 = T.shape
    T = T.reshape(chi_1, chi_2*chi_3)
    A, W = utility.split_matrix_svd(T, D1*D2)
    W = W.reshape(D1, D2, chi_2, chi_3)
    W = W.transpose(2, 0, 1, 3) # D1, D2, chi_2, chi_3 -> chi_2, D1, D2, chi_3
    if initialize != "identity" or N_iters_pre_disentangler > 0:
        _, W = disentangle.initialize_disentangle(W, init_U=initialize, N_iters_pre_disentangler=N_iters_pre_disentangler)
    W = W.reshape(chi_2*D1, D2*chi_3)
    # logging of debug information
    if debug_logger.tripartite_decomposition_log_iterates:
        iterates = []
    C0 = None
    # main loop
    for n in range(N_iters - 1):
        # Reduce rank of W
        W, C0 = _reduce_rank(W, chi, C0)
        # Reverse isometry arrows
        W = W.reshape(D1, chi_2*chi_3, D2)
        W = W.transpose(1, 0, 2) # D1, (chi_2, chi_3), D2 -> (chi_1, chi_3), D1, D2
        W = W.reshape(chi_2*chi_3, D1*D2)
        W, _ = backend.qr(W, mode="reduced")
        # Optimize A
        A = T@backend.conj(W)
        # Reverse isometry arrows
        A, _ = backend.qr(A, mode="reduced")
        # Optimize W
        W = backend.conj(A).T@T
        W = W.reshape(D1, D2, chi_2, chi_3)
        W = W.transpose(0, 2, 3, 1) # D1, D2, chi_2, chi_3 -> D1, chi_2, chi_3, D2
        W = W.reshape(D1*chi_2, chi_3*D2)
        # log debug information
        if debug_logger.tripartite_decomposition_log_iterates:
            if N_iters_inner is None:
                B_temp, C_temp = utility.split_matrix_svd(W, chi)
            else:
                B_temp, C_temp, _, _ = utility.split_matrix_iterate_QR(W, chi, N_iters=N_iters_inner, C0=C0)
            iterates.append({
                "A": A.reshape(chi_1, D1, D2), 
                "B": B_temp.reshape(D1, chi_2, chi), 
                "C": C_temp.reshape(chi, chi_3, D2).transpose(2, 0, 1) / backend.norm(C_temp)})
    # finalize
    if N_iters_inner is None:
        B, C = utility.split_matrix_svd(W, chi)
    else:
        B, C, _, _ = utility.split_matrix_iterate_QR(W, chi, N_iters=N_iters_inner, C0=C0)
    # debug logging
    if debug_logger.tripartite_decomposition_log_iterates:
        debug_logger.append_to_log_list(("tripartite_decomopsition_info", "iterates"), iterates)
    if debug_logger.tripartite_decomposition_log_info:
        debug_logger.append_to_log_list(("tripartite_decomopsition_info", "N_iters"), N_iters)
    # return results
    return A.reshape(chi_1, D1, D2), B.reshape(D1, chi_2, chi), C.reshape(chi, chi_3, D2).transpose(2, 0, 1) / backend.norm(C)