import numpy as np
from .. import utility
from . import tripartite_decomposition_svd
from .. import debug_logging

def tripartite_decomposition(T, D1, D2, chi, N_iters=100, eps=1e-9, initialize="svd", A0=None, B0=None, debug_logger=debug_logging.DebugLogger()):
    """
    Performs the tripartite decomposition by iterating over the three tensors while optimizing them.
    The isometry condition is enforced using the polar decomposition.
    
    Parameters
    ----------
    T : np.ndarray of shape (chi_1, chi_2, chi_3)
        the tensor that is to be decomposed.
    D1 : int
        bond dimension of the leg connecting the A and B tensor.
    D2 : int
        bond dimension of the leg connecting the A and C tensor.
    chi : int
        bond dimension of the leg connecting the B and C tensor.
    N_iters : int, optional
        number of iterations the algorithm is run for. Default: 100.
    eps : float, optional
        if the change in truncation error after one iteartion is smaller than eps,
        the algorithm is terminated. Default: 1e-9.
    initialize : str, one of {"simple", "smart", "random", "svd"}
        String determining the method of initializing the split tensors. Default: "svd"
    A0 : np.ndarray of shape (chi_1, D1, D2) or None, optional
        Initialization of the A tensor. If this is not None, B0 must also be specified.
        Default: None.
    B0 : np.ndarray of shape (D1, chi_2, chi) or None, optional
        Initialization of the A tensor. If this is not None, B0 must also be specified.
        Default: None
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.
    
    Returns
    -------
    A : np.ndarray of shape (chi_1, D1, D2)
        resulting A tensor, isometry along (chi_1, (D1, D2))
    B : np.ndarray of shape (D1, chi_2, chi)
        resulting B tensor, isometry along ((D1, chi_2), chi)
    C : np.ndarray of shape (D2, chi, chi_3)
        resulting C tensor, normalized.
    """
    # Parse arguments
    if A0 is None:
        # Initialization
        chi_1, chi_2, chi_3 = T.shape
        B = np.reshape(np.eye(D1*chi_2, chi), (D1, chi_2, chi)) # {chi^2 D}
        if initialize == "simple":
            # Naive initialization method: Just slice the T tensor to the desired size
            C = T.copy()[:D2, :chi, :]
            if chi > chi_2:
                C = np.tensordot(C, np.eye(chi_2, chi), ([1], [0])) # D2 [chi] chi_3; [chi*] chi -> D2 chi_3 chi; {chi^3 D}
                C = np.transpose(C, (0, 2, 1)) # D2, chi_3, chi -> D2, chi, chi_3; {chi^2 D}
            C /= np.linalg.norm(C)
        elif initialize == "smart":
            # "smart" initialization: Order T wrt. absolute column/row size and slice.
            C = T.copy()
            if chi < chi_2:
                # find the chi largest columns
                temp = np.sum(np.abs(T), (0, 2))
                piv = np.argsort(temp)[::-1][:chi]
                C = C[:, piv, :]
            if D2 < chi_1:
                # find the D2 largest rows
                temp = np.sum(np.abs(T), (1, 2))
                piv = np.argsort(temp)[::-1][:D2]
                C = C[piv, :, :]
            if chi > chi_2:
                C = np.tensordot(C, np.eye(chi_2, chi), ([1], [0])) # D2 [chi] chi_3; [chi*] chi -> D2 chi_3 chi; {chi^3 D}
                C = np.transpose(C, (0, 2, 1)) # D2, chi_3, chi -> D2, chi, chi_3; {chi^2 D}
            C /= np.linalg.norm(C) # {chi^2 D}
        elif initialize == "random":
            # Initialize the C tensor randomly
            C = np.random.random((D2, chi, chi_3)) # {chi^2 D}
            C /= np.linalg.norm(C) # {chi^2 D}
        elif initialize == "svd":
            # initialize tensors with an svd
            _, B, C = tripartite_decomposition_svd.tripartite_decomposition(T, D1, D2, chi, disentangle=True, disentangle_options={"mode": "none", "init_U": "polar", "N_iters_pre_disentangler": 0})
            C /= np.linalg.norm(C)
        else:
            raise NotImplementedError(f"initialization mode \"{initialize}\" is not implemented for the tripartite decomposition with mode \"iterate\"!")
    else:
        A = A0
        B = B0
        C = np.tensordot(temp, np.conj(B), ([0, 2], [1, 0])) # [chi_2] chi_3 [D1*] D2*; [D1*] [chi_2*] chi* -> chi_3 D2* chi*
        C = np.transpose(C, (1, 2, 0)) # chi_3, D2*, chi* = chi_3, D2, chi -> D2, chi, chi_3
        C /= np.linalg.norm(C)
    # debug information
    if debug_logger.tripartite_decomposition_log_environment_per_iteration:
        iterates = []
    if debug_logger.tripartite_decomposition_log_info_per_iteration:
        costs = []
    # Main loop
    f = None
    for n in range(N_iters):
        # Optimize A
        A = np.tensordot(T, np.conj(B), ([1], [1])) # chi_1 [chi_2] chi_3; D1* [chi_2*] chi* -> chi_1 chi_3 D1* chi*
        A = np.tensordot(A, np.conj(C), ([1, 3], [2, 1])) # chi_1 [chi_3] D1* [chi*]; D2* [chi*] [chi_3*] -> chi_1 D1* D2*
        A = np.reshape(A, (chi_1, D1*D2))
        A = np.reshape(utility.isometrize_polar(A), (chi_1, D1, D2))
        # Optimize B
        temp = np.tensordot(T, np.conj(A), ([0], [0])) # [chi_1] chi_2 chi_3; [chi_1*] D1* D2* -> chi_2 chi_3 D1* D2*
        B = np.tensordot(temp, np.conj(C), ([1, 3], [2, 0])) # chi_2 [chi_3] D1* [D2*]; [D2*] chi* [chi_3*] -> chi_2 D1* chi*
        B = np.transpose(B, (1, 0, 2)) # chi_2, D1, chi -> D1, chi_2, chi
        B = np.reshape(B, (D1*chi_2, chi))
        B = np.reshape(utility.isometrize_polar(B), (D1, chi_2, chi))
        # Optimize C
        C = np.tensordot(temp, np.conj(B), ([0, 2], [1, 0])) # [chi_2] chi_3 [D1*] D2*; [D1*] [chi_2*] chi* -> chi_3 D2* chi*
        C = np.transpose(C, (1, 2, 0)) # chi_3, D2*, chi* = chi_3, D2, chi -> D2, chi, chi_3
        # We can easily compute the current truncation error from the norm of the not normalized C tensor
        norm = np.linalg.norm(C) 
        f_new = np.sqrt(max(2.0 - 2.0*norm, 0.0))
        C /= norm
        # log debug information
        if debug_logger.tripartite_decomposition_log_iterates:
            iterates.append({"A": A, "B": B, "C": C})
        if debug_logger.tripartite_decomposition_log_info_per_iteration:
            costs.append(f_new)
        # Check if algorithm should terminate
        if f is not None and np.abs(f - f_new) < eps:
            if debug_logger.tripartite_decomposition_log_iterates:
                debug_logger.append_to_log_list(("tripartite_decomopsition_info", "iterates"), iterates)
            if debug_logger.tripartite_decomposition_log_info_per_iteration:
                debug_logger.append_to_log_list(("tripartite_decomopsition_info", "costs"), costs)
            if debug_logger.tripartite_decomposition_log_info:
                debug_logger.append_to_log_list(("tripartite_decomopsition_info", "N_iters"), n+1)
            return A, B, C
        f = f_new
    if debug_logger.tripartite_decomposition_log_iterates:
        debug_logger.append_to_log_list(("tripartite_decomopsition_info", "iterates"), iterates)
    if debug_logger.tripartite_decomposition_log_info_per_iteration:
        debug_logger.append_to_log_list(("tripartite_decomopsition_info", "costs"), costs)
    if debug_logger.tripartite_decomposition_log_info:
        debug_logger.append_to_log_list(("tripartite_decomopsition_info", "N_iters"), N_iters)
    return A, B, C