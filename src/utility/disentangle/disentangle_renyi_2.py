from .. import backend
from .. import utility
from .. import debug_logging

def disentangle(theta, eps=1e-10, N_iters=200, min_iters=0, debug_logger=debug_logging.DebugLogger()):
    """
    Disentangles the given wavefunction theta by minimizing the renyi-2-entropy, using the fast power iteration method. This disentangler was introduced in
    [1] J. Hauschild, E. Leviatan, J. H. Bardarson, E. Altman, M. P. Zaletel, and F. Pollmann: "Finding purifications with minimal entanglement", https://arxiv.org/abs/1711.01288 
    
    Parameters
    ----------
    theta : backend.array_type of shape (l, i, j, r)
        Wavefunction tensor to be disentangled.
    eps : float, optional
        After the difference in renyi entropy of two consecutive iterations is smaller than this threshhold value,
        the algorithm terminates. Default: 1e-15.
    N_iters : int, optional
        Maximum number of iterations the algorith is run for. Default: 200.
    min_iters : int, optional
        Minimum number of iterations, before the eps condition leads to a termination of the algorithm.
        Mostly for debugging purposes. Default: 0.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.

    Returns
    -------
    U_final : backend.array_type of shape (i, j, i*, j*)
        final disentangling unitary after optimization
    """
    # Helper function
    def _U2(theta):
        """
        Helper function used in the disentangling algorithm.

        Parameters
        ----------
        theta : backend.array_type of shape (l, d1, d2, r)
            current wavefunction tensor
        
        Returns
        -------
        s : float
            renyi 2 entropy of the wavefunction
        u : backend.array_type of shape (d1*d2, d1*d2)
            update for disentangling unitary
        """
        chi = theta.shape
        rhoL = backend.tensordot(theta, backend.conj(theta), axes = [[2, 3], [2, 3]]) # ml d1 [d2] [mr]; ml* d1* [d2*] [mr*] -> ml d1 ml* d1* { D^9 }

        dS = backend.tensordot(rhoL, theta, axes = [[2, 3], [0, 1] ]) # ml d1 [ml*] [d1*]; [ml] [d1] d2 mr -> ml d1 d2 mr { D^9 }
        dS = backend.tensordot( backend.conj(theta), dS, axes = [[0, 3], [0, 3]]) # [ml] d1 d2 [mr]; [ml*] d1* d2* [mr*] { D^8 }

        dS = dS.reshape((chi[1]*chi[2], -1))
        s2 = backend.trace( dS )
        
        X, Y, Z = utility.safe_svd(dS)
        return -backend.log(s2), (backend.dot(X, Z).T).conj()
    # Initialize
    _, d1, d2, _ = theta.shape
    U = backend.eye(d1*d2, dtype = theta.dtype) # { D^4 }
    # debug info
    if debug_logger.disentangling_log_iterates:
        iterates = []
    # Main loop
    m = 0
    go = True
    Ss = []
    while m < N_iters and (go or m < min_iters):
        s, u = _U2(theta) 
        U = backend.dot(u, U)
        if debug_logger.disentangling_log_iterates:
            iterates.append(U.reshape((d1,d2,d1,d2)))
        u = u.reshape((d1,d2,d1,d2))
        theta = backend.tensordot(u.conj(), theta, axes = [[2, 3], [1, 2]]).transpose([2, 0, 1, 3])
        Ss.append(s)
        if m > 1:
            go = Ss[-2] - Ss[-1] > eps 
        m+=1
    # Save debug information
    if debug_logger.disentangling_log_iterates:
        debug_logger.append_to_log_list(("disentangler_info", "iterates"), iterates)
    if debug_logger.disentangling_log_info:
        debug_logger.append_to_log_list(("disentangler_info", "N_iters"), m)
    # Return result
    return backend.reshape(backend.conj(U), (d1, d2, d1, d2))