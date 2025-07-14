import numpy as np
from ...utility import utility
from ...utility.tripartite_decomposition import tripartite_decomposition as tripartite_decomposition_lib
from ...utility import debug_logging

"""
This file implements the application of a two-site time evolution operator as used in TEBD for honeycomb isoTPS.
We need to differentiate between the two different possible twosite environments in the honeycomb isoTPS.
The following functions thus all carry either the subscript 1 or 2.
"""

def _compute_overlap_except_W_prime_1(T1, T2, W, T1_prime, T2_prime, U):
    """
    Helper function that contracts the overlap of the two wavefunctions defined by T1, T2, W and
    T1', T2'. To fully compute the overlap, the resulting contraction must still be contracted
    with the complex conjugate of W'. The (time evolution) operator U is sandwiched between the
    wavefunctions.

    Parameters
    ----------
    T1, T2, W: np.ndarrays
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_1().
    T1_prime, T2_prime: np.ndarrays
        two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_1().
    U: np.ndarray of shape (i, j, i, j)
        two-site (time evolution) operator.

    Returns
    -------
    contr: np.ndarray of shape (l', (u', d'), r')
        result of the contraction.
    """
    contr = np.tensordot(T2, np.conj(T2_prime), ([1], [1])) # j [(ru rd)] l; j* [(ru rd)*] l* -> j l j* l*
    contr = np.tensordot(contr, U, ([0, 2], [3, 1])) # [j] l [j*] l*: i [j] i* [j*] -> l l* i i*
    contr = np.tensordot(contr, W, ([0], [2])) # [l] l* i i*; l (u d) [r] -> l* i i* l (u d)
    temp = np.tensordot(T1, np.conj(T1_prime), ([2], [2])) # i r [(ld lu)]; i* r* [(ld lu)*] -> i r i* r*
    contr = np.tensordot(contr, temp, ([1, 2, 3], [2, 0, 1])) # l* [i] [i*] [l] (u d); [i] [r] [i*] r* -> l* (u d) r* = l (u d) r
    contr = contr.transpose(2, 1, 0) # l, (u, d), r = r, (u, d), l -> l, (u, d), r
    return contr

def _compute_overlap_1(T1, T2, W, T1_prime, T2_prime, W_prime, U):
    """
    Helper function that contracts the overlap of the two wavefunctions defined by T1, T2, W and
    T1', T2', W'. The (time evolution) operator U is sandwiched between the
    wavefunctions. This is used in error computation.

    Parameters
    ----------
    T1, T2, W: np.ndarrays
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_1().
    T1_prime, T2_prime, W_prime: np.ndarrays
        two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_1().
    U: np.ndarray of shape (i, j, i, j)
        two-site (time evolution) operator.

    Returns
    -------
    contr: float
        result of the contraction.
    """
    contr = _compute_overlap_except_W_prime_1(T1, T2, W, T1_prime, T2_prime, U)
    return np.tensordot(contr, np.conj(W_prime), ([0, 1, 2], [0, 1, 2]))

def _compute_error_1(T1, T2, W, T1_prime, T2_prime, W_prime, U):
    """
    Helper function that computes the error |U@psi - psi_prime| = sqrt(2 - 2*Re(<psi_prime|U|psi>)),
    where we assumed <psi|psi> = <psi_prime|psi_prime> = 1.

    Parameters
    ----------
    T1, T2, W: np.ndarrays
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
    T1_prime, T2_prime, W_prime: np.ndarrays
        two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step().
    U: np.ndarray of shape (i, j, i, j)
        two-site (time evolution) operator.

    Returns
    -------
    error: float
        the computed error.
    """
    # temp should be 1, but compute it just to make sure.
    temp = _compute_overlap_1(T1, T2, W, T1, T2, W, np.tensordot(U, np.conj(U), ([2, 3], [2, 3]))) # i j [i*] [j*]; i* j* [i] [j] -> i j i* j*
    error = temp + 1 - 2*np.real(_compute_overlap_1(T1, T2, W, T1_prime, T2_prime, W_prime, U))
    error = np.sqrt(error)
    return error

def tebd_step_1(T1, T2, W, U, mode="svd", debug_logger=debug_logging.DebugLogger(), **kwargs):
    r"""
    Performs a single TEBD step by approximating the updated tensors of the two site wavefunction, after the time evolution operator
    U has been applied. the bond dimension of all tensors is kept the same.

        \  _|___________|_  /               \                   / 
         \ |______U______| /                 \  |           |  /  
          \ |           | /                   \ |           | /   
           T1-----W-----T2       \approx       T1'----W'----T2'   
          /      | |      \                   /      | |      \   
         /       | |       \                 /       | |       \  
        /                   \               /                   \ 

    Parameters
    ----------
    T1: np.ndarray of shape (i, r1, ld, lu)
        part of the two-site wavefunction
    T2: np.ndarray of shape (j, ru, rd, l1)
        part of the two-site wavefunction
    W: np.ndarray of shape (l, u, r, d) = (r1, u, l2, d)
        part of the two-site wavefunction
    U: np.ndarray of shape (i, j, i, j)
        two-site (time evolution) operator.
    mode: str, one of {"svd", "iterate_polar"}
        selecting the tebd_step implementation that is called. Depending on the implementation,
        additional parameters may be needed, that get passed on through kwargs. See individual
        tebd_step implementations for more detail.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.
    **kwargs:
        keyword arguments get passed on to the selected tebd_step implementation.

    Returns
    -------
    T1_prime : np.ndarray of shape (i, r1, ld, lu)
        part of the updated two-site wavefunction
    T2_prime : np.ndarray of shape (j, ru, rd, l1)
        part of the updated two-site wavefunction
    W_prime : np.ndarray of shape (l, u, r, d) = (r1, u, l2, d)
        part of the updated two-site wavefunction
    error : float
        error of applying the time evolution operator, or -float("inf") if debug_logger.log_local_tebd_update_errors and
        debug_logger.log_approximate_column_error_tebd are set to false.                                
    """
    # group legs s.t. we have to do less reshaping during the subroutine
    l, u, r, d = W.shape
    W = W.transpose(0, 1, 3, 2).reshape(l, u*d, r) # l, u, r, d -> l, u, d, r -> l, (u, d), r
    i, r1, ld1, lu1 = T1.shape
    T1 = T1.reshape(i, r1, ld1*lu1) # i, r, ld, lu -> i, r, (ld, lu)
    j, ru2, rd2, l2 = T2.shape
    T2 = T2.reshape(j, ru2*rd2, l2) # j ru, rd, l -> j, (ru, rd), l

    if mode == "svd":
        raise NotImplementedError()
    elif mode == "iterate_polar":
        T1_prime, T2_prime, W_prime = tebd_step_iterate_polar_1(T1, T2, W, U, **kwargs)
    else:
        raise NotImplementedError(f'tebd_step not implemented for mode {mode}')
    # Compute error
    error = -float("inf")
    if debug_logger.log_local_tebd_update_errors or debug_logger.log_approximate_column_error_tebd:
        error = _compute_error_1(T1, T2, W, T1_prime, T2_prime, W_prime, U)

    # ungroup legs again
    W_prime = W_prime.reshape(l, u, d, r).transpose(0, 1, 3, 2) # l, (u, d), r -> l, u, d, r -> l, u, r, d
    T1_prime = T1_prime.reshape(i, r1, ld1, lu1) # i, r, (ld, lu) -> i, r, ld, lu
    T2_prime = T2_prime.reshape(j, ru2, rd2, l2) # j, (ru, rd), l -> j ru, rd, l
    
    return T1_prime, T2_prime, W_prime, error

def tebd_step_iterate_polar_1(T1, T2, W, U, N_iters=100):
    """
    Computes the updated two-site wavefunction by iteratively optimizing the overlap
    with the original wavefunction with the time evolution operator applied, enforcing isometry
    conditions using polar decompositions. This is similar to the MPS TEBD implementation in
    Jakob Unfried, Johannes Hausschild, and Frank Pollmann: "Fast Time-Evolution of Matrix-Product States using the QR decomposition"
    (https://arxiv.org/abs/2212.09782).

    Parameters
    ----------
    T1, T2, W: np.ndarrays
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_1().
    U: np.ndarray of shape (i, j, i, j)
        time evolution operator
    N_iters: int, optional
        number of iterations. Default: 100.

    Returns
    -------
    T1_prime, T2_prime, W_prime : np.ndarrays
        the updated wave function. For shapes of individual tensors, see tebd_step_1()
    """

    def _update_T1(T1, T2, W, T2_prime, W_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out T1_prime.
        The updated T1_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, W: np.ndarrays
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_1().
        T2_prime, W_prime: np.ndarrays
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_1().
        U: np.ndarray of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        T1_prime: np.ndarray of shape (i, r, (ld, lu))
            updated T1_prime tensor.
        """
        # Flip arrow of W_prime
        l, ud, r = W_prime.shape
        W_prime = W_prime.reshape(l, ud*r) # l, (u, d), r -> l, ((u, d), r)
        W_prime, _ = np.linalg.qr(W_prime.T, mode="reduced") # l, ((u, d), r) -> ((u, d), r), l
        W_prime = W_prime.reshape(ud, r, l) # (u, d), r, l
        # Contract everything except T1_prime
        contr = np.tensordot(T2, np.conj(T2_prime), ([1], [1])) # j [(ru rd)] l; j* [(ru rd)*] l* -> j l j* l*
        contr = np.tensordot(contr, U, ([0, 2], [3, 1])) # [j] l [j*] l*: i [j] i* [j*] -> l l* i i*
        contr = np.tensordot(contr, W, ([0], [2])) # [l] l* i i*; l (u d) [r] -> l* i i* l (u d)
        contr = np.tensordot(np.conj(W_prime), contr, ([0, 1], [4, 0])) # [(u d)*] [r*] l*; [l*] i i* l [(u d)] -> l* i i* l
        contr = np.tensordot(contr, T1, ([2, 3], [0, 1])) # l* i [i*] [l]; [i] [r] (ld lu) -> l* i (ld lu) = r i (ld lu)
        # Isometrize
        r, i, ld_lu = contr.shape
        contr = contr.reshape(r, i*ld_lu) # r, i, (ld, lu) -> r, (i, (ld, lu))
        contr, _ = np.linalg.qr(contr.T, mode="reduced") # r, (i, (ld, lu)) -> (i, (ld, lu)), r
        contr = contr.reshape(i, ld_lu, r) # (i, (ld, lu)), r -> i, (ld, lu), r
        contr = contr.transpose(0, 2, 1) # i, (ld lu), r -> i, r, (ld, lu)
        return contr

    def _update_W(T1, T2, W, T1_prime, T2_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out W_prime.
        The updated W_prime is then computed from the contraction, normalized, and returned.

        Parameters
        ----------
        T1, T2, W: np.ndarrays
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_1().
        T1_prime, T2_prime: np.ndarrays
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_1().
        U: np.ndarray of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        W_prime: np.ndarray of shape (l, (u, d), r)
            updated W_prime tensor.
        """
        # Contract everything except W_prime
        contr = _compute_overlap_except_W_prime_1(T1, T2, W, T1_prime, T2_prime, U)
        # normalize
        return contr / np.linalg.norm(contr)

    def _update_T2(T1, T2, W, T1_prime, W_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out T2_prime.
        The updated T2_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, W: np.ndarrays
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_1().
        T1_prime, W_prime: np.ndarrays
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_1().
        U: np.ndarray of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        T2_prime: np.ndarray of shape (j, (ru, rd), l)
            updated T2_prime tensor.
        """
        # Flip arrow of W_prime
        l, ud, r = W_prime.shape
        W_prime = W_prime.reshape(l*ud, r) # l, (u, d), r -> (l, (u, d)), r
        W_prime, _ = np.linalg.qr(W_prime, mode="reduced")
        W_prime = W_prime.reshape(l, ud, r) # (l, (u, d)), r -> l, (u, d), r
        # Contract everything except T2_prime
        contr = np.tensordot(T1, np.conj(T1_prime), ([2], [2])) # i r [(ld lu)]; i* r* [(ld lu)*] -> i r i* r*
        contr = np.tensordot(contr, U, ([0, 2], [2, 0])) # [i] r [i*] r*; [i] j [i*] j* -> r r* j j*
        contr = np.tensordot(contr, W, ([0], [0])) # [r] r* j j*; [l] (u d) r -> r* j j* (u d) r
        contr = np.tensordot(contr, np.conj(W_prime), ([0, 3], [0, 1])) # [r*] j j* [(u d)] r; [l*] [(u d)*] r* -> j j* r r*
        contr = np.tensordot(contr, T2, ([1, 2], [0, 2])) # j [j*] [r] r*; [j] (ru rd) [l] -> j r* (ru rd) = j l (ru rd)
        # Isometrize
        j, l, ru_rd = contr.shape
        contr = contr.transpose(0, 2, 1) # j, l, (ru rd) -> j, (ru, rd), l
        contr = contr.reshape(j*ru_rd, l) # j, (ru, rd), l -> (j, (ru, rd)), l
        contr, _ = np.linalg.qr(contr, mode="reduced")
        contr = contr.reshape(j, ru_rd, l)
        return contr

    def _update_tensors(T1, T2, W, W_prime, T2_prime, U):
        """
        Helper function that performs one iteration of the TEBD step, sweeping over and updating
        all 3 tensors of the 2-site wavefunction.
        """
        T1_prime = _update_T1(T1, T2, W, T2_prime, W_prime, U)
        W_prime = _update_W(T1, T2, W, T1_prime, T2_prime, U)
        T2_prime = _update_T2(T1, T2, W, T1_prime, W_prime, U)
        W_prime = _update_W(T1, T2, W, T1_prime, T2_prime, U)
        return T1_prime, T2_prime, W_prime

    W_prime = W.copy()
    T2_prime = T2.copy()
    for _ in range(N_iters):
        T1_prime, T2_prime, W_prime = _update_tensors(T1, T2, W, W_prime, T2_prime, U)

    assert(T1_prime.shape == T1.shape)
    assert(T2_prime.shape == T2.shape)
    assert(W_prime.shape == W.shape)

    return T1_prime, T2_prime, W_prime

def _compute_overlap_except_W_prime_2(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U):
    """
    Helper function that contracts the overlap of the two wavefunctions defined by T1, T2, Wm1, W, Wp1 and
    T1', T2', Wm1', Wp1'. To fully compute the overlap, the resulting contraction must still be contracted
    with the complex conjugate of W'. The (time evolution) operator U is sandwiched between the
    wavefunctions.

    Parameters
    ----------
    T1, T2, Wm1, W, Wp1: np.ndarrays
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_2().
    T1_prime, T2_prime, Wm1_prime, Wp1_prime: np.ndarrays
        two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_2().
    U: np.ndarray of shape (i, j, i, j)
        two-site (time evolution) operator.

    Returns
    -------
    contr: np.ndarray of shape (l, u, r, d)
        result of the contraction.
    """
    # contract everything except W
    if Wm1 is None:
        if Wp1 is None:
            contr = np.tensordot(T2, np.conj(T2_prime), ([1, 3], [1, 3])) # p2 [r2] ld2 [lu2]; p2* [r2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
            contr = np.tensordot(contr, U, ([0, 2], [3, 1])) # [p2] ld2 [p2*] ld2*; i [j] i* [j*] -> ld2 ld2* i i*
            contr = np.tensordot(contr, W, ([0], [2])) # [ld2] ld2* i i*; l u [r] d -> ld2* i i* l u d
            temp = np.tensordot(T1, np.conj(T1_prime), ([2, 3], [2, 3])) # p1 ru1 [rd1] [l1]; p1* ru1* [rd1*] [l1*] -> p1 ru1 p1* ru1*
            contr = np.tensordot(temp, contr, ([0, 1, 2], [2, 3, 1])) # [p1] [ru1] [p1*] ru1*; ld2* [i] [i*] [l] u d -> ru1* ld2* u d = l r u d
            contr = np.transpose(contr, (0, 2, 1, 3)) # l, r, u, d -> l, u, r, d
        else:
            contr = np.tensordot(Wp1, np.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
            temp = np.tensordot(T2, np.conj(T2_prime), ([1], [1])) # p2 [r2] ld2 lu2; p2* [r2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
            temp = np.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i*
            contr = np.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i*
            contr = np.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d
            temp = np.tensordot(T1, np.conj(T1_prime), ([2, 3], [2, 3])) # p1 ru1 [rd1] [l1]; p1* ru1* [rd1*] [l1*] -> p1 ru1 p1* ru1*
            contr = np.tensordot(temp, contr, ([0, 1, 2], [3, 4, 2])) # [p1] [ru1] [p1*] ru1*; dp1* ld2* [i] [i*] [l] d -> ru1* dp1* ld2* d = l u r d
    else:
        if Wp1 is None:
            contr = np.tensordot(Wm1, np.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
            temp = np.tensordot(T1, np.conj(T1_prime), ([3], [3])) # p1 ru1 rd1 [l1]; p1* ru1* rd1* [l1*] -> p1 ru1 rd1 p1* ru1* rd1* 
            temp = np.tensordot(temp, U, ([0, 3], [2, 0])) # [p1] ru1 rd1 [p1*] ru1* rd1*; [i] j [i*] j* -> ru1 rd1 ru1* rd1* j j*
            contr = np.tensordot(contr, temp, ([0, 2], [1, 3])) # [lm1] um1 [lm1*] um1*; ru1 [rd1] ru1* [rd1*] j j* -> um1 um1* ru1 ru1* j j*
            contr = np.tensordot(contr, W, ([0, 2], [3, 0])) # [um1] um1* [ru1] ru1* j j*; [l] u r [d] -> um1* ru1* j j* u r
            temp = np.tensordot(T2, np.conj(T2_prime), ([1, 3], [1, 3])) # p2 [r2] ld2 [lu2]; p2* [r2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
            contr = np.tensordot(contr, temp, ([2, 3, 5], [2, 0, 1])) # um1* ru1* [j] [j*] u [r]; [p2] [ld2] [p2*] ld2* -> um1* ru1* u ld2* = d l u r
            contr = np.transpose(contr, (1, 2, 3, 0)) # d, l, u, r -> l, u, r, d
        else:
            contr = np.tensordot(Wp1, np.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
            temp = np.tensordot(T2, np.conj(T2_prime), ([1], [1])) # p2 [r2] ld2 lu2; p2* [r2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
            temp = np.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i*
            contr = np.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i*
            contr = np.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d
            temp = np.tensordot(T1, np.conj(T1_prime), ([3], [3])) # p1 ru1 rd1 [l1]; p1* ru1* rd1* [l1*] -> p1 ru1 rd1 p1* ru1* rd1* 
            temp2 = np.tensordot(Wm1, np.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
            temp = np.tensordot(temp, temp2, ([2, 5], [0, 2])) # p1 ru1 [rd1] p1* ru1* [rd1*]; [lm1] um1 [lm1*] um1* -> p1 ru1 p1* ru1* um1 um1*
            contr = np.tensordot(temp, contr, ([0, 1, 2, 4], [3, 4, 2, 5])) # [p1] [ru1] [p1*] ru1* [um1] um1*; dp1* ld2* [i] [i*] [l] [d] -> ru1* um1* dp1* ld2*
            contr = np.transpose(contr, (0, 2, 3, 1)) # ru1*, um1*, dp1*, ld2* = l, d, u, r -> l, u, r, d
    return contr

def _compute_overlap_2(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U):
    """
    Helper function that contracts the overlap of the two wavefunctions defined by T1, T2, Wm1, W, Wp1 and
    T1', T2', Wm1', W', Wp1'. The (time evolution) operator U is sandwiched between the
    wavefunctions. This is used in error computation.

    Parameters
    ----------
    T1, T2, Wm1, W, Wp1: np.ndarrays
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_2().
    T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime: np.ndarrays
        two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_2().
    U: np.ndarray of shape (i, j, i, j)
        two-site (time evolution) operator.

    Returns
    -------
    contr: float
        result of the contraction.
    """
    contr = _compute_overlap_except_W_prime_2(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U)
    return np.tensordot(contr, np.conj(W_prime), ([0, 1, 2, 3], [0, 1, 2, 3]))

def _compute_error_2(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U):
    """
    Helper function that computes the error |U@psi - psi_prime| = sqrt(2 - 2*Re(<psi_prime|U|psi>)),
    where we assumed <psi|psi> = <psi_prime|psi_prime> = 1.

    Parameters
    ----------
    T1, T2, Wm1, W, Wp1: np.ndarrays
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_2().
    T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime: np.ndarrays
        two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_2().
    U: np.ndarray of shape (i, j, i, j)
        two-site (time evolution) operator.

    Returns
    -------
    error: float
        the computed error.
    """
    # temp should be 1, but compute it just to make sure.
    temp = _compute_overlap_2(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, np.tensordot(U, np.conj(U), ([2, 3], [2, 3]))) # i j [i*] [j*]; i* j* [i] [j] -> i j i* j*
    error = temp + 1 - 2*np.real(_compute_overlap_2(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U))
    error = np.sqrt(error)
    return error

def tebd_step_iterate_polar_2(T1, T2, Wm1, W, Wp1, U, N_iters):
    """
    Computes the updated two-site wavefunction by iteratively optimizing the overlap
    with the original wavefunction with the time evolution operator applied, enforcing isometry
    conditions using polar decompositions. This is similar to the MPS TEBD implementation in
    Jakob Unfried, Johannes Hausschild, and Frank Pollmann: "Fast Time-Evolution of Matrix-Product States using the QR decomposition"
    (https://arxiv.org/abs/2212.09782).

    Parameters
    ----------
    T1, T2, W: np.ndarrays
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_2().
    U: np.ndarray of shape (i, j, i, j)
        time evolution operator
    N_iters: int, optional
        number of iterations. Default: 100.

    Returns
    -------
    T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime : np.ndarrays
        the updated wave function. For shapes of individual tensors, see tebd_step_2()
    """

    def _update_T1(T1, T2, Wm1, W, Wp1, T2_prime, Wm1_prime, W_prime, Wp1_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out T1_prime.
        The updated T1_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: np.ndarrays
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_2().
        T2_prime, Wm1_prime, W_prime, Wp1_prime: np.ndarrays
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_2().
        U: np.ndarray of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        T1_prime: np.ndarray of shape (i, ru1, rd1, l1)
            updated T1_prime tensor.
        """
        # Contract everything except T1
        if Wm1 is None:
            if Wp1 is None:
                contr = np.tensordot(T2, np.conj(T2_prime), ([1, 3], [1, 3])) # p2 [r2] ld2 [lu2]; p2* [r2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
                contr = np.tensordot(contr, U, ([0, 2], [3, 1])) # [p2] ld2 [p2*] ld2*; i [j] i* [j*] -> ld2 ld2* i i*
                contr = np.tensordot(contr, W, ([0], [2])) # [ld2] ld2* i i*; l u [r] d -> ld2* i i* l u d
                contr = np.tensordot(contr, np.conj(W_prime), ([0, 4, 5], [2, 1, 3])) # [ld2*] i i* l [u] [d]; l* [u*] [r*] [d*] -> i i* l l*
            else:
                contr = np.tensordot(Wp1, np.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
                temp = np.tensordot(T2, np.conj(T2_prime), ([1], [1])) # p2 [r2] ld2 lu2; p2* [r2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
                temp = np.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i*
                contr = np.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i*
                contr = np.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d
                contr = np.tensordot(contr, np.conj(W_prime), ([0, 1, 5], [1, 2, 3])) # [dp1*] [ld2*] i i* l [d]; l* [u*] [r*] [d*] -> i i* l l*
            contr = np.tensordot(contr, T1, ([1, 2], [0, 1])) # i [i*] [l] l*; [p1] [ru1] rd1 l1 -> i l* rd1 l1 = p1 ru1 rd1 l1
            T1_prime = np.transpose(contr, (0, 3, 2, 1)) # p1, ru1, rd1, l1 -> p1, l1, rd1, ru1
            p1, ru1, rd1, l1 = U.shape[2], W_prime.shape[0], T1.shape[2], T1.shape[3]
        else:
            if Wp1 is None:
                contr = np.tensordot(T2, np.conj(T2_prime), ([1, 3], [1, 3])) # p2 [r2] ld2 [lu2]; p2* [r2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
                contr = np.tensordot(contr, U, ([0, 2], [3, 1])) # [p2] ld2 [p2*] ld2*; i [j] i* [j*] -> ld2 ld2* i i*
                contr = np.tensordot(contr, W, ([0], [2])) # [ld2] ld2* i i*; l u [r] d -> ld2* i i* l u d
                contr = np.tensordot(contr, np.conj(W_prime), ([0, 4], [2, 1])) # [ld2*] i i* l [u] d; l* [u*] [r*] d* -> i i* l d l* d*
            else:
                contr = np.tensordot(Wp1, np.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
                temp = np.tensordot(T2, np.conj(T2_prime), ([1], [1])) # p2 [r2] ld2 lu2; p2* [r2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
                temp = np.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i*
                contr = np.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i*
                contr = np.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d
                contr = np.tensordot(contr, np.conj(W_prime), ([0, 1], [1, 2])) # [dp1*] [ld2*] i i* l d; l* [u*] [r*] d* -> i i* l d l* d*
            temp = np.tensordot(Wm1, np.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
            contr = np.tensordot(temp, contr, ([1, 3], [3, 5])) # lm1 [um1] lm1* [um1*]; i i* l [d] l* [d*] -> lm1 lm1* i i* l l*
            contr = np.tensordot(contr, T1, ([0, 3, 4], [2, 0, 1])) # [lm1] lm1* i [i*] [l] l*; [p1] [ru1] [rd1] l1 -> lm1* i l* l1
            T1_prime = np.transpose(contr, (1, 3, 0, 2)) # lm1*, i, l*, l1 = rd1, p1, ru1, l1 -> p1, l1, rd1, ru1
            p1, ru1, rd1, l1 = U.shape[2], W_prime.shape[0], Wm1_prime.shape[0], T1.shape[3]
        # isometrize T1
        if Wm1 is None:
            T1_prime = np.reshape(T1_prime, (p1*l1*rd1, ru1)) # p1, l1, rd1, ru1 -> (p1, l1, rd1), ru1
            T1_prime = utility.isometrize_polar(T1_prime)
            T1_prime = np.reshape(T1_prime, (p1, l1, rd1, ru1)) # (p1, l1, rd1), ru1 -> p1, l1, rd1, ru1
        else:
            T1_prime = np.reshape(T1_prime, (p1*l1, rd1*ru1)) # p1, l1, rd1, ru1 -> (p1, l1), (rd1, ru1)
            T1_prime = utility.isometrize_polar(T1_prime)
            T1_prime = np.reshape(T1_prime, (p1, l1, rd1, ru1)) # (p1, l1), (rd1, ru1) -> p1, l1, rd1, ru1
        T1_prime = np.transpose(T1_prime, (0, 3, 2, 1)) # p1, l1, rd1, ru1 -> p1, ru1, rd1, l1
        return T1_prime

    def _update_T2(T1, T2, Wm1, W, Wp1, T1_prime, Wm1_prime, W_prime, Wp1_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out T2_prime.
        The updated T2_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: np.ndarrays
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_2().
        T1_prime, Wm1_prime, W_prime, Wp1_prime: np.ndarrays
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_2().
        U: np.ndarray of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        T2_prime: np.ndarray of shape (j, r2, ld2, lu2)
            updated T2_prime tensor.
        """
        # Contract everything except T2
        if Wm1 is None:
            contr = np.tensordot(T1, np.conj(T1_prime), ([2, 3], [2, 3])) # p1 ru1 [l1] [lu1]; p1* ru1* [rd1*] [l1*] -> p1 ru1 p1* ru1*
            contr = np.tensordot(contr, U, ([0, 2], [2, 0])) # [p1] ru1 [p1*] ru1*; [i] j [i*] j* -> ru1 ru1* j j*
            contr = np.tensordot(contr, W, ([0], [0])) # [ru1] ru1* j j*; [l] u r d -> ru1* j j* u r d
            if Wp1 is None:
                contr = np.tensordot(contr, np.conj(W_prime), ([0, 3, 5], [0, 1, 3])) # [ru1*] j j* [u] r [d]; [l*] [u*] r* [d*] -> j j* r r*
                contr = np.tensordot(contr, T2, ([1, 2], [0, 2])) # j [j*] [r] r*; [p2] r2 [ld2] lu2 -> j r* r2 lu2 = p2 ld2 r2 lu2
                T2_prime = np.transpose(contr, (0, 2, 1, 3)) # p2, ld2, r2, lu2 -> p2, r2, ld2, lu2
                p2, r2, ld2, lu2 = U.shape[3], T2.shape[1], W_prime.shape[2], T2.shape[3]
            else:
                contr = np.tensordot(contr, np.conj(W_prime), ([0, 5], [0, 3])) # [ru1*] j j* u r [d]; [l*] u* r* [d*] -> j j* u r u* r*
                temp = np.tensordot(Wp1, np.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
                contr = np.tensordot(contr, temp, ([2, 4], [1, 3])) # j j* [u] r [u*] r*; rp1 [dp1] rp1* [dp1*] -> j j* r r* rp1 rp1*
                contr = np.tensordot(contr, T2, ([1, 2, 4], [0, 2, 3])) # j [j*] [r] r* [rp1] rp1*; [p2] r2 [ld2] [lu2] -> j r* rp1* r2
                T2_prime = np.transpose(contr, (0, 3, 1, 2)) # j, r*, rp1*, r2 = p2, ld2, lu2, r2 -> p2, r2, ld2, lu2
                p2, r2, ld2, lu2 = U.shape[3], T2.shape[1], W_prime.shape[2], Wp1_prime.shape[2]
        else:
            contr = np.tensordot(Wm1, np.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
            temp = np.tensordot(T1, np.conj(T1_prime), ([3], [3])) # p1 ru1 rd1 [l1]; p1* ru1* rd1* [l1*] -> p1 ru1 rd1 p1* ru1* rd1* 
            temp = np.tensordot(temp, U, ([0, 3], [2, 0])) # [p1] ru1 rd1 [p1*] ru1* rd1*; [i] j [i*] j* -> ru1 rd1 ru1* rd1* j j*
            contr = np.tensordot(contr, temp, ([0, 2], [1, 3])) # [lm1] um1 [lm1*] um1*; ru1 [rd1] ru1* [rd1*] j j* -> um1 um1* ru1 ru1* j j*
            contr = np.tensordot(contr, W, ([0, 2], [3, 0])) # [um1] um1* [ru1] ru1* j j*; [l] u r [d] -> um1* ru1* j j* u r
            if Wp1 is None:
                contr = np.tensordot(contr, np.conj(W_prime), ([0, 1, 4], [3, 0, 1])) # [um1*] [ru1*] j j* [u] r; [l*] [u*] r* [d*] -> j j* r r*
                contr = np.tensordot(contr, T2, ([1, 2], [0, 2])) # j [j*] [r] r*; [p2] r2 [ld2] lu2 -> j r* r2 lu2 = p2 ld2 r2 lu2
                T2_prime = np.transpose(contr, (0, 2, 1, 3)) # p2, ld2, r2, lu2 -> p2, r2, ld2, lu2
                p2, r2, ld2, lu2 = U.shape[3], T2.shape[1], W_prime.shape[2], T2.shape[3]
            else: 
                contr = np.tensordot(contr, np.conj(W_prime), ([0, 1], [3, 0])) # [um1*] [ru1*] j j* u r; [l*] u* r* [d*] -> j j* u r u* r*
                temp = np.tensordot(Wp1, np.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
                contr = np.tensordot(contr, temp, ([2, 4], [1, 3])) # j j* [u] r [u*] r*; rp1 [dp1] rp1* [dp1*] -> j j* r r* rp1 rp1*
                contr = np.tensordot(contr, T2, ([1, 2, 4], [0, 2, 3])) # j [j*] [r] r* [rp1] rp1*; [p2] r2 [ld2] [lu2] -> j r* rp1* r2
                T2_prime = np.transpose(contr, (0, 3, 1, 2)) # j, r*, rp1*, r2 = p2, ld2, lu2, r2 -> p2, r2, ld2, lu2
                p2, r2, ld2, lu2 = U.shape[3], T2.shape[1], W_prime.shape[2], Wp1_prime.shape[2]
        # Isometrize T2
        if Wp1 is None:
            T2_prime = np.transpose(T2_prime, (0, 1, 3, 2)) # p2, r2, ld2, lu2 -> p2, r2, lu2, ld2
            T2_prime = np.reshape(T2_prime, (p2*r2*lu2, ld2)) # p2, r2, lu2, ld2 -> (p2, r2, lu2), ld2
            T2_prime = utility.isometrize_polar(T2_prime)
            T2_prime = np.reshape(T2_prime, (p2, r2, lu2, ld2)) # (p2, r2, lu2), ld2 -> p2, r2, lu2, ld2
            T2_prime = np.transpose(T2_prime, (0, 1, 3, 2)) # p2, r2, lu2, ld2 -> p2, r2, ld2, lu2
        else:
            T2_prime = np.reshape(T2_prime, (p2*r2, ld2*lu2)) # p2, r2, ld2, lu2 -> (p2, r2), (ld2, lu2)
            T2_prime = utility.isometrize_polar(T2_prime)
            T2_prime = np.reshape(T2_prime, (p2, r2, ld2, lu2)) # (p2, r2), (ld2, lu2) -> p2, r2, ld2, lu2
        return T2_prime

    def _update_Wm1(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, W_prime, Wp1_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out Wm1_prime.
        The updated Wm1_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: np.ndarrays
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_2().
        T1_prime, T2_prime, W_prime, Wp1_prime: np.ndarrays
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_2().
        U: np.ndarray of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        Wm1_prime: np.ndarray of shape (lm1, um1, rm1, dm1)
            updated Wm1_prime tensor.
        """
        if Wm1 is None:
            return None
        # Contract everything except Wm1
        if Wp1 is None:
            contr = np.tensordot(T2, np.conj(T2_prime), ([1, 3], [1, 3])) # p2 [r2] ld2 [lu2]; p2* [r2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
            contr = np.tensordot(contr, U, ([0, 2], [3, 1])) # [p2] ld2 [p2*] ld2*; i [j] i* [j*] -> ld2 ld2* i i*
            contr = np.tensordot(contr, W, ([0], [2])) # [ld2] ld2* i i*; l u [r] d -> ld2* i i* l u d
            contr = np.tensordot(contr, np.conj(W_prime), ([0, 4], [2, 1])) # [ld2*] i i* l [u] d; l* [u*] [r*] d* -> i i* l d l* d*
        else:
            contr = np.tensordot(Wp1, np.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
            temp = np.tensordot(T2, np.conj(T2_prime), ([1], [1])) # p2 [r2] ld2 lu2; p2* [r2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
            temp = np.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i*
            contr = np.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i*
            contr = np.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d
            contr = np.tensordot(contr, np.conj(W_prime), ([0, 1], [1, 2])) # [dp1*] [ld2*] i i* l d; l* [u*] [r*] d* -> i i* l d l* d*
        temp = np.tensordot(T1, np.conj(T1_prime), ([3], [3])) # p1 ru1 rd1 [l1]; p1* ru1* rd1* [l1*] -> p1 ru1 rd1 p1* ru1* rd1*
        contr = np.tensordot(temp, contr, ([0, 1, 3, 4], [1, 2, 0, 4])) # [p1] [ru1] rd1 [p1*] [ru1*] rd1*; [i] [i*] [l] d [l*] d* -> rd1 rd1* d d*
        contr = np.tensordot(contr, Wm1, ([0, 2], [0, 1])) # [rd1] rd1* [d] d*; [lm1] [um1] rm1 dm1 -> rd1* d* rm1 dm1
        # Isometrize Wm1
        Wm1_prime = np.transpose(contr, (0, 2, 3, 1)) # rd1*, d*, rm1, dm1 = lm1, um1, rm1, dm1 -> lm1, rm1, dm1, um1
        lm1, um1, rm1, dm1 = T1_prime.shape[2], W_prime.shape[3], Wm1.shape[2], Wm1.shape[3]
        Wm1_prime = np.reshape(Wm1_prime, (lm1*rm1*dm1, um1)) # lm1, rm1, dm1, um1 -> (lm1, rm1, dm1), um1
        Wm1_prime = utility.isometrize_polar(Wm1_prime)
        Wm1_prime = np.reshape(Wm1_prime, (lm1, rm1, dm1, um1)) # (lm1, rm1, dm1), um1 -> lm1, rm1, dm1, um1
        Wm1_prime = np.transpose(Wm1_prime, (0, 3, 1, 2)) # lm1, rm1, dm1, um1 -> lm1, um1, rm1, dm1
        return Wm1_prime

    def _update_W(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out W_prime.
        The updated W_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: np.ndarrays
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_2().
        T1_prime, T2_prime, Wm1_prime, Wp1_prime: np.ndarrays
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_2().
        U: np.ndarray of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        W_prime: np.ndarray of shape (l, u, r, d)
            updated W_prime tensor.
        """
        # contract everything except W
        W_prime = _compute_overlap_except_W_prime_2(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U)
        # Update W
        W_prime /= np.linalg.norm(W_prime)
        return W_prime

    def _update_Wp1(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out Wp1_prime.
        The updated Wp1_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: np.ndarrays
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step_2().
        T1_prime, T2_prime, Wm1_prime, W_prime: np.ndarrays
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step_2().
        U: np.ndarray of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        Wp1_prime: np.ndarray of shape (lp1, up1, rp1, dp1)
            updated Wp1_prime tensor.
        """
        if Wp1 is None:
            return None
        # Contract everything except Wp1
        if Wm1 is None:
            contr = np.tensordot(T1, np.conj(T1_prime), ([2, 3], [2, 3])) # p1 ru1 [rd1] [l1]; p1* ru1* [rd1*] [l1*] -> p1 ru1 p1* ru1*
            contr = np.tensordot(contr, U, ([0, 2], [2, 0])) # [p1] ru1 [p1*] ru1*; [i] j [i*] j* -> ru1 ru1* j j*
            contr = np.tensordot(contr, W, ([0], [0])) # [ru1] ru1* j j*; [l] u r d -> ru1* j j* u r d
            contr = np.tensordot(contr, np.conj(W_prime), ([0, 5], [0, 3])) # [ru1*] j j* u r [d]; [l*] u* r* [d*] -> j j* u r u* r*
        else:
            contr = np.tensordot(Wm1, np.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
            temp = np.tensordot(T1, np.conj(T1_prime), ([3], [3])) # p1 ru1 rd1 [l1]; p1* ru1* rd1* [l1*] -> p1 ru1 rd1 p1* ru1* rd1* 
            temp = np.tensordot(temp, U, ([0, 3], [2, 0])) # [p1] ru1 rd1 [p1*] ru1* rd1*; [i] j [i*] j* -> ru1 rd1 ru1* rd1* j j*
            contr = np.tensordot(contr, temp, ([0, 2], [1, 3])) # [lm1] um1 [lm1*] um1*; ru1 [rd1] ru1* [rd1*] j j* -> um1 um1* ru1 ru1* j j*
            contr = np.tensordot(contr, W, ([0, 2], [3, 0])) # [um1] um1* [ru1] ru1* j j*; [l] u r [d] -> um1* ru1* j j* u r
            contr = np.tensordot(contr, np.conj(W_prime), ([0, 1], [3, 0])) # [um1*] [ru1*] j j* u r; [l*] u* r* [d*] -> j j* u r u* r*
        temp = np.tensordot(T2, np.conj(T2_prime), ([1], [1])) # p2 [r2] ld2 lu2; p2* [r2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
        contr = np.tensordot(contr, temp, ([0, 1, 3, 5], [3, 0, 1, 4])) # [j] [j*] u [r] u* [r*]; [p2] [ld2] lu2 [p2*] [ld2*] lu2* -> u u* lu2 lu2*
        contr = np.tensordot(contr, Wp1, ([0, 2], [3, 2])) # [u] u* [lu2] lu2*; lp1 up1 [rp1] [dp1] -> u* lu2* lp1 up1
        # Isometrize Wp1
        Wp1_prime = np.transpose(contr, (2, 3, 1, 0)) # u*, lu2*, lp1, up1 = dp1, rp1, lp1, up1 -> lp1, up1, rp1, dp1
        lp1, up1, rp1, dp1 = Wp1.shape[0], Wp1.shape[1], T2_prime.shape[3], W_prime.shape[1]
        Wp1_prime = np.reshape(Wp1_prime, (lp1*up1*rp1, dp1)) # lp1, up1, rp1, dp1 -> (lp1, up1, rp1), dp1
        Wp1_prime = utility.isometrize_polar(Wp1_prime)
        Wp1_prime = np.reshape(Wp1_prime, (lp1, up1, rp1, dp1)) # (lp1, up1, rp1), dp1 -> lp1, up1, rp1, dp1
        return Wp1_prime

    def _update_tensors(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U):
        """
        Helper function that performs one iteration of the TEBD step, sweeping over and updating
        all 5 tensors of the 2-site wavefunction.
        """
        T1_prime = _update_T1(T1, T2, Wm1, W, Wp1, T2_prime, Wm1_prime, W_prime, Wp1_prime, U)
        Wm1_prime = _update_Wm1(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, W_prime, Wp1_prime, U)
        W_prime = _update_W(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U)
        T2_prime = _update_T2(T1, T2, Wm1, W, Wp1, T1_prime, Wm1_prime, W_prime, Wp1_prime, U)
        Wp1_prime = _update_Wp1(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, U)
        W_prime = _update_W(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U)
        return T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime

    T1_prime = T1.copy()
    T2_prime = T2.copy()
    Wm1_prime = None
    if Wm1 is not None:
        Wm1_prime = Wm1.copy()
    W_prime = W.copy()
    Wp1_prime = None
    if Wp1 is not None:
        Wp1_prime = Wp1.copy()
    for _ in range(N_iters):
        T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime = _update_tensors(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U)

    assert(T1_prime.shape == T1.shape)
    assert(T2_prime.shape == T2.shape)
    if Wm1 is None:
        assert(Wm1_prime is None)
    else:
        assert(Wm1_prime.shape == Wm1.shape)
    assert(W_prime.shape == W.shape)
    if Wp1 is None:
        assert(Wp1_prime is None)
    else:
        assert(Wp1_prime.shape == Wp1.shape)

    return T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime

def tebd_step_2(T1, T2, Wm1, W, Wp1, U, chi_max, mode="svd", debug_logger=debug_logging.DebugLogger(), **kwargs):
    """
    Performs a single TEBD step by approximating the updated tensors of the two site wavefunction, after the time evolution operator
    U has been applied. the bond dimension of all tensors is kept the same.

    Parameters
    ----------
    T1 : np.ndarray of shape (i, ru1, rd1, l1)
        part of the two-site wavefunction
    T2 : np.ndarray of shape (j, r2, ld2, lu2)
        part of the two-site wavefunction
    Wm1 : np.ndarray of shape (lm1, um1, rm1, dm1) = (rd1, d, rm1, dm1) or None
        part of the two-site wavefunction
    W : np.ndarray of shape (l, u, r, d) = (ru1, u, ld2, d)
        part of the two-site wavefunction
    Wp1 : np.ndarray of shape (lp1, up1, rp1, dp1) = (lp1, up1, lu2, u) or None
        part of the two-site wavefunction
    U: np.ndarray of shape (i, j, i, j)
        two-site (time evolution) operator.
    mode: str, one of {"svd", "iterate_polar"}
        selecting the tebd_step implementation that is called. Depending on the implementation,
        additional parameters may be needed, that get passed on through kwargs. See individual
        tebd_step implementations for more detail.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.
    **kwargs:
        keyword arguments get passed on to the selected tebd_step implementation.

    Returns
    -------
    T1_prime : np.ndarray of shape (i, ru1, rd1, l1)
        part of the updated two-site wavefunction
    T2_prime : np.ndarray of shape (j, r2, ld2, lu2)
        part of the updated two-site wavefunction
    Wm1_prime : np.ndarray of shape (lm1, um1', rm1, dm1) = (rd1, d', rm1, dm1) or None
        part of the updated two-site wavefunction
    W_prime : np.ndarray of shape (l, u', r, d') = (ru1, u', ld2, d')
        part of the updated two-site wavefunction
    Wp1_prime : np.ndarray of shape (lp1, up1, rp1, dp1') = (lp1, up1, lu2, u') or None
        part of the updated two-site wavefunction
    error : float
        error of applying the time evolution operator, or -float("inf") if debug_logger.log_local_tebd_update_errors and
        debug_logger.log_approximate_column_error_tebd are set to false.                     
    """
    if mode == "svd":
        raise NotImplementedError()
    elif mode == "iterate_polar":
        T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime = tebd_step_iterate_polar_2(T1, T2, Wm1, W, Wp1, U, **kwargs)
    else:
        raise NotImplementedError(f'tebd_step not implemented for mode {mode}')
    # Compute error
    error = -float("inf")
    if debug_logger.log_local_tebd_update_errors or debug_logger.log_approximate_column_error_tebd:
        error = _compute_error_2(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U)

    return T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, error