from ...utility import utility
from ...utility import backend
from ...utility.tripartite_decomposition import tripartite_decomposition as tripartite_decomposition_lib
from ...utility import debug_logging

"""
This file implements the application of a two-site time evolution operator as used in TEBD for square isoTPS.
"""

def _contract_full(T1, T2, Wm1, W, Wp1, U=None):
        """
        Helper function that fully contracts the two site wave function, optionally applying the time evolution operator U.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: backend.array_types
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
        U: backend.array_type of shape (i, j, i, j) or None, optional
            two-site (time evolution) operator. If set to None, an identity is applied instead. Default: None.

        Returns
        -------
        psi: backend.array_type of shape (ld1, lu1, ru2, rd2, rm1, dm1, lp1, up1, i, j)
            fully contracted and normalized wave function
        """
        psi = backend.tensordot(T1, W, ([1], [0])) # i [ru1] rd1 ld1 lu1; [l] u r d; i rd1 ld1 lu1 u r d
        psi = backend.tensordot(psi, T2, ([5], [3])) # i rd1 ld1 lu1 u [r] d; j ru2 rd2 [ld2] lu2 -> i rd1 ld1 lu1 u d j ru2 rd2 lu2
        if Wm1 is not None:
            # i [rd1] ld1 lu1 u [d] j ru2 rd2 lu2; [lm1] [um1] rm1 dm1 -> i ld1 lu1 u j ru2 rd2 lu2 rm1 dm1
            psi = backend.tensordot(psi, Wm1, ([1, 5], [0, 1]))
        else:
            # i rd1 ld1 lu1 u d j ru2 rd2 lu2 -> i ld1 lu1 u j ru2 rd2 lu2 rd1 d = i ld1 lu1 u j ru2 rd2 lu2 rm1 dm1
            psi = backend.transpose(psi, (0, 2, 3, 4, 6, 7, 8, 9, 1, 5))
        if Wp1 is not None:
            # i ld1 lu1 [u] j ru2 rd2 [lu2] rm1 dm1; lp1 up1 [rp1] [dp1] -> i ld1 lu1 j ru2 rd2 rm1 dm1 lp1 up1
            psi = backend.tensordot(psi, Wp1, ([3, 7], [3, 2]))
        else:
            # i ld1 lu1 u j ru2 rd2 lu2 rm1 dm1 -> i ld1 lu1 j ru2 rd2 rm1 dm1 lu2 u = i ld1 lu1 j ru2 rd2 rm1 dm1 lp1 up1
            psi = backend.transpose(psi, (0, 1, 2, 4, 5, 6, 8, 9, 7, 3))
        if U is None:
            psi = backend.transpose(psi, (1, 2, 4, 5, 6, 7, 8, 9, 0, 3)) # i, ld1, lu1, j, ru2, rd2, rm1, dm1, lp1, up1 -> ld1, lu1, ru2, rd2, rm2, dm1, lp1, up1, i, j
        else:
            psi = backend.tensordot(psi, U, ([0, 3], [2, 3])) # [i] ld1 lu1 [j] ru2 rd2 rm1 dm1 lp1 up1; i j [i*] [j*] -> ld1 lu1 ru2 rd2 rm1 dm1 lp1 up1 i j
        return psi / backend.norm(psi)

def _compute_overlap_except_W_prime(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U):
    """
    Helper function that contracts the overlap of the two wavefunctions defined by T1, T2, Wm1, W, Wp1 and
    T1', T2', Wm1', Wp1'. To fully compute the overlap, the resulting contraction must still be contracted
    with the complex conjugate of W'. The (time evolution) operator U is sandwiched between the wavefunctions.
    Complexity scaling: O( chi^3 D^3 d^2 + D^6 d^2 )

    Parameters
    ----------
    T1, T2, Wm1, W, Wp1: backend.array_types
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
    T1_prime, T2_prime, Wm1_prime, Wp1_prime: backend.array_types
        two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step().
    U: backend.array_type of shape (i, j, i, j)
        two-site (time evolution) operator.

    Returns
    -------
    contr: backend.array_type of shape (l', u', r', d')
        result of the contraction.
    """
    # contract everything except W
    if Wm1 is None:
        if Wp1 is None:
            contr = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2, 4], [1, 2, 4])) # p2 [ru2] [rd2] ld2 [lu2]; p2* [ru2*] [rd2*] ld2* [lu2*] -> p2 ld2 p2* ld2* { D^5 d^2 }
            contr = backend.tensordot(contr, U, ([0, 2], [3, 1])) # [p2] ld2 [p2*] ld2*; i [j] i* [j*] -> ld2 ld2* i i* { D^2 d^4 }
            contr = backend.tensordot(contr, W, ([0], [2])) # [ld2] ld2* i i*; l u [r] d -> ld2* i i* l u d { chi^2 D^3 d^2 }
            temp = backend.tensordot(T1, backend.conj(T1_prime), ([2, 3, 4], [2, 3, 4])) # p1 ru1 [rd1] [ld1] [lu1]; p1* ru1* [rd1*] [ld1*] [lu1*] -> p1 ru1 p1* ru1* { D^5 d^2 }
            contr = backend.tensordot(temp, contr, ([0, 1, 2], [2, 3, 1])) # [p1] [ru1] [p1*] ru1*; ld2* [i] [i*] [l] u d -> ru1* ld2* u d = l r u d { chi^2 D^3 d^2 }
            contr = backend.transpose(contr, (0, 2, 1, 3)) # l, r, u, d -> l, u, r, d { chi^2 D^2 }
        else:
            contr = backend.tensordot(Wp1, backend.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1* { chi^3 D^3 }
            temp = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2], [1, 2])) # p2 [ru2] [rd2] ld2 lu2; p2* [ru2*] [rd2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2* { D^6 d^2 }
            temp = backend.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i* { d^4 D^4 }
            contr = backend.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i* { chi^2 D^4 d^2 }
            contr = backend.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d { chi^3 D^3 d^2 }
            temp = backend.tensordot(T1, backend.conj(T1_prime), ([2, 3, 4], [2, 3, 4])) # p1 ru1 [rd1] [ld1] [lu1]; p1* ru1* [rd1*] [ld1*] [lu1*] -> p1 ru1 p1* ru1* { D^5 d^2 }
            contr = backend.tensordot(temp, contr, ([0, 1, 2], [3, 4, 2])) # [p1] [ru1] [p1*] ru1*; dp1* ld2* [i] [i*] [l] d -> ru1* dp1* ld2* d = l u r d { chi^2 D^3 d^2 }
    else:
        if Wp1 is None:
            contr = backend.tensordot(Wm1, backend.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1* { chi^3 D^3 }
            temp = backend.tensordot(T1, backend.conj(T1_prime), ([3, 4], [3, 4])) # p1 ru1 rd1 [ld1] [lu1]; p1* ru1* rd1* [ld1*] [lu1*] -> p1 ru1 rd1 p1* ru1* rd1* { D^6 d^2 }
            temp = backend.tensordot(temp, U, ([0, 3], [2, 0])) # [p1] ru1 rd1 [p1*] ru1* rd1*; [i] j [i*] j* -> ru1 rd1 ru1* rd1* j j* { D^4 d^4 }
            contr = backend.tensordot(contr, temp, ([0, 2], [1, 3])) # [lm1] um1 [lm1*] um1*; ru1 [rd1] ru1* [rd1*] j j* -> um1 um1* ru1 ru1* j j* { chi^2 D^4 d^2 }
            contr = backend.tensordot(contr, W, ([0, 2], [3, 0])) # [um1] um1* [ru1] ru1* j j*; [l] u r [d] -> um1* ru1* j j* u r { chi^3 D^3 d^2 }
            temp = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2, 4], [1, 2, 4])) # p2 [ru2] [rd2] ld2 [lu2]; p2* [ru2*] [rd2*] ld2* [lu2*] -> p2 ld2 p2* ld2* { D^5 d^2 }
            contr = backend.tensordot(contr, temp, ([2, 3, 5], [2, 0, 1])) # um1* ru1* [j] [j*] u [r]; [p2] [ld2] [p2*] ld2* -> um1* ru1* u ld2* = d l u r { chi^2 D^3 d^2 }
            contr = backend.transpose(contr, (1, 2, 3, 0)) # d, l, u, r -> l, u, r, d { chi^2 D^2 }
        else:
            contr = backend.tensordot(Wp1, backend.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1* { chi^3 D^3 }
            temp = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2], [1, 2])) # p2 [ru2] [rd2] ld2 lu2; p2* [ru2*] [rd2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2* { D^6 d^2 }
            temp = backend.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i* { d^4 D^4 }
            contr = backend.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i* { chi^2 D^4 d^2 }
            contr = backend.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d { chi^3 D^3 d^2 }
            temp = backend.tensordot(T1, backend.conj(T1_prime), ([3, 4], [3, 4])) # p1 ru1 rd1 [ld1] [lu1]; p1* ru1* rd1* [ld1*] [lu1*] -> p1 ru1 rd1 p1* ru1* rd1* { D^6 d^2 }
            temp2 = backend.tensordot(Wm1, backend.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1* { chi^3 D^3 }
            temp = backend.tensordot(temp, temp2, ([2, 5], [0, 2])) # p1 ru1 [rd1] p1* ru1* [rd1*]; [lm1] um1 [lm1*] um1* -> p1 ru1 p1* ru1* um1 um1* { chi^2 D^4 d^2 }
            contr = backend.tensordot(temp, contr, ([0, 1, 2, 4], [3, 4, 2, 5])) # [p1] [ru1] [p1*] ru1* [um1] um1*; dp1* ld2* [i] [i*] [l] [d] -> ru1* um1* dp1* ld2* { chi^3 D^3 d^2 }
            contr = backend.transpose(contr, (0, 2, 3, 1)) # ru1*, um1*, dp1*, ld2* = l, d, u, r -> l, u, r, d { chi^2 D^2 }
    return contr

def _compute_overlap(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U):
    """
    Helper function that contracts the overlap of the two wavefunctions defined by T1, T2, Wm1, W, Wp1 and
    T1', T2', Wm1', W', Wp1'. The (time evolution) operator U is sandwiched between the
    wavefunctions. This is used in error computation.
    Complexity scaling: O( chi^3 D^3 d^2 + D^6 d^2 )

    Parameters
    ----------
    T1, T2, Wm1, W, Wp1: backend.array_types
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
    T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime: backend.array_types
        two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step().
    U: backend.array_type of shape (i, j, i, j)
        two-site (time evolution) operator.

    Returns
    -------
    contr: float
        result of the contraction.
    """
    contr = _compute_overlap_except_W_prime(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U)
    return backend.tensordot(contr, backend.conj(W_prime), ([0, 1, 2, 3], [0, 1, 2, 3]))

def _compute_error(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U):
    """
    Helper function that computes the error |U@psi - psi_prime| = sqrt(2 - 2*Re(<psi_prime|U|psi>)),
    where we assumed <psi|psi> = <psi_prime|psi_prime> = 1.
    Complexity scaling: O( chi^3 D^3 d^2 + D^6 d^2 )

    Parameters
    ----------
    T1, T2, Wm1, W, Wp1: backend.array_types
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
    T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime: backend.array_types
        two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step().
    U: backend.array_type of shape (i, j, i, j)
        two-site (time evolution) operator.

    Returns
    -------
    error: float
        the computed error.
    """
    # temp should be 1, but compute it just to make sure.
    temp = _compute_overlap(T1, T2, Wm1, W, Wp1, T1, T2, Wm1, W, Wp1, backend.tensordot(U, backend.conj(U), ([2, 3], [2, 3]))) # i j [i*] [j*]; i* j* [i] [j] -> i j i* j*
    error = 1 + temp - 2*backend.real(_compute_overlap(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U))
    return backend.real_if_close(backend.sqrt(error))

def tebd_step(T1, T2, Wm1, W, Wp1, U, chi_max, mode="svd", debug_logger=debug_logging.DebugLogger(), **kwargs):
    """
    Performs a single TEBD step by approximating the updated tensors of the two site wavefunction, after the time evolution operator
    U has been applied. the bond dimension of T tensors is kept the same. The bond dimension between W tensors is truncated
    to chi_max. 

    Parameters
    ----------
    T1 : backend.array_type of shape (i, ru1, rd1, ld1, lu1)
        part of the two-site wavefunction
    T2 : backend.array_type of shape (j, ru2, rd2, ld2, lu2)
        part of the two-site wavefunction
    Wm1 : backend.array_type of shape (lm1, um1, rm1, dm1) = (rd1, d, rm1, dm1) or None
        part of the two-site wavefunction
    W : backend.array_type of shape (l, u, r, d) = (ru1, u, ld2, d)
        part of the two-site wavefunction
    Wp1 : backend.array_type of shape (lp1, up1, rp1, dp1) = (lp1, up1, lu2, u) or None
        part of the two-site wavefunction
    U : backend.array_type of shape (i, j, i, j)
        two-site (time evolution) operator.
    chi_max : int
        maximal bond dimension along the orthogonality surface.
    mode : str, one of {"svd", "iterate_polar"}
        selecting the tebd_step implementation that is called. Depending on the implementation,
        additional parameters may be needed, that get passed on through kwargs. See individual
        tebd_step implementations for more detail.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.
    **kwargs :
        keyword arguments get passed on to the selected tebd_step implementation.

    Returns
    -------
    T1_prime : backend.array_type of shape (i, ru1, rd1, ld1, lu1)
        part of the updated two-site wavefunction
    T2_prime : backend.array_type of shape (j, ru2, rd2, ld2, lu2)
        part of the updated two-site wavefunction
    Wm1_prime : backend.array_type of shape (lm1, um1', rm1, dm1) = (rd1, d', rm1, dm1) or None
        part of the updated two-site wavefunction
    W_prime : backend.array_type of shape (l, u', r, d') = (ru1, u', ld2, d')
        part of the updated two-site wavefunction
    Wp1_prime : backend.array_type of shape (lp1, up1, rp1, dp1') = (lp1, up1, lu2, u') or None
        part of the updated two-site wavefunction
    error : float
        error of applying the time evolution operator, or -float("inf") if debug_logger.log_local_tebd_update_errors and
        debug_logger.log_approximate_column_error_tebd are set to false.
    """
    if mode == "svd":
        T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime = tebd_step_svd(T1, T2, Wm1, W, Wp1, U, chi_max, **kwargs)
    elif mode == "iterate_polar":
        T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime = tebd_step_iterate_polar(T1, T2, Wm1, W, Wp1, U, debug_logger=debug_logger, **kwargs)
    else:
        raise NotImplementedError(f'tebd_step not implemented for mode {mode}')
    if debug_logger.log_local_tebd_update_errors or debug_logger.log_approximate_column_error_tebd:
        error = _compute_error(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U)
        return T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, error

    return T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, -float("inf")

def tebd_step_svd(T1, T2, Wm1, W, Wp1, U, chi_max, **kwargs):
    """
    Performs the TEBD step by contracting everything together and reconstructing the wavefunction tensors 
    using two tripartite decompositions.

    Parameters
    ----------
    T1, T2, Wm1, W, Wp1: backend.array_types
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
    U: backend.array_type of shape (i, j, i, j)
        two-site (time evolution) operator.
    chi_max: int
        maximal bond dimension along the orthogonality surface.
    **kwargs:
        keyword arguments get passed to tripartite decomposition subruotine.
        See "src/utility/tripartite_decomposition/tripartite_decomposition.py" for more information.

    Returns
    -------
    T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime : backend.array_types
        the updated wave function. For shapes of individual tensors, see tebd_step()
    """
    i, ru1, rd1, ld1, lu1 = T1.shape
    j, ru2, rd2, ld2, lu2 = T2.shape
    l, u, r, d = W.shape
    if Wm1 is not None:
        _, _, rm1, dm1 = Wm1.shape
    else:
        rm1, dm1, = rd1, d
    if Wp1 is not None:
        lp1, up1, _, _ = Wp1.shape
    else:
        lp1, up1 = lu2, u

    # 1.) Contract the two-site wave function and aply time evolution operator
    psi = _contract_full(T1, T2, Wm1, W, Wp1, U)

    # 2.) Reshape and split
    # ld1, lu1, ru2, rd2, rm1, dm1, lp1, up1, i, j -> ld1, lu1, rm1, dm1, i, ru2, rd2, lp1, up1, j
    psi = backend.transpose(psi, (0, 1, 4, 5, 8, 2, 3, 6, 7, 9))
    # ld1, lu1, rm1, dm1, i, ru2, rd2, lp1, up1, j -> (ld1, lu1, rm1, dm1, i), (ru2, rd2, lp1, up1, j)
    psi = backend.reshape(psi, (ld1 * lu1 * rm1 * dm1 * i, ru2 * rd2 * lp1 * up1 * j))
    lower, upper = utility.split_matrix_svd(psi, l * d)

    # 3.) Reconstruct upper part
    upper = backend.reshape(upper, (l * d, ru2, rd2, lp1, up1, j)) # (l, d), (ru2, rd2, lp1, up1, j) -> (l, d), ru2, rd2, lp1, up1, j
    if Wp1 is None:
        upper = backend.transpose(upper, (1, 2, 3, 5, 0, 4)) # (l, d), ru2, rd2, lp1, up1, j = (l, d), ru2, rd2, lu2, u, j -> ru2, rd2, lu2, j, (l, d), u
        upper = backend.reshape(upper, (ru2 * rd2 * lu2 * j, l * d * u)) # ru2, rd2, lu2, j, (l, d), u -> (ru2, rd2, lu2, j), ((l, d), u)
        T2, W_prime = utility.split_matrix_svd(upper, ld2)
        T2 = backend.reshape(T2, (ru2, rd2, lu2, j, ld2)) # (ru2, rd2, lu2, j), ld2 -> ru2, rd2, lu2, j, ld2
        T2 = backend.transpose(T2, (3, 0, 1, 4, 2)) # ru2, rd2, lu2, j, ld2 -> j, ru2, rd2, ld2, lu2
        W = backend.reshape(W_prime, (r, (l * d), u)) # r, ((l, d), u) -> r, (l, d), u
        W = backend.transpose(W, (2, 0, 1)) # r, (l, d), u -> u, r, (l, d)
    else:
        # Use tripartite decomposition to reconstruct upper part
        upper = backend.transpose(upper, (1, 2, 5, 3, 4, 0)) # (l, d), ru2, rd2, lp1, up1, j -> ru2, rd2, j, lp1, up1, (l, d)
        upper = backend.reshape(upper, (ru2*rd2*j, lp1*up1, l*d)) # ru2, rd2, j, lp1, up1, (l, d) -> (ru2, rd2, j), (lp1, up1), (l, d)
        if chi_max is not None:
            chi = min(chi_max, lp1*up1*lu2, l*u*ld2)
        else:
            chi = u
        T2, Wp1, W = tripartite_decomposition_lib.tripartite_decomposition(upper, lu2, ld2, chi, **kwargs)
        T2 = backend.reshape(T2, (ru2, rd2, j, lu2, ld2)) # (ru2, rd2, j), lu2, ld2 -> ru2, rd2, j, lu2, ld2
        T2 = backend.transpose(T2, (2, 0, 1, 4, 3)) # ru2, rd2, j, lu2, ld2 -> p2, ru2, rd2, ld2, lu2
        Wp1 = backend.reshape(Wp1, (lu2, lp1, up1, chi)) # lu2, (lp1, up1), chi -> lu2, lp1, up1, chi = rp1, lp1, up1, dp1
        Wp1 = backend.transpose(Wp1, (1, 2, 0, 3)) # rp1, lp1, up1, dp1 -> lp1, up1, rp1, dp1
        W = backend.transpose(W, (1, 0, 2)) # ld2, chi, (l, d) = r, u, (l, d) -> u, r, (l, d)
    
    # 4.) Reconstruct lower part
    lower = backend.tensordot(W, lower, ([2], [1])) # u r [(l d)]; (ld1 lu1 rm1 dm1 i) [(l d)] -> u r (ld1 lu1 rm1 dm1 i)
    u = W.shape[0]
    lower = backend.reshape(lower, (u, r, ld1, lu1, rm1, dm1, i)) # u, r, (ld1, lu1, rm1, dm1, i) -> u, r, ld1, lu1, rm1, dm1, i
    if Wm1 is None:
        lower = backend.transpose(lower, (2, 3, 4, 6, 0, 1, 5)) # u, r, ld1, lu1, rm1, dm1, i = u, r, ld1, lu1, rd1, d, i -> ld1, lu1, rd1, i, u, r, d
        lower = backend.reshape(lower, (ld1 * lu1 * rd1 * i, u * r * d)) # ld1, lu1, rd1, i, u, r, d -> (ld1, lu1, rd1, i) (u, r, d)
        T1, W_prime = utility.split_matrix_svd(lower, ru1) 
        T1 = backend.reshape(T1, (ld1, lu1, rd1, i, ru1)) # (ld1, lu1, rd1, i), ru1 -> ld1, lu1, rd1, i, ru1
        T1 = backend.transpose(T1, (3, 4, 2, 0, 1)) # ld1, lu1, rd1, i, ru1 -> i, ru1, rd1, ld1, lu1
        W = backend.reshape(W_prime, (l, u, r, d)) # (l, (u, r, d)
    else:
        # Use tripartite decomposition to reconstruct lower part
        lower = backend.transpose(lower, (2, 3, 6, 4, 5, 0, 1)) # u, r, ld1, lu1, rm1, dm1, i -> ld1, lu1, i, rm1, dm1, u, r
        lower = backend.reshape(lower, (ld1*lu1*i, rm1*dm1, u*r)) # ld1, lu1, i, rm1, dm1, u, r -> (ld1, lu1, i), (rm1, dm1), (u, r)
        if chi_max is not None:
            chi = min(chi_max, rm1*dm1*rd1, u*r*ru1)
        else:
            chi = d
        T1, Wm1, W = tripartite_decomposition_lib.tripartite_decomposition(lower, rd1, ru1, chi, **kwargs)
        T1 = backend.reshape(T1, (ld1, lu1, i, rd1, ru1)) # (ld1, lu1, i), rd1, ru1 -> ld1, lu1, i, rd1, ru1
        T1 = backend.transpose(T1, (2, 4, 3, 0, 1)) # ld1, lu1, i, rd1, ru1 -> i, ru1, rd1, ld1, lu1
        Wm1 = backend.reshape(Wm1, (rd1, rm1, dm1, chi)) # rd1, (rm1, dm1), chi -> rd1, rm1, dm1, chi = lm1, rm1, dm1, um1
        Wm1 = backend.transpose(Wm1, (0, 3, 1, 2)) # lm1, rm1, dm1, um1 -> lm1, um1, rm1, dm1
        W = backend.reshape(W, (ru1, chi, u, r)) # ru1, chi, (u, r) -> ru1, chi, u, r = l, d, u, r
        W = backend.transpose(W, (0, 2, 3, 1)) # l, d, u, r -> l, u, r, d

    return T1, T2, Wm1, W, Wp1

def tebd_step_iterate_polar(T1, T2, Wm1, W, Wp1, U, N_iters=100, eps=1e-13, debug_logger=debug_logging.DebugLogger()):
    """
    Computes the updated two-site wavefunction by iteratively optimizing the overlap
    with the original wavefunction with the time evolution operator applied, enforcing isometry
    conditions using polar decompositions. This is similar to the MPS TEBD implementation in
    Jakob Unfried, Johannes Hausschild, and Frank Pollmann: "Fast Time-Evolution of Matrix-Product States using the QR decomposition"
    (https://arxiv.org/abs/2212.09782).

    Parameters
    ----------
    T1, T2, Wm1, W, Wp1: backend.array_types
        two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
    U: backend.array_type of shape (i, j, i, j)
        time evolution operator
    N_iters: int, optional
        number of iterations. Default: 100.
    eps: float, optional
        if the relative change in error is smaller than this eps, the algorithm
        is terminated.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.

    Returns
    -------
    T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime : backend.array_types
        the updated wave function. For shapes of individual tensors, see tebd_step()
    """

    def _update_T1(T1, T2, Wm1, W, Wp1, T2_prime, Wm1_prime, W_prime, Wp1_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out T1_prime.
        The updated T1_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: backend.array_types
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
        T2_prime, Wm1_prime, W_prime, Wp1_prime: backend.array_types
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step().
        U: backend.array_type of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        T1_prime: backend.array_type of shape (i, ru1, rd1, ld1, lu1)
            updated T1_prime tensor.
        """
        # Contract everything except T1
        if Wm1 is None:
            if Wp1 is None:
                contr = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2, 4], [1, 2, 4])) # p2 [ru2] [rd2] ld2 [lu2]; p2* [ru2*] [rd2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
                contr = backend.tensordot(contr, U, ([0, 2], [3, 1])) # [p2] ld2 [p2*] ld2*; i [j] i* [j*] -> ld2 ld2* i i*
                contr = backend.tensordot(contr, W, ([0], [2])) # [ld2] ld2* i i*; l u [r] d -> ld2* i i* l u d
                contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 4, 5], [2, 1, 3])) # [ld2*] i i* l [u] [d]; l* [u*] [r*] [d*] -> i i* l l*
            else:
                contr = backend.tensordot(Wp1, backend.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
                temp = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2], [1, 2])) # p2 [ru2] [rd2] ld2 lu2; p2* [ru2*] [rd2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
                temp = backend.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i*
                contr = backend.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i*
                contr = backend.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d
                contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 1, 5], [1, 2, 3])) # [dp1*] [ld2*] i i* l [d]; l* [u*] [r*] [d*] -> i i* l l*
            contr = backend.tensordot(contr, T1, ([1, 2], [0, 1])) # i [i*] [l] l*; [p1] [ru1] rd1 ld1 lu1 -> i l* rd1 ld1 lu1 = p1 ru1 rd1 ld1 lu1
            T1_prime = backend.transpose(contr, (0, 3, 4, 2, 1)) # p1, ru1, rd1, ld1, lu1 -> p1, ld1, lu1, rd1, ru1
            p1, ru1, rd1, ld1, lu1 = U.shape[2], W_prime.shape[0], T1.shape[2], T1.shape[3], T1.shape[4]
        else:
            if Wp1 is None:
                contr = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2, 4], [1, 2, 4])) # p2 [ru2] [rd2] ld2 [lu2]; p2* [ru2*] [rd2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
                contr = backend.tensordot(contr, U, ([0, 2], [3, 1])) # [p2] ld2 [p2*] ld2*; i [j] i* [j*] -> ld2 ld2* i i*
                contr = backend.tensordot(contr, W, ([0], [2])) # [ld2] ld2* i i*; l u [r] d -> ld2* i i* l u d
                contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 4], [2, 1])) # [ld2*] i i* l [u] d; l* [u*] [r*] d* -> i i* l d l* d*
            else:
                contr = backend.tensordot(Wp1, backend.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
                temp = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2], [1, 2])) # p2 [ru2] [rd2] ld2 lu2; p2* [ru2*] [rd2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
                temp = backend.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i*
                contr = backend.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i*
                contr = backend.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d
                contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 1], [1, 2])) # [dp1*] [ld2*] i i* l d; l* [u*] [r*] d* -> i i* l d l* d*
            temp = backend.tensordot(Wm1, backend.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
            contr = backend.tensordot(temp, contr, ([1, 3], [3, 5])) # lm1 [um1] lm1* [um1*]; i i* l [d] l* [d*] -> lm1 lm1* i i* l l*
            contr = backend.tensordot(contr, T1, ([0, 3, 4], [2, 0, 1])) # [lm1] lm1* i [i*] [l] l*; [p1] [ru1] [rd1] ld1 lu1 -> lm1* i l* ld1 lu1
            T1_prime = backend.transpose(contr, (1, 3, 4, 0, 2)) # lm1*, i, l*, ld1, lu1 = rd1, p1, ru1, ld1, lu1 -> p1, ld1, lu1, rd1, ru1
            p1, ru1, rd1, ld1, lu1 = U.shape[2], W_prime.shape[0], Wm1_prime.shape[0], T1.shape[3], T1.shape[4]
        # isometrize T1
        if Wm1 is None:
            T1_prime = backend.reshape(T1_prime, (p1*ld1*lu1*rd1, ru1)) # p1, ld1, lu1, rd1, ru1 -> (p1, ld1, lu1, rd1), ru1
            T1_prime = utility.isometrize_polar(T1_prime)
            T1_prime = backend.reshape(T1_prime, (p1, ld1, lu1, rd1, ru1)) # (p1, ld1, lu1, rd1), ru1 -> p1, ld1, lu1, rd1, ru1
        else:
            T1_prime = backend.reshape(T1_prime, (p1*ld1*lu1, rd1*ru1)) # p1, ld1, lu1, rd1, ru1 -> (p1, ld1, lu1), (rd1, ru1)
            T1_prime = utility.isometrize_polar(T1_prime)
            T1_prime = backend.reshape(T1_prime, (p1, ld1, lu1, rd1, ru1)) # (p1, ld1, lu1), (rd1, ru1) -> p1, ld1, lu1, rd1, ru1
        T1_prime = backend.transpose(T1_prime, (0, 4, 3, 1, 2)) # p1, ld1, lu1, rd1, ru1 -> p1, ru1, rd1, ld1, lu1
        return T1_prime

    def _update_T2(T1, T2, Wm1, W, Wp1, T1_prime, Wm1_prime, W_prime, Wp1_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out T2_prime.
        The updated T2_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: backend.array_types
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
        T1_prime, Wm1_prime, W_prime, Wp1_prime: backend.array_types
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step().
        U: backend.array_type of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        T2_prime: backend.array_type of shape (j, ru2, rd2, ld2, lu2)
            updated T2_prime tensor.
        """
        # Contract everything except T2
        if Wm1 is None:
            contr = backend.tensordot(T1, backend.conj(T1_prime), ([2, 3, 4], [2, 3, 4])) # p1 ru1 [rd1] [ld1] [lu1]; p1* ru1* [rd1*] [ld1*] [lu1*] -> p1 ru1 p1* ru1*
            contr = backend.tensordot(contr, U, ([0, 2], [2, 0])) # [p1] ru1 [p1*] ru1*; [i] j [i*] j* -> ru1 ru1* j j*
            contr = backend.tensordot(contr, W, ([0], [0])) # [ru1] ru1* j j*; [l] u r d -> ru1* j j* u r d
            if Wp1 is None:
                contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 3, 5], [0, 1, 3])) # [ru1*] j j* [u] r [d]; [l*] [u*] r* [d*] -> j j* r r*
                contr = backend.tensordot(contr, T2, ([1, 2], [0, 3])) # j [j*] [r] r*; [p2] ru2 rd2 [ld2] lu2 -> j r* ru2 rd2 lu2 = p2 ld2 ru2 rd2 lu2
                T2_prime = backend.transpose(contr, (0, 2, 3, 1, 4)) # p2, ld2, ru2, rd2, lu2 -> p2, ru2, rd2, ld2, lu2
                p2, ru2, rd2, ld2, lu2 = U.shape[3], T2.shape[1], T2.shape[2], W_prime.shape[2], T2.shape[4]
            else:
                contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 5], [0, 3])) # [ru1*] j j* u r [d]; [l*] u* r* [d*] -> j j* u r u* r*
                temp = backend.tensordot(Wp1, backend.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
                contr = backend.tensordot(contr, temp, ([2, 4], [1, 3])) # j j* [u] r [u*] r*; rp1 [dp1] rp1* [dp1*] -> j j* r r* rp1 rp1*
                contr = backend.tensordot(contr, T2, ([1, 2, 4], [0, 3, 4])) # j [j*] [r] r* [rp1] rp1*; [p2] ru2 rd2 [ld2] [lu2] -> j r* rp1* ru2 rd2
                T2_prime = backend.transpose(contr, (0, 3, 4, 1, 2)) # j, r*, rp1*, ru2, rd2 = p2, ld2, lu2, ru2, rd2 -> p2, ru2, rd2, ld2, lu2
                p2, ru2, rd2, ld2, lu2 = U.shape[3], T2.shape[1], T2.shape[2], W_prime.shape[2], Wp1_prime.shape[2]
        else:
            contr = backend.tensordot(Wm1, backend.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
            temp = backend.tensordot(T1, backend.conj(T1_prime), ([3, 4], [3, 4])) # p1 ru1 rd1 [ld1] [lu1]; p1* ru1* rd1* [ld1*] [lu1*] -> p1 ru1 rd1 p1* ru1* rd1* 
            temp = backend.tensordot(temp, U, ([0, 3], [2, 0])) # [p1] ru1 rd1 [p1*] ru1* rd1*; [i] j [i*] j* -> ru1 rd1 ru1* rd1* j j*
            contr = backend.tensordot(contr, temp, ([0, 2], [1, 3])) # [lm1] um1 [lm1*] um1*; ru1 [rd1] ru1* [rd1*] j j* -> um1 um1* ru1 ru1* j j*
            contr = backend.tensordot(contr, W, ([0, 2], [3, 0])) # [um1] um1* [ru1] ru1* j j*; [l] u r [d] -> um1* ru1* j j* u r
            if Wp1 is None:
                contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 1, 4], [3, 0, 1])) # [um1*] [ru1*] j j* [u] r; [l*] [u*] r* [d*] -> j j* r r*
                contr = backend.tensordot(contr, T2, ([1, 2], [0, 3])) # j [j*] [r] r*; [p2] ru2 rd2 [ld2] lu2 -> j r* ru2 rd2 lu2 = p2 ld2 ru2 rd2 lu2
                T2_prime = backend.transpose(contr, (0, 2, 3, 1, 4)) # p2, ld2, ru2, rd2, lu2 -> p2, ru2, rd2, ld2, lu2
                p2, ru2, rd2, ld2, lu2 = U.shape[3], T2.shape[1], T2.shape[2], W_prime.shape[2], T2.shape[4]
            else: 
                contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 1], [3, 0])) # [um1*] [ru1*] j j* u r; [l*] u* r* [d*] -> j j* u r u* r*
                temp = backend.tensordot(Wp1, backend.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
                contr = backend.tensordot(contr, temp, ([2, 4], [1, 3])) # j j* [u] r [u*] r*; rp1 [dp1] rp1* [dp1*] -> j j* r r* rp1 rp1*
                contr = backend.tensordot(contr, T2, ([1, 2, 4], [0, 3, 4])) # j [j*] [r] r* [rp1] rp1*; [p2] ru2 rd2 [ld2] [lu2] -> j r* rp1* ru2 rd2
                T2_prime = backend.transpose(contr, (0, 3, 4, 1, 2)) # j, r*, rp1*, ru2, rd2 = p2, ld2, lu2, ru2, rd2 -> p2, ru2, rd2, ld2, lu2
                p2, ru2, rd2, ld2, lu2 = U.shape[3], T2.shape[1], T2.shape[2], W_prime.shape[2], Wp1_prime.shape[2]
        # Isometrize T2
        if Wp1 is None:
            T2_prime = backend.transpose(T2_prime, (0, 1, 2, 4, 3)) # p2, ru2, rd2, ld2, lu2 -> p2, ru2, rd2, lu2, ld2
            T2_prime = backend.reshape(T2_prime, (p2*ru2*rd2*lu2, ld2)) # p2, ru2, rd2, lu2, ld2 -> (p2, ru2, rd2, lu2), ld2
            T2_prime = utility.isometrize_polar(T2_prime)
            T2_prime = backend.reshape(T2_prime, (p2, ru2, rd2, lu2, ld2)) # (p2, ru2, rd2, lu2), ld2 -> p2, ru2, rd2, lu2, ld2
            T2_prime = backend.transpose(T2_prime, (0, 1, 2, 4, 3)) # p2, ru2, rd2, lu2, ld2 -> p2, ru2, rd2, ld2, lu2
        else:
            T2_prime = backend.reshape(T2_prime, (p2*ru2*rd2, ld2*lu2)) # p2, ru2, rd2, ld2, lu2 -> (p2, ru2, rd2), (ld2, lu2)
            T2_prime = utility.isometrize_polar(T2_prime)
            T2_prime = backend.reshape(T2_prime, (p2, ru2, rd2, ld2, lu2)) # (p2, ru2, rd2), (ld2, lu2) -> p2, ru2, rd2, ld2, lu2
        return T2_prime

    def _update_Wm1(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, W_prime, Wp1_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out Wm1_prime.
        The updated Wm1_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: backend.array_types
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
        T1_prime, T2_prime, W_prime, Wp1_prime: backend.array_types
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step().
        U: backend.array_type of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        Wm1_prime: backend.array_type of shape (lm1, um1, rm1, dm1)
            updated Wm1_prime tensor.
        """
        if Wm1 is None:
            return None
        # Contract everything except Wm1
        if Wp1 is None:
            contr = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2, 4], [1, 2, 4])) # p2 [ru2] [rd2] ld2 [lu2]; p2* [ru2*] [rd2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
            contr = backend.tensordot(contr, U, ([0, 2], [3, 1])) # [p2] ld2 [p2*] ld2*; i [j] i* [j*] -> ld2 ld2* i i*
            contr = backend.tensordot(contr, W, ([0], [2])) # [ld2] ld2* i i*; l u [r] d -> ld2* i i* l u d
            contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 4], [2, 1])) # [ld2*] i i* l [u] d; l* [u*] [r*] d* -> i i* l d l* d*
        else:
            contr = backend.tensordot(Wp1, backend.conj(Wp1_prime), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
            temp = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2], [1, 2])) # p2 [ru2] [rd2] ld2 lu2; p2* [ru2*] [rd2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
            temp = backend.tensordot(temp, U, ([0, 3], [3, 1])) # [p2] ld2 lu2 [p2*] ld2* lu2*; i [j] i* [j*] -> ld2 lu2 ld2* lu2* i i*
            contr = backend.tensordot(contr, temp, ([0, 2], [1, 3])) # [rp1] dp1 [rp1*] dp1*; ld2 [lu2] ld2* [lu2*] i i* -> dp1 dp1* ld2 ld2* i i*
            contr = backend.tensordot(contr, W, ([0, 2], [1, 2])) # [dp1] dp1* [ld2] ld2* i i*; l [u] [r] d -> dp1* ld2* i i* l d
            contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 1], [1, 2])) # [dp1*] [ld2*] i i* l d; l* [u*] [r*] d* -> i i* l d l* d*
        temp = backend.tensordot(T1, backend.conj(T1_prime), ([3, 4], [3, 4])) # p1 ru1 rd1 [ld1] [lu1]; p1* ru1* rd1* [ld1*] [lu1*] -> p1 ru1 rd1 p1* ru1* rd1*
        contr = backend.tensordot(temp, contr, ([0, 1, 3, 4], [1, 2, 0, 4])) # [p1] [ru1] rd1 [p1*] [ru1*] rd1*; [i] [i*] [l] d [l*] d* -> rd1 rd1* d d*
        contr = backend.tensordot(contr, Wm1, ([0, 2], [0, 1])) # [rd1] rd1* [d] d*; [lm1] [um1] rm1 dm1 -> rd1* d* rm1 dm1
        # Isometrize Wm1
        Wm1_prime = backend.transpose(contr, (0, 2, 3, 1)) # rd1*, d*, rm1, dm1 = lm1, um1, rm1, dm1 -> lm1, rm1, dm1, um1
        lm1, um1, rm1, dm1 = T1_prime.shape[2], W_prime.shape[3], Wm1.shape[2], Wm1.shape[3]
        Wm1_prime = backend.reshape(Wm1_prime, (lm1*rm1*dm1, um1)) # lm1, rm1, dm1, um1 -> (lm1, rm1, dm1), um1
        Wm1_prime = utility.isometrize_polar(Wm1_prime)
        Wm1_prime = backend.reshape(Wm1_prime, (lm1, rm1, dm1, um1)) # (lm1, rm1, dm1), um1 -> lm1, rm1, dm1, um1
        Wm1_prime = backend.transpose(Wm1_prime, (0, 3, 1, 2)) # lm1, rm1, dm1, um1 -> lm1, um1, rm1, dm1
        return Wm1_prime

    def _update_W(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out W_prime.
        The updated W_prime is then computed from the contraction, normalized, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: backend.array_types
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
        T1_prime, T2_prime, Wm1_prime, Wp1_prime: backend.array_types
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step().
        U: backend.array_type of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        W_prime: backend.array_type of shape (l, u, r, d)
            updated W_prime tensor.
        """
        # contract everything except W
        W_prime = _compute_overlap_except_W_prime(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, Wp1_prime, U)
        # Update W
        W_prime /= backend.norm(W_prime)
        return W_prime

    def _update_Wp1(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, U):
        """
        Helper function that contracts the two-site wavefunction, leaving out Wp1_prime.
        The updated Wp1_prime is then computed from the contraction, isometrized using a polar decomposition, and returned.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: backend.array_types
            two site wavefunction psi. For the shapes of the individual tensors, see tebd_step().
        T1_prime, T2_prime, Wm1_prime, W_prime: backend.array_types
            two site wavefunction psi_prime. For the shapes of the individual tensors, see tebd_step().
        U: backend.array_type of shape (i, j, i, j)
            time evolution operator

        Returns
        -------
        Wp1_prime: backend.array_type of shape (l, u, r, d)
            updated Wp1_prime tensor.
        """
        if Wp1 is None:
            return None
        # Contract everything except Wp1
        if Wm1 is None:
            contr = backend.tensordot(T1, backend.conj(T1_prime), ([2, 3, 4], [2, 3, 4])) # p1 ru1 [rd1] [ld1] [lu1]; p1* ru1* [rd1*] [ld1*] [lu1*] -> p1 ru1 p1* ru1*
            contr = backend.tensordot(contr, U, ([0, 2], [2, 0])) # [p1] ru1 [p1*] ru1*; [i] j [i*] j* -> ru1 ru1* j j*
            contr = backend.tensordot(contr, W, ([0], [0])) # [ru1] ru1* j j*; [l] u r d -> ru1* j j* u r d
            contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 5], [0, 3])) # [ru1*] j j* u r [d]; [l*] u* r* [d*] -> j j* u r u* r*
        else:
            contr = backend.tensordot(Wm1, backend.conj(Wm1_prime), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
            temp = backend.tensordot(T1, backend.conj(T1_prime), ([3, 4], [3, 4])) # p1 ru1 rd1 [ld1] [lu1]; p1* ru1* rd1* [ld1*] [lu1*] -> p1 ru1 rd1 p1* ru1* rd1* 
            temp = backend.tensordot(temp, U, ([0, 3], [2, 0])) # [p1] ru1 rd1 [p1*] ru1* rd1*; [i] j [i*] j* -> ru1 rd1 ru1* rd1* j j*
            contr = backend.tensordot(contr, temp, ([0, 2], [1, 3])) # [lm1] um1 [lm1*] um1*; ru1 [rd1] ru1* [rd1*] j j* -> um1 um1* ru1 ru1* j j*
            contr = backend.tensordot(contr, W, ([0, 2], [3, 0])) # [um1] um1* [ru1] ru1* j j*; [l] u r [d] -> um1* ru1* j j* u r
            contr = backend.tensordot(contr, backend.conj(W_prime), ([0, 1], [3, 0])) # [um1*] [ru1*] j j* u r; [l*] u* r* [d*] -> j j* u r u* r*
        temp = backend.tensordot(T2, backend.conj(T2_prime), ([1, 2], [1, 2])) # p2 [ru2] [rd2] ld2 lu2; p2* [ru2*] [rd2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
        contr = backend.tensordot(contr, temp, ([0, 1, 3, 5], [3, 0, 1, 4])) # [j] [j*] u [r] u* [r*]; [p2] [ld2] lu2 [p2*] [ld2*] lu2* -> u u* lu2 lu2*
        contr = backend.tensordot(contr, Wp1, ([0, 2], [3, 2])) # [u] u* [lu2] lu2*; lp1 up1 [rp1] [dp1] -> u* lu2* lp1 up1
        # Isometrize Wp1
        Wp1_prime = backend.transpose(contr, (2, 3, 1, 0)) # u*, lu2*, lp1, up1 = dp1, rp1, lp1, up1 -> lp1, up1, rp1, dp1
        lp1, up1, rp1, dp1 = Wp1.shape[0], Wp1.shape[1], T2_prime.shape[4], W_prime.shape[1]
        Wp1_prime = backend.reshape(Wp1_prime, (lp1*up1*rp1, dp1)) # lp1, up1, rp1, dp1 -> (lp1, up1, rp1), dp1
        Wp1_prime = utility.isometrize_polar(Wp1_prime)
        Wp1_prime = backend.reshape(Wp1_prime, (lp1, up1, rp1, dp1)) # (lp1, up1, rp1), dp1 -> lp1, up1, rp1, dp1
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

    if debug_logger.log_iterative_local_tebd_update_errors_per_iteration:
        errors = []

    T1_prime = T1.copy()
    T2_prime = T2.copy()
    Wm1_prime = None
    if Wm1 is not None:
        Wm1_prime = Wm1.copy()
    W_prime = W.copy()
    Wp1_prime = None
    if Wp1 is not None:
        Wp1_prime = Wp1.copy()
    error = _compute_error(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U)
    num_iters = N_iters
    for n in range(N_iters):
        # Update
        T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime = _update_tensors(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U)
        # Recompute error
        new_error = _compute_error(T1, T2, Wm1, W, Wp1, T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime, U)
        if debug_logger.log_iterative_local_tebd_update_errors_per_iteration:
            errors.append(new_error)
        # Check stopping criterion
        if error == 0.0 or backend.abs((error - new_error) / error) <= eps:
            num_iters = n + 1
            break
        error = new_error
    if debug_logger.log_iterative_local_tebd_update_errors_per_iteration:
        debug_logger.append_to_log_list(("local_tebd_update_errors_per_iteration"), errors)
    if debug_logger.log_iterative_local_tebd_update_info:
        debug_logger.append_to_log_list(("iterative_local_tebd_update_info", "N_iters"), num_iters)
    return T1_prime, T2_prime, Wm1_prime, W_prime, Wp1_prime