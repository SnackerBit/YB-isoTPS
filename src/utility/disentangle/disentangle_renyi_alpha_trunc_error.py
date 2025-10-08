from .. import backend
from . import disentangle_renyi_alpha
from . import disentangle_trunc_error
from .. import utility
from .. import debug_logging

def disentangle(theta, chi, options_renyi, options_trunc, debug_logger=debug_logging.DebugLogger()):
    """
    disentangles the wavefunction tensor theta by first minimizing the renyi entropy and
    then minimizing the truncation error.

    Parameters
    ----------
    theta : backend.array_type of shape (l, i, j, r)
        wavefunction tensor to be disentangled.
    chi : int
        bond dimension that the SVD of Utheta is truncated to. Smaller chi will speed up the algorithm
        but generally lead to worse results.
    options_renyi : dict
        dictionary with options that get passed as kwargs to the renyi alpha disentangler
        See src/utility/disentangle/disentangle_renyi_alpha.py for more information.
    options_trunc : dict
        dictionary with options that get passed as kwargs to the trunc error disentangler.
        See src/utility/disentangle/disentangle_trunc_error.py for more information.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.

    Returns
    -------
    U_final : backend.array_type of shape (i, j, i*, j*)
        final disentangling unitary after optimization
    """
    _, d1, d2, _ = theta.shape
    # Disentangle with Renyi disentangler
    debug_suffix = debug_logger.key_suffix
    debug_logger.key_suffix += "_renyi_alpha"
    U = disentangle_renyi_alpha.disentangle(theta, debug_logger=debug_logger, **options_renyi)
    theta = backend.tensordot(U, theta, ([2, 3], [1, 2])).transpose(2, 0, 1, 3) # i j [i*] [j*]; ml [d1] [d2] mr -> i j ml mr -> ml i j mr
    # Disentangle with truncation error disentangler
    debug_logger.key_suffix = debug_suffix + "_renyi_alpha"
    U2 = disentangle_trunc_error.disentangle(theta, chi, debug_logger=debug_logger, **options_trunc)
    U = backend.dot(U2.reshape(d1*d2, d1*d2), U).reshape(d1, d2, d1, d2)
    debug_logger.key_suffix = debug_suffix
    return U

def disentangle_approx(theta, chi, options_renyi, options_trunc, debug_logger=debug_logging.DebugLogger()):
    """
    disentangles the wavefunction tensor theta by first minimizing the approximate renyi entropy 
    and then minimizing the approximate truncation error.

    Parameters
    ----------
    theta : backend.array_type of shape (l, i, j, r)
        wavefunction tensor to be disentangled.
    chi : int
        bond dimension that the SVD of Utheta is truncated to. Smaller chi will speed up the algorithm
        but generally lead to worse results.
    options_renyi : dict
        dictionary with options that get passed as kwargs to the renyi alpha disentangler
        See src/utility/disentangle/disentangle_renyi_alpha.py for more information.
    options_trunc : dict
        dictionary with options that get passed as kwargs to the trunc error disentangler.
        See src/utility/disentangle/disentangle_trunc_error.py for more information.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.

    Returns
    -------
    U_final : backend.array_type of shape (i, j, i*, j*)
        final disentangling unitary after optimization
    """
    _, d1, d2, _ = theta.shape
    # Disentangle with Renyi disentangler
    debug_suffix = debug_logger.key_suffix
    debug_logger.key_suffix += "_renyi_alpha"
    U = disentangle_renyi_alpha.disentangle_approx(theta, chi, debug_logger=debug_logger, **options_renyi)
    theta = backend.tensordot(U, theta, ([2, 3], [1, 2])).transpose(2, 0, 1, 3) # i j [i*] [j*]; ml [d1] [d2] mr -> i j ml mr -> ml i j mr
    # Disentangle with truncation error disentangler
    debug_logger.key_suffix = debug_suffix + "_renyi_alpha"
    U2 = disentangle_trunc_error.disentangle_approx(theta, chi, debug_logger=debug_logger, **options_trunc)
    U = backend.dot(U2.reshape(d1*d2, d1*d2), U).reshape(d1, d2, d1, d2)
    debug_logger.key_suffix = debug_suffix
    return U