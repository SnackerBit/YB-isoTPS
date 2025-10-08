from .. import backend
from .. import utility
from . import disentangle_renyi_alpha
from . import disentangle_trunc_error
from . import disentangle_renyi_alpha_trunc_error
from .. import debug_logging

def initialize_disentangle(theta, init_U="polar", N_iters_pre_disentangler=200):
    """
    Initializes the disentangling procedure by computing an initial disentangling unitary U0 and applying it to the
    wavefunction tensor theta.

    Parameters
    ----------
    theta : backend.array_type of shape (l, i, j, r)
        wavefunction tensor to be disentangled.
    init_U : str, one of {"polar", "identity", "qr", "random"}, optional
        String selecting the method for initializing the disentangling unitary U. Default : "polar".
    N_iters_pre_disentangler : int, optional
        number of pre-disentangling iterations done before tha actual call to the disentangler. The pre-disentangler
        uses the fast power method to minimize the renyi-2 entropy. Default : 200.

    Returns
    -------
    U0 : backend.array_type of shape (i, j, i*, j*)
        initial disentangling unitary.
    theta : backend.array_type of shape (l, i, j, r)
        wavefunction tensor multiplied with U0.
    """
    # Initialize disentangling unitary
    ml, d1, d2, mr = theta.shape
    if init_U == "identity":
        U0 = backend.eye(d1*d2)
    elif init_U == "polar":
        # Polar initialization taken from [1, 2].
        if ml >= d1 and mr >= d2:
            theta = theta.transpose(1, 2, 0, 3).reshape((d1*d2, ml, mr)) # ml, d1, d2, mr -> d1, d2, ml, mr -> (d1, d2), ml, mr { D^6 }
            psi = theta.copy()
            if ml > d1:
                rho = backend.tensordot(psi, psi.conj(), ([0, 2], [0, 2])) # [(d1 d2)] ml [mr]; [(d1 d2)*] ml* [mr*] -> ml ml* { D^8 }
                p, u = backend.eigh(rho) # { D^6 }
                # eigenvalues from backend.eigh are in ascending order: Take the d1 largest ones!
                u = u[:, -d1:]
                psi = backend.tensordot(psi, u.conj(), ([1], [0])).transpose([0, 2, 1]) # (d1 d2) [ml] mr; [ml*] d1* -> (d1 d2) mr d1* -> (d1 d2) d1* mr { D^7 }
            if mr > d2:
                rho = backend.tensordot(psi, psi.conj(), ([0, 1], [0, 1])) # [(d1 d2)] [ml] mr; [(d1 d2)*] [ml*] mr* -> mr mr* { D^8 }
                p, u = backend.eigh(rho) # { D^6 }
                # eigenvalues from backend.eigh are in ascending order: Take the d2 largest ones!
                u = u[:, -d2:]
                psi = backend.tensordot(psi, u.conj(), ([2], [0])) # (d1 d2) ml [mr]; [mr*] d2* -> (d1 d2) ml d2* { D^7 }
            # Renormalize
            psi /= backend.norm(psi)
            # Isometrize using polar decomposition
            u, s, v = utility.safe_svd(psi.reshape(d1*d2, d1*d2), full_matrices=False)
            Zp = backend.dot(u, v)
            U0 = Zp.T.conj()
            theta = backend.dot(U0, theta.reshape(d1*d2, ml*mr))
            theta = theta.reshape(d1, d2, ml, mr).transpose(2, 0, 1, 3) # (d1, d2), (ml, mr) -> d1, d2, ml, mr ->  ml, d1, d2, mr { D^6 }
        else:
            U0 = backend.eye(d1*d2)
    elif init_U == "qr":
        U0, theta = utility.split_matrix_svd(theta.transpose(1, 2, 0, 3).reshape((d1*d2, ml*mr)), d1*d2) # ml, d1, d2, mr -> d1, d2, ml, mr -> (d1, d2), (ml, mr) { D^6 }
        U0 = U0.T.conj()
        theta = theta.reshape(d1, d2, ml, mr).transpose(2, 0, 1, 3) # (d1, d2), (ml, mr) -> d1, d2, ml, mr -> ml, d1, d2, mr
    elif init_U == "random":
        raise NotImplementedError("init_U == random is not implemented")
    else:
        raise NotImplementedError(f"init_U == \"{init_U}\" is not implemented!")

    # Perform initial disentangling using the fast renyi-2 disentangler
    if N_iters_pre_disentangler > 0:
        U = disentangle_renyi_alpha.disentangle(theta, debug_logger=debug_logging.DebugLogger(), renyi_alpha=2.0, method="power_iteration", N_iters=N_iters_pre_disentangler)
        theta = backend.tensordot(U, theta, ([2, 3], [1, 2])).transpose(2, 0, 1, 3) # i j [i*] [j*]; ml [d1] [d2] mr -> i j ml mr -> ml i j mr
        U0 = backend.dot(U.reshape(d1*d2, d1*d2), U0)
    
    return U0, theta

def disentangle(theta, mode="renyi", init_U="polar", N_iters_pre_disentangler=200, chi=None, debug_logger=debug_logging.DebugLogger(), **kwargs):
    r"""
    Disentangles a given wavefunction tensor theta by optimizing over the unitary U:
    
            ___________                               ___________                                                                                 
           |           |                             |           |                                                                                      
    ml ----|   theta   |---- mR       =       ml ----|   theta   |---- mR                                                                      
           |___________|                             |___________|                                                                                                            
             |       |                                _|_______|_                                                                                                                 
             |       |                               |_U^\dagger |                                                                                    
             d1      d2                               _|_______|_   
                                                     |_____U_____|
                                                       |       |           
                                                                        
    After the disentangling process, the singular values of theta if we split between the grouped (ml, d1), (d2, mr) legs should
    fall off faster, ie. the first few singular values become larger, whereas the other singular values become smaller.
    This allows for lower truncation error. Implemented disentanglers minimize either the Renyi-alpha-entropy or the truncation error,
    see also
    [1]: M. P. Zaletel and F. Pollmann, "Isometric tensor network states in two dimensions", https://arxiv.org/abs/2112.08394
    [2]: S.-H. Lin, M. P. Zaletel and F. Pollmann, "Efficient Simulation of Dynamics in Two-Dimensional Quantum Spin Systems with Isometric Tensor Networks", https://arxiv.org/abs/1902.05100

    Parameters
    ----------
    theta : backend.array_type of shape (l, i, j, r)
        wavefunction tensor to be disentangled.
    mode : str, one of {"renyi", "trunc", "renyi_approx", "trunc_approx", "renyi_trunc", "renyi_trunc_approx", "none"}, optional
        String selecting the cost function that is used for disentangling. Default : "renyi". If this is set to "none",
        no disentangling is made. This can be used for debugging purposes, when only the init_U functionality is to be used
        but no disentangling.
    init_U : str, one of {"polar", "identity", "qr", "random"}, optional
        String selecting the method for initializing the disentangling unitary U. Default : "polar".
    N_iters_pre_disentangler : int, optional
        number of pre-disentangling iterations done before tha actual call to the disentangler. The pre-disentangler
        uses the fast power method to minimize the renyi-2 entropy. Default : 200.
    chi : int or None, optional
        truncated bond dimension used for the computation of the truncation error cost function.
        this parameter is not needed when using "trunc" as the cost function.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.
    **kwargs
        remaining kwargs are passed into the respective method chosen with method.
        See the different called functions for more information.

    Returns
    -------
    U_final : backend.array_type of shape (i, j, i*, j*)
        final disentangling unitary after optimization
    """
    _, d1, d2, _ = theta.shape
    # Initialize disentangling unitary
    U0, theta = initialize_disentangle(theta, init_U, N_iters_pre_disentangler)
    if debug_logger.disentangling_log_iterates:
        debug_logger.append_to_log_list(("disentangler_info", "U0s"), U0.reshape(d1, d2, d1, d2))
    # Perform disentangling
    if mode == "renyi":
        U = disentangle_renyi_alpha.disentangle(theta, debug_logger=debug_logger, **kwargs)
    elif mode == "renyi_approx":
        U = disentangle_renyi_alpha.disentangle_approx(theta, chi, debug_logger=debug_logger, **kwargs)
    elif mode == "trunc":
        U = disentangle_trunc_error.disentangle(theta, chi, debug_logger=debug_logger, **kwargs)
    elif mode == "trunc_approx":
        U = disentangle_trunc_error.disentangle_approx(theta, chi, debug_logger=debug_logger, **kwargs)
    elif mode == "renyi_trunc":
        assert("options_renyi" in kwargs and type("options_renyi") == dict)
        assert("options_trunc" in kwargs and type("options_trunc") == dict)
        U = disentangle_renyi_alpha_trunc_error.disentangle(theta, chi, options_renyi=kwargs["options_renyi"], options_trunc=kwargs["options_trunc"], debug_dict=debug_dict)
    elif mode == "renyi_trunc_approx":
        assert("options_renyi" in kwargs and type("options_renyi") == dict)
        assert("options_trunc" in kwargs and type("options_trunc") == dict)
        U = disentangle_renyi_alpha_trunc_error.disentangle_approx(theta, chi, options_renyi=kwargs["options_renyi"], options_trunc=kwargs["options_trunc"], debug_dict=debug_dict)
    elif mode == "none":
        return U0.reshape(d1, d2, d1, d2)
    else:
        temp = r'{"renyi", "trunc", "renyi_approx", "trunc_approx", "renyi_trunc", "renyi_trunc_approx"}'
        raise NotImplementedError(f"disentangling with mode \"{mode}\" is not implemented. Choose one of {temp}!")

    # Compute final disentangling tensor
    U = backend.dot(U.reshape(d1*d2, d1*d2), U0).reshape(d1, d2, d1, d2)
    return U