from ...utility import utility
from ...utility import backend
from ...utility.tripartite_decomposition import tripartite_decomposition
from ...utility import debug_logging

def yang_baxter_move_1(W1, W2, T, D_max, debug_logger=debug_logging.DebugLogger()):
    r"""
    Performs the first of two variants of the Yang-Baxter move necessary for the honeycomb lattice.

    \   |                          \           |     
     \  |                           \          |     
      \ |                            \         |     
       (W2)                           \        |     
        | \  |                         \  |    |     
        |  \ |                          \ |    |     
        |   \|                           \|    |     
        |    T-------      ->             T->-(W)----
        |   /                            /     |     
        |  /                            /      |     
        | /                            /       |     
       (W1)                           /        |     
      / |                            /         |     
     /  |                           /          |     
    /   |                          /           |     

    This is the simpler of the two moves. It is easily implemented using only contractions and one truncated SVD.

    Parameters
    ----------
    W1: backend.array_type of shape (l, u, r, d)
        either isometric tensor part of the ortho surface, or orthogonality center.
        Either W1 or W2 can also be None, but not both.  
    W2: backend.array_type of shape (lp1, up1, rp1, dp1)
        either isometric tensor part of the ortho surface, or orthogonality center
        Either W1 or W2 can also be None, but not both.
    T: backend.array_type of shape (i, r, ld, lu)
        isometric site tensor
    D_max: int 
        maximal bond dimension of T tensors
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.

    Returns
    -------
    W1': backend.array_type of shape (l, u, r, d)
        updated orthogonality center
    T': backend.array_type of shape (i, ru, rd, ld, lu)
        updated isometric site tensor
    error: float
        If debug_logger.log_yb_move_errors is True or debug_logger.log_approximate_column_error_yb is True, the relative error 
        norm(contr_before - contr_after) / norm(contr_before) is returned. Else, -float("inf") is returned.
    """
    # Contract everything together
    if W1 is None:
        contr = backend.tensordot(W2, T, ([2], [3])) # lp1 up1 [rp1] dp1; i r ld [lu] -> lp1 up1 dp1 i r ld
        contr = contr.transpose(3, 5, 0, 1, 4, 2) # lp1, up1, dp1, i, r, ld -> i, ld, lp1, up1, r, dp1 = i, ld, lu, u, r, d
    elif W2 is None:
        contr = backend.tensordot(W1, T, ([2], [2])) # l u [r] d; i r [ld] lu -> l u d i r lu
        contr = contr.transpose(3, 0, 5, 1, 4, 2) # l, u, d, i, r, lu -> i, l, lu, u, r, d = i, ld, lu, u, r, d
    else:
        contr = backend.tensordot(W1, W2, ([1], [3])) # l [u] r d; lp1 up1 rp1 [dp1] -> l r d lp1 up1 rp1
        contr = backend.tensordot(contr, T, ([1, 5], [2, 3])) # l [r] d lp1 up1 [rp1]; i r [ld] [lu] -> i l d lp1 up1 i r
        contr = contr.transpose(4, 0, 2, 3, 5, 1) # l, d, lp1, up1, i, r -> i, l, lp1, up1, r, d = i, ld, lu, u, r, d
    # Perform truncated SVD to split into T and W
    i, ld, lu, u, r, d = contr.shape
    contr = contr.reshape(i*ld*lu, u*r*d) # i, ld, lu, u, r, d -> (i, ld, lu), (u, r, d)
    T, W = utility.split_matrix_svd(contr, D_max)
    # Compute error
    log_error = debug_logger.log_yb_move_errors or debug_logger.log_approximate_column_error_yb
    error = -float("inf")
    if log_error:
        error = backend.norm(T@W - contr) / backend.norm(contr)
    # Reshape and transpose to obtain updated tensors
    T = T.reshape(i, ld, lu, -1).transpose(0, 3, 1, 2) # (i, ld, lu), r -> i, ld, lu, r -> i, r, ld, lu
    W = W.reshape(W.shape[0], u, r, d) # l, (u, r, d) -> l, u, r, d
    return W, T, error

def yang_baxter_move_2(W, T, D_max, chi_max, mode="both", larger_bond_direction="down", options={"mode": "svd"}, debug_logger=debug_logging.DebugLogger()):
    r"""
    Performs the second of two variants of the Yang-Baxter move necessary for the honeycomb lattice.

         |           /                       |  /   
         |          /                        | /   
         |         /                         |/   
         |        /                        (W2)    
         |    |  /                      |  / |   
         |    | /                       | >  | 
         |    |/                        |/   |
    ----(W)---T             ->      ----T    |                
         |     \                         \   |
         |      \                         >  |  
         |       \                         \ |   
         |        \                        (W1)    
         |         \                         |\    
         |          \                        | \    
         |           \                       |  \    

    This is the harder of the two moves. The implementation is similar to the YB move on the square lattice.
    Disentangling is recommended to decrease the error of the YB move.

    Parameters
    ----------
    W: backend.array_type of shape (l, u, r, d)
        orthogonality center.
    T: backend.array_type of shape (i, ru, rd, l)
        isometric site tensor
    D_max: int
        maximal bond dimension of T tensors
    chi_max: int 
        maximal bond dimension of the orthogonality surface
    mode: string, one of ["up", "down", "both"]
        used to switch between normal YB move and edge cases. If set to "up", W1' will be None. 
        If set to "down", W2' will be None. If set to "both", both W1' and W2' will not be None.
    options: dictionary, optional
       dictionary specifying which algorithm should be used, and options for these algorithms.
       Options are passed as keyword arguments to the tripartite decomposition subroutine, see
       "src/utility/tripartite_decomposition/tripartite_decomposition.py" for more information.
       Default value: {"mode" : "svd"}.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.
    Returns
    -------
    W1': backend.array_type of shape (l, u, r, d) or None
        updated W1, part of the ortho surface. If mode == "up", this is None.
    W2': backend.array_type of shape (lp1, up1, rp1, dp1)
        updated W2, part of the ortho surface. If mode == "down", this is None.
    T': backend.array_type of shape (i, ru, rd, l)
        updated isometric site tensor
    error: float
        If debug_logger.log_yb_move_errors is True or debug_logger.log_approximate_column_error_yb is True, the relative error 
        norm(contr_before - contr_after) / norm(contr_before) is returned. Else, -float("inf") is returned.
    """
    log_error = debug_logger.log_yb_move_errors or debug_logger.log_approximate_column_error_yb
    # 1.) Check for and handle edge cases
    if mode != "both":
        psi = backend.tensordot(W, T, ([2], [3])) # l u [r] d; i ru rd [l] -> l u d i ru rd
        if mode == "up":
            psi = psi.transpose(0, 3, 5, 1, 4, 2) # l, u, d, i, ru, rd = l, u, d, i, r, rd -> l, i, rd, u, r, d
            l, i, rd, u, r, d = psi.shape
            psi = psi.reshape(l*i*rd, u*r*d)
            T, W = utility.split_matrix_svd(psi, D_max) # l, i, rd, u, r, d -> (l, i, rd), (u, r, d)
            error = -float("inf")
            if log_error:
                error = backend.norm(psi - T@W) / backend.norm(psi)
            T = T.reshape(l, i, rd, -1).transpose(1, 3, 2, 0) # (l, i, rd), ru -> l, i, rd, ru -> i, ru, rd, l
            W = W.reshape(W.shape[0], u, r, d) # l, (u, r, d) -> l, u, r, d
            return None, W, T, error
        elif mode == "down":
            psi = psi.transpose(0, 3, 4, 1, 5, 2) # l, u, d, i, ru, rd = l, u, d, i, ru, r -> l, i, ru, u, r, d
            l, i, ru, u, r, d = psi.shape
            psi = psi.reshape(l*i*ru, u*r*d)
            T, W = utility.split_matrix_svd(psi, D_max) # l, i, ru, u, r, d -> (l, i, ru), (u, r, d)
            error = -float("inf")
            if log_error:
                error = backend.norm(psi - T@W) / backend.norm(psi)
            T = T.reshape(l, i, ru, -1).transpose(1, 2, 3, 0) # (l, i, ru), rd -> l, i, ru, rd -> i, ru, rd, l
            W = W.reshape(W.shape[0], u, r, d) # l, (u, r, d) -> l, u, r, d
            return W, None, T, error
        else:
            raise NotImplementedError(f"mode \"{mode}\" is not a valid mode for the honeycomb yang-baxter move!")

    # 2.) Contract 1-site wavefunction
    psi = backend.tensordot(W, T, ([2], [3])) # l u [r] d; i ru rd [l] -> l u d i ru rd
    psi = psi.transpose(3, 0, 2, 5, 1, 4) # l, u, d, i, ru, rd -> i, l, d, rd, u, ru
    psi /= backend.norm(psi)
    i, l, d, rd, u, ru = psi.shape

    # 3.) Determine the split bond dimensions
    if larger_bond_direction == "down":
        D2, D1 = utility.split_dims(i*l, D_max)
    elif larger_bond_direction == "up":
        D1, D2 = utility.split_dims(i*l, D_max)
    else:
        raise ValueError(f"\"{larger_bond_direction}\" is an invalid larger_bond_direction, must be one of [\"up\", \"down\"].")
    
    # 4.) Reshape wavefunction to prepare for tripartite decomposition subroutine
    psi = backend.reshape(psi, (i*l, d*rd, u*ru)) # i, l, d, rd, u, ru -> (i, l), (d, rd), (u, ru)

    # 5.) Call tripartite decomposition subroutine
    chi = min(chi_max, psi.shape[1]*D1, psi.shape[2]*D2)
    T, W1, W2 = tripartite_decomposition.tripartite_decomposition(psi, D1, D2, chi, debug_logger=debug_logger, **options)

    # 6.) Finalize and return tensors
    T = T.reshape(i, l, D1, D2).transpose(0, 3, 2, 1) # (i, l), D1, D2 -> i, l, D1, D2 = i, l, rd, ru -> i, ru, rd, l
    W1 = W1.reshape(D1, d, rd, chi).transpose(0, 3, 2, 1) # D1, (d, rd), chi -> D1, d, rd, chi = l, d, r, u -> l, u, r, d
    W2 = W2.reshape(D2, chi, u, ru).transpose(0, 2, 3, 1) # D2, chi, (u, ru) -> D2, chi, u, ru = l, d, u, r -> l, u, r, d

    if log_error:
        psi_prime = backend.tensordot(W1, W2, ([1], [3])) # l [u] r d; lp1 up1 rp1 [dp1] -> l r d lp1 up1 rp1
        psi_prime = backend.tensordot(T, psi_prime, ([1, 2], [3, 0])) # i [ru] [rd] l; [l] r d [lp1] up1 rp1 -> i l r d up1 rp1
        psi_prime = psi_prime.transpose(0, 1, 3, 2, 4, 5).reshape(i*l, d*rd, u*ru) # i, l, r, d, up1, rp1 -> i, l, d, r, up1, rp1 = i, l, d, rd, u, ru -> (i, l), (d, rd), (u, ru)
        return W1, W2, T, backend.norm(psi_prime - psi) / backend.norm(psi)
    
    return W1, W2, T, -float("inf")
