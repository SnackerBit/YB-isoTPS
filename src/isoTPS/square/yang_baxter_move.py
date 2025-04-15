import numpy as np
from ...utility import utility
from ...utility.tripartite_decomposition import tripartite_decomposition
from ...utility import debug_logging

def yang_baxter_move(W1, W2, T, D_max, chi_max, larger_bond_direction="down", options={"mode": "svd"}, debug_logger=debug_logging.DebugLogger()):
    r"""
    Performs the Yang baxter move

    \   |           /               \           |   /        
     \  |          /                 \          |  /                       
      \ |         /                   \         | /                     
       (W2)      /                     \      (W2')                    
        | \  |  /                       \  |  / |                    
        |  \ | /                         \ | /  |                          
        |   \|/                           \|/   |                            
        |    T             ->              T    ^                      
        |   / \                           / \   |                        
        |  /   \                         /   \  |                          
        | /     \                       /     \ |                            
       (W1)      \                     /      (W1')                           
      / |         \                   /         | \                         
     /  |          \                 /          |  \                    
    /   |           \               /           |   \                  
    
    Parameters
    ----------
    W1: np.ndarray of shape (l, u, r, d)
        either isometric tensor part of the ortho surface, or orthogonality center.
        Either W1 or W2 can also be None, but not both.  
    W2: np.ndarray of shape (lp1, up1, rp1, dp1)
        either isometric tensor part of the ortho surface, or orthogonality center
        Either W1 or W2 can also be None, but not both.
    T: np.ndarray of shape (i, ru, rd, ld, lu)
        isometric site tensor
    D_max: int 
        maximal bond dimension of T tensors
    chi_max: int 
        maximal bond dimension of the orthogonality surface
    larger_bond_direction: str, one of {"up", "down"}, optional
        selects in which direction the arrow points after the YB move. when choosing "up", the W1' tensor is
        the new isometry center, else it is the W2' tensor. Default: "down".
    options: dictionary, optional
        dictionary specifying which algorithm should be used, and options for these algorithms.
        Options are passed as keyword arguments to the tripartite decomposition subroutine, see
        "src/utility/tripartite_decomposition/tripartite_decomposition.py" for more information.
        Default value: {"mode" : "svd"}.
    debug_logger : DebugLogger instance, optional
        DebugLogger instance managing debug logging. See 'src/utility/debug_logging.py' for more details.
        
    Returns
    -------
    W1': np.ndarray of shape (l, u, r, d)
        updated isometric tensor or oorthogonality center (depending on the value of arrows_should_point_up), part of the ortho surface
    W2': np.ndarray of shape (lp1, up1, rp1, dp1)
        updated isometric tensor or oorthogonality center (depending on the value of arrows_should_point_up), part of the ortho surface
    T': np.ndarray of shape (i, ru, rd, ld, lu)
        updated isometric site tensor
    error: float
        If debug_logger.log_yb_move_errors is True or debug_logger.log_approximate_column_error_yb is True, the relative error 
        norm(contr_before - contr_after) / norm(contr_before) is returned. Else, -float("inf") is returned.
    """
    log_error = debug_logger.log_yb_move_errors or debug_logger.log_approximate_column_error_yb
    # 1.) Handle edge cases
    if W1 is None:
        # Contract everything together
        contr = np.tensordot(T, W2, ([4], [2])) # i ru rd ld [lu]; l2 u2 [r2] d2 -> i ru rd ld l2 u2 d2
        contr = np.transpose(contr, (0, 2, 3, 4, 5, 1, 6)) # i, ru, rd, ld, l2, u2, d2 -> i, rd, ld, l2, u2, ru, d2
        i, rd, ld, l2, u2, ru, d2 = contr.shape
        contr = np.reshape(contr, (i*rd*ld*l2, u2*ru*d2)) # i, rd, ld, l2, u2, ru, d2 -> (i, rd, ld, l2), (u2, ru, d2)
        # Call subroutine to seperate into T and W
        T, W2 = utility.split_matrix_svd(contr, min(D_max, contr.shape[0]))
        T = np.reshape(T, (i, rd, ld, l2, T.shape[-1])) # (i, rd, ld, l2), D -> i, rd, ld, l2, D = i, rd, ld, lu, ru
        T = np.transpose(T, (0, 4, 1, 2, 3)) # i, rd, ld, lu, ru -> i, ru, rd, ld, lu
        W2 = np.reshape(W2, (W2.shape[0], u2, ru, d2)) # D, (u2, ru, d2) ->  D, u2 ru, d2 =  l2, u2, r2, d2
        if log_error:
            contr2 = np.tensordot(T, W2, ([1], [0])) # i [ru] rd ld lu; [l2] u2 r2 d2 -> i rd ld lu u2 r2 d2 = i rd ld l2 u2 ru d2
            contr2 = np.reshape(contr2, (i*rd*ld*l2, u2*ru*d2)) # i, rd, ld, l2, u2, ru, d2 -> (i, rd, ld, l2), (u2, ru, d2)
            return None, W2, T, np.linalg.norm(contr2 - contr) / np.linalg.norm(contr)
        return None, W2, T, -float("inf")
    elif W2 is None:
        # Contract everything together
        contr = np.tensordot(T, W1, ([3], [2])) # i ru rd [ld] lu; l1 u1 [r1] d1 -> i ru rd lu l1 u1 d1
        contr = np.transpose(contr, (0, 1, 4, 3, 5, 2, 6)) # i, ru, rd, lu, l1, u1, d1, -> i, ru, l1, lu, u1, rd, d1
        i, ru, l1, lu, u1, rd, d1 = contr.shape
        contr = np.reshape(contr, (i*ru*l1*lu, u1*rd*d1)) # i, ru, l1, lu, u1, rd, d1 -> (i, ru, l1, lu), (u1, rd, d1)
        # Call subroutine to seperate into T and W
        T, W1 = utility.split_matrix_svd(contr, min(D_max, contr.shape[0]))
        T = np.reshape(T, (i, ru, l1, lu, T.shape[-1])) # (i, ru, l1, lu), D -> i, ru, l1, lu, D = i, ru, ld, lu, rd
        T = np.transpose(T, (0, 1, 4, 2, 3)) # i, ru, ld, lu, rd -> i, ru, rd, ld, lu
        W1 = np.reshape(W1, (W1.shape[0], u1, rd, d1)) # D, (u1, rd, d1) -> D, u1, rd, d1 -> l1, u1, r1, d1
        if log_error:
            contr2 = np.tensordot(T, W1, ([2], [0])) # i ru [rd] ld lu; [l1] u1 r1 d1 -> i ru ld lu u1 r1 d1 = i ru l1 lu u1 rd d1
            contr2 = np.reshape(contr2, (i*ru*l1*lu, u1*rd*d1)) # i, ru, l1, lu, u1, rd, d1 -> (i, ru, l1, lu), (u1, rd, d1)
            return W1, None, T, np.linalg.norm(contr2 - contr) / np.linalg.norm(contr)
        return W1, None, T, -float("inf")

    # 2.) Contract 1-site wavefunction
    psi = np.tensordot(W1, W2, ([1], [3])) # l [u] r d; lp1 up1 rp1 [dp1] -> l r d lp1 up1 rp1; complexity: O(chi^3 D^4)
    psi = np.tensordot(psi, T, ([1, 5], [3, 4])) # l [r] d lp1 up1 [rp1]; i ru rd [ld] [lu] -> l d lp1 up1 i ru rd; comlpexity: O(chi^2 D^6 p)
    psi = np.transpose(psi, (0, 2, 4, 1, 6, 3, 5)) # l, d, lp1, up1, i, ru, rd -> l, lp1, i, d, rd, up1, ru
    psi /= np.linalg.norm(psi)
    l, lp1, i, d, rd, up1, ru = psi.shape
    
    # 3.) Determine the split bond dimensions
    if larger_bond_direction == "down":
        D2, D1 = utility.split_dims(l*lp1*i, D_max)
    elif larger_bond_direction == "up":
        D1, D2 = utility.split_dims(l*lp1*i, D_max)
    else:
        raise ValueError(f"\"{larger_bond_direction}\" is an invalid larger_bond_direction, must be one of [\"up\", \"down\"].")
        
    # 4.) Reshape wavefunction to prepare for tripartite decomposition subroutine
    psi = np.reshape(psi, (l*lp1*i, d*rd, up1*ru)) # l, lp1, i, d, rd, up1, ru -> (l, lp1, i), (d, rd), (up1, ru)

    # 5.) Call tripartite decomposition subroutine
    chi = min(chi_max, psi.shape[1]*D1, psi.shape[2]*D2)
    T, W1_prime, W2_prime = tripartite_decomposition.tripartite_decomposition(psi, D1, D2, chi, debug_logger=debug_logger, **options)

    # 6.) Finalize and return tensors
    T = np.reshape(T, (l, lp1, i, D1, D2)) # (l, lp1, i), D1, D2 -> l, lp1, i, D1, D2 = ld, lu, p, rd, ru
    T = np.transpose(T, (2, 4, 3, 0, 1)) # ld, lu, p, rd, ru -> p, ru, rd, ld, lu
    W1_prime = np.reshape(W1_prime, (D1, d, rd, W1_prime.shape[-1])) # D1, (d, rd), chi -> D1, d, rd, chi = lm1, dm1, rm1, um1
    W1_prime = np.transpose(W1_prime, (0, 3, 2, 1)) # lm1, dm1, rm1, um1 -> lm1, um1, rm1, dm1
    W2_prime = np.reshape(W2_prime, (D2, W2_prime.shape[1], up1, ru)) # D2, chi, (up1, ru) -> D2, chi, up1, ru = l, d, u, r
    W2_prime = np.transpose(W2_prime, (0, 2, 3, 1)) # l, d, u, r -> l, u, r, d
    W2_prime /= np.linalg.norm(W2_prime)

    if log_error:
        psi2 = np.tensordot(W1_prime, W2_prime, ([1], [3])) # l [u] r d; lp1 up1 rp1 [dp1] -> l r d lp1 up1 rp1; complexity: O(chi^3 D^4)
        psi2 = np.tensordot(T, psi2, ([1, 2], [3, 0])) # i [ru] [rd] ld lu; [l] r d [lp1] up1 rp1 -> i ld lu r d up1 rp1; comlpexity: O(chi^2 D^6 p)
        psi2 = np.transpose(psi2, (1, 2, 0, 4, 3, 5, 6)) # i, ld, lu, r, d, up1, rp1 -> ld, lu, i, d, r, up1, rp1 = l, lp1, i, d, rd, up1, ru
        psi2 = np.reshape(psi2, (l*lp1*i, d*rd, up1*ru)) # l, lp1, i, d, rd, up1, ru -> (l, lp1, i), (d, rd), (up1, ru)
        return W1_prime, W2_prime, T, np.linalg.norm(psi2 - psi) / np.linalg.norm(psi)
    
    return W1_prime, W2_prime, T, -float("inf")