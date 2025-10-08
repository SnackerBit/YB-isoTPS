import h5py

from . import backend

"""
This file implements several utility functions that are used throughout the code base
"""

def safe_svd(A, full_matrices=True):
    """
    Computes the Singular Value Decomposition A = U@S@V. If the numpy svd does not converge,
    scipy's SVD with the less efficient but more involved general rectangular approach is used,
    which is more likely to converge.

    Parameters
    ----------
    A : backend.array_type of shape (n, m)
        The matrix that should be decomposed using SVD.
    full_matrices : bool, optional
        determines the shape of the output matrices. See official
        numpy documentation for more details.
    
    Returns
    -------
    U : backend.array_type of shape (n, chi)
        isometric matrix. A = U@backend.diag(S)@V.
    S : backend.array_type of shape (chi, )
        vector containing the real singular values >= 0. A = U@backend.diag(S)@V.
    V : backend.array_type of shape (chi, m)
        V.T is an isometric matrix. A = U@backend.diag(S)@V.
    """
    return backend.safe_svd(A, full_matrices) # TODO: REMOVE THIS FROM UTILITY.PY

def split_and_truncate(A, chi_max=0, eps=0):
    r"""
    Performs an SVD of the matrix A and truncates the singular values to the bond dimension chi_max.

    Parameters
    ----------
    A : backend.array_type of shape (n, m)
        The matrix that should be split using SVD.
    chi_max : int, optional
        The maximum bond dimension to which the result is truncated. If this is set to zero, the algorithm
        acts as if there is no maximum bond dimension. Default: 0.
    eps : float, optional
        All singular values smaller than eps are truncated. Default: 0.

    Returns
    -------
    U : backend.array_type of shape (n, chi)
        isometric matrix. A \approx U@backend.diag(S)@V.
    S : backend.array_type of shape (chi, )
        vector containing the normalized real singular values >= 0. A \approx U@backend.diag(S)@V.
    V : backend.array_type of shape (chi, m)
        V.T is an isometric matrix. A \approx U@backend.diag(S)@V.
    norm : float
        the norm of the unnormalized (but already truncated) singular values, backend.norm(S).
    error : float
        the error of the truncation (sum of the square of all singular values being thrown away).
    """
    # perform SVD
    U, S, V = safe_svd(A, full_matrices=False)
    # truncate
    if chi_max > 0:
        chi_new = min(chi_max, linalg.sum(S >= eps))
    else:
        chi_new = backend.sum(S>=eps)
    assert chi_new >= 1
    piv = backend.argsort(S)[::-1][:chi_new]  # keep the largest chi_new singular values
    error = backend.sum(S[chi_new:]**2)
    if error > 1.e-1:
        print("[WARNING]: larger error detected in SVD, error =", error, "sum of remaining singular values:", backend.sum(S[:chi_new]**2))
    U, S, V = U[:, piv], S[piv], V[piv, :]
    # renormalize
    norm = backend.norm(S)
    if norm < 1.e-7 and norm != 0.0:
        print(f"[WARNING]: Small singular values, norm(S) = {norm}")
    if norm != 0.0:
        S = S / norm
    return U, S, V, norm, error
    
def random_isometry(N, M):
    """
    Returns a random (N, M) complex isometry.

    Parameters
    ----------
    N, M : int
        shape of the isometry
    
    Returns
    -------
    W : backend.array_type of shape (N, M)
        random isometry
    """
    z = (backend.random((N, N)) + 1j*backend.random((N, N)))/backend.sqrt(2)
    q, r = backend.qr(z)
    d = backend.diag(r)
    ph = d/backend.abs(d)
    q = q@backend.diag(ph)@q
    return q[:, :M]

def random_unitary(N):
    """
    Returns a random (N, N) unitary drawn from the Haar measure
    
    Parameters
    ----------
    N: int
        dimension of the unitary
    
    Returns
    -------
    U : backend.array_type of shape (N, N)
        random unitary
    """
    backend.random_unitary(N) # TODO: REMOVE THIS FROM UTILITY.PY

def check_isometry(A):
    r"""
    Cecks if the given matrix A of shape (in, out) is in fact an isometry, ie. if
    A^\dagger A = 1 and if P = A A^\dagger is a projector, ie. P^2 = P.
    Returns True on success and False on failure.

    Parameters
    ----------
    A : backend.array_type of shape (in, out)
        the matrix that is to be checked

    Returns
    -------
    result : bool
        wether A is an isometry or not
    """
    AA_dagger = A@backend.conj(A.T)
    return backend.all(backend.isclose(backend.conj(A.T)@A, backend.eye(A.shape[1]))) and backend.all(backend.isclose(AA_dagger, AA_dagger@AA_dagger))

def flip_W(W):
    """
    Flips the given W tensor along the vertical axis

         u                   u
         |                   |
     l--(W)--r    <->    r--(W)--l
         |                   |
         d                   d

    Parameters
    ----------
    W : backend.array_type with ndim = 4 or None
        the W tensor to be flipped

    Returns
    -------
    W_prime : backend.array_type with ndim = 4 or None
        the flipped W tensor, or None if W == None
    """
    if W is None: 
        return None
    return backend.transpose(W, (2, 1, 0, 3)) # l, u, r, d <-> r, u, l, d

def flip_T_square(T):
    r"""
    Flips the given T tensor (square isoTPS) along the vertical axis

     lu     ru          ru     lu     
      \  p  /            \     /     
       \ | /              \   /    
        (T)       <->      (T)  
       /   \              /   \     
      /     \            /     \    
     ld     rd          rd     ld

    Parameters
    ----------
    T : backend.array_type with ndim = 5 or None
        the T tensor to be flipped

    Returns
    -------
    T_prime : backend.array_type with ndim = 5 or None
        the flipped T tensor, or None if T == None
    """
    if T is None:
        return None
    return backend.transpose(T, (0, 4, 3, 2, 1)) # p, ru, rd, ld, lu <-> p, lu, ld, rd, ru

def flip_T_honeycomb(T):
    r"""
    Flips the given T tensor (honeycomb isoTPS) along the vertical axis

         p  ru           ru  p           lu  p                   
         | /               \ |             \ |                   
     l--(T)       <->       (T)--l   =      (T)--r                    
           \               /               /                     
            rd           rd              ld  

    Parameters
    ----------
    T : backend.array_type with ndim = 4 or None
        the T tensor to be flipped

    Returns
    -------
    T_prime : backend.array_type with ndim = 4 or None
        the flipped T tensor, or None if T == None        
    """
    if T is None:
        return None
    return T.transpose(0, 3, 2, 1) # p, ru, rd, l <-> p, r, ld, lu

def flip_onesite_square(T, W, Wp1):
    """
    Flips all the tensors of the one-site wave function (square isoTPS) along the vertical axis
    """
    return flip_T_square(T), flip_W(W), flip_W(Wp1)

def flip_twosite_square(T1, T2, Wm1, W, Wp1):
    """
    Flips all the tensors of the two-site wave function (square isoTPS) along the vertical axis
    """
    return flip_T_square(T1), flip_T_square(T2), flip_W(Wm1), flip_W(W), flip_W(Wp1)

def flip_onesite_honeycomb(T, W, Wp1):
    """
    Flips all the tensors of the one-site wave function (honeycomb isoTPS) along the vertical axis
    """
    return flip_T_honeycomb(T), flip_W(W), flip_W(Wp1)

def flip_twosite_honeycomb(T1, T2, Wm1, W, Wp1):
    """
    Flips all the tensors of the one-site wave function (honeycomb isoTPS) along the vertical axis
    """
    return flip_T_honeycomb(T1), flip_T_honeycomb(T2), flip_W(Wm1), flip_W(W), flip_W(Wp1)

def flip_twosite_op(op):
    """
    Flips the given two site operator along the vertical axis

       i*    j*                j*    i*             
       |     |                 |     |             
    |-----------|           |-----------|          
    |    op     |    <->    |    op     |
    |-----------|           |-----------|          
       |     |                 |     |          
       i     j                 j     i          

    Parameters
    ----------
    op : backend.array_type with ndim = 4 or None
        the op tensor to be flipped

    Returns
    -------
    op_prime : backend.array_type with ndim = 4 or None
        the flipped op tensor, or None if op == None
    """
    if op is None:
        return None
    return backend.transpose(op, (1, 0, 3, 2)) # i, j, i*, j* -> j, i, j*, i*

def lq(X):
    """
    Performs an LQ decomposition of the given matrix

    Parameters
    ----------
    X: backend.array_type of shape (n, m)
        the matrix of which the LQ decomposition is taken
    
    Returns
    -------
    L: backend.array_type of shape (n, chi)
        L factor of the LQ decomposition.
    Q: backend.array_type of shape (chi, m)
        Q factor of the LQ decomposition. Q.T is an isometry.
    """
    Q, R = backend.qr(X.T)
    return R.T, Q.T

def split_dims(chi, D_max):
    """
    This function tries to find integers D1, D2 > 0 such that |chi - D1*D2| is as small as possible and D1, D2 <= D_max.
    for the best achievable distance |D1*D2 - chi| the function also tries to split D1 and D2 as evenly as possible.
    It holds D1 <= D2.

    Parameters
    ----------
    chi : int
        integer to be split
    D_max : int
        maximal bond dimension

    Returns
    -------
    D1, D2 : int, int:
        best splitting found by the algorithm. It holds D1*D2 <= chi; D1, D2 <= D_max and D1 <= D2.
    """
    if chi >= D_max**2:
        return D_max, D_max
    best_prod = 1
    best_sum = 2
    best_D1 = 1
    best_D2 = 1
    for D1 in range(1, int(backend.floor(backend.sqrt(chi))) + 1):
        D2 = min(chi // D1, D_max)
        if D1 * D2 > best_prod:
            best_prod = D1 * D2
            best_sum = D1 + D2
            best_D1, best_D2 = D1, D2
        elif D1 * D2 == best_prod and D1 + D2 < best_sum:
            best_sum = D1 + D2
            best_D1, best_D2 = D1, D2
    return best_D1, best_D2

def split_matrix_svd(A, chi):
    r"""
    Splits a (n x m) matrix A into a (n x chi) isometry B and a (chi x m) matrix C,
    using Singular Value Decomposition. This function asssumes chi <= max(n, m).
    If chi >= min(n, m), the decomposition is numerically exact and the QR decomposition is used, 
    because it is faster than the SVD in practice.

    Parameters
    ----------
    A : backend.array_type of shape (n, m)
        the matrix to be split.
    chi : int
        split dimension. should be >= 1.

    Returns
    -------
    B : backend.array_type of shape (n, chi)
        first factor of the split. B is an isometry. A \approx B@C.
    C : backend.array_type of shape (chi, m)
        second factor of the split. A \approx B@C.
    """
    assert(chi > 0)
    if chi == min(A.shape[0], A.shape[1]):
        return backend.qr(A)
    elif chi > min(A.shape[0], A.shape[1]):
        Q, R = backend.qr(A)
        Q = Q @ backend.eye(Q.shape[1], chi)
        Q, R2 = backend.qr(Q)
        R = R2 @ backend.eye(chi, R.shape[0]) @ R
        return Q, R
    # Split and truncate A via SVD
    B, S, V = safe_svd(A, full_matrices=False)
    piv = backend.argsort(S)[::-1][:chi]
    B, S, V = B[:, piv], S[piv], V[piv, :]
    # Renormalize
    S /= backend.norm(S)
    # Isometrize B
    B, R = backend.qr(B)
    # Absorb R and S into V to form C
    C = R @ backend.diag(S) @ V
    return B, C

def split_matrix_iterate_QR(A, chi, N_iters, eps=1e-9, C0=None, smart_initial_condition=True, normalize=True, log_iterates=False):
    r"""
    Splits a (n x m) matrix A into a (n x chi) isometry B and a (chi x m) matrix C,
    using N_iters iterations of a sweeping algorithm using only QR decompositions and matrix products.
    Per iteration 2 QR decompositions and 2 matrix products are computed.
    This function asssumes chi <= max(n, m). If chi >= min(n, m), a single QR decomposition suffices
    to compute the numerically exact solution.

    Parameters
    ----------
    A : backend.array_type of shape (n, m)
        the matrix to be split.
    chi : int
        split dimension. should be >= 1.
    N_iters : int
        maximum number of iterations
    eps : float, optional
        if the relative decrease of the error after one iteration is smaller than eps,
        the algorithm terminates. Default value: 1e-9.
    C0 : backend.array_type or None, optional
        Initialization for the C matrix. If multiple splits are executed on similar matrices,
        the result of a previous split can be a very good initialization for the next split.
        Default value: None
    smart_initial_condition: bool, optional
        Determines the initialization of C, if C0 is None.
        If this is set to True, the C matrix is initialized by a reordered slicing of A.
        If this is set to False, the C matrix is initialized with identity.
        Default value: True
    normalize : bool, optional
        Determines wether the C matrix is to be normalized. Default: True
    log_iterates : bool, optional
        If this is set to True, the iterates are stored in a list and returned. Default: False.

    Returns
    -------
    B : backend.array_type of shape (n, chi)
        first factor of the split. B is an isometry. A \approx B@C.
    C : backend.array_type of shape (chi, m)
        second factor of the split. A \approx B@C.
    num_iters : int
        the number of iterations used
    iterates : List of (backend.array_type, backend.array_type) or None
        List of iterates (B_i, C_i). If log_iterates is set to False,
        None is returned instead.
    """
    assert(chi > 0)
    iterates = None
    if log_iterates:
        iterates = []
    if chi is None or chi == min(A.shape[0], A.shape[1]):
        Q, R = backend.qr(A)
        Q, R = backend.ascontiguousarray(Q), backend.ascontiguousarray(R)
        if log_iterates:
            iterates.append((Q, R))
        return Q, R, 0, iterates
    elif chi > min(A.shape[0], A.shape[1]):
        Q, R = backend.qr(A)
        Q = backend.ascontiguousarray(Q) @ backend.eye(Q.shape[1], chi, dtype=Q.dtype)
        Q, R2 = backend.qr(Q)
        R = backend.ascontiguousarray(R2) @ backend.eye(chi, R.shape[0], dtype=R.dtype) @ R
        Q, R = backend.ascontiguousarray(Q), backend.ascontiguousarray(R)
        if log_iterates:
            iterates.append((Q, R))
        return Q, R, 0, iterates
    assert(N_iters > 0)
    if C0 is not None:
        C = C0
    elif smart_initial_condition:
        # find the chi largest rows
        temp = backend.sum(backend.abs(A), 1)
        piv = backend.argsort(temp)[::-1][:chi]
        # slice A matrix
        C = A[piv, :]
    else:
        # Initialize C with identity
        C = backend.eye(chi, A.shape[1], dtype=A.dtype)
    error = None
    for n in range(N_iters):
        # Isometrize C
        C, _ = backend.qr(C.T)
        # Compute B
        B = backend.dot(A, backend.conj(C))
        # isometrize B
        B, _ = backend.qr(B)
        # Compute C
        C = backend.dot(backend.conj(B).T, A)
        # Store iterates
        if log_iterates:
            iterates.append((B, C))
        # Check if we are done
        error_new = backend.norm(A - B@C)
        if error is not None and (backend.isclose(error_new, 0) or backend.abs((error - error_new)/error) < eps):
            break
        error = error_new
    if normalize:
        return B, C / backend.norm(C), n + 1, iterates
    else:
        return B, C, n + 1, iterates

def split_matrix(A, chi, mode, N_iters=None):
    r"""
    Splits a (n x m) matrix A into a (n x chi) isometry B and a (chi x m) matrix C.
    Depending on the selected mode, either the function split_matrix_svd() or
    split_matrix_iterate_QR() is called for the splitting.

    Parameters
    ----------
    A : backend.array_type of shape (n, m)
        the matrix to be split.
    chi : int
        split dimension. should be >= 1.
    mode : str, one of {"svd", "iterate"}
        used for selecting the splitting mode.
    N_iters : int or None, optional
        maximum number of iterations. Only used when mode == "iterate". Default: None.

    Returns
    -------
    B : backend.array_type of shape (n, chi)
        first factor of the split. B is an isometry. A \approx B@C.
    C : backend.array_type of shape (chi, m)
        second factor of the split. A \approx B@C.
    """
    if mode == "svd":
        return split_matrix_svd(A, chi)
    elif mode == "iterate":
        B, C, _, _ = split_matrix_iterate_QR(A, chi, N_iters)
        return B, C
    else:
        raise NotImplementedError(f"split_matrix is not implemted for mode = {mode}")

def isometrize_polar(A):
    """
    Finds the isometry B, which has the same shape as A and minimizes the distance |A - B|, using a polar decomposition.
    It is assumed that A is an (n, m) matrix with n >= m. The polar decomposition is implemented using an SVD.

    Parameters
    ----------
    A : backend.array_type of shape (n, m)
        real or complex matrix

    Returns
    -------
    B : backend.array_type of shape (n, m)
        isometry closest to A
    """
    U, _, V = safe_svd(A, full_matrices=False)
    return U@V

def calc_U_bonds(H_bonds, dt):
    """
    Given the Hamiltonian H as a list of two-site operators,
    calculate expm(-dt*H). Note that no imaginary 'i' is included,
    thus real 'dt' means imaginary time evolution!

    Parameters
    ----------
    H_bonds : List of backend.array_type with ndim = 4
        list of two-site operators making up the Hamiltonian.
    dt : complex
        real or imaginary time. Note that real dt means imaginary time evolution.

    Returns
    -------
    U_bonds : List of backend.array_type with ndim = 4
        list of two-site real or imaginary time evolution operators
    """
    U_bonds = []
    d = H_bonds[0].shape[0]
    for H in H_bonds:
        if H is None:
            U_bonds.append(None)
        else:
            H = backend.reshape(H, (d*d, d*d))
            U = backend.expm(-dt*H)
            U_bonds.append(backend.reshape(U, (d, d, d, d)))
    return U_bonds

def compute_op_list(L, op):
    """
    Given a single site operator, computes a list of operators
    eye x eye x ... x eye x op x eye x ... x eye,
    where the ith entry of the list puts the operator on site i

    Parameters
    ----------
    L : int
        number of sites in the chain
    op : backend.sparse_array_type of size (d, d)
        single-site operator

    Returns
    -------
    result : List of backend.array_type of size (d^L, d^L)
        list of single site operators acting on the full Hilbert space
    """
    result_list = []
    eye = backend.eye(*op.shape)
    for j in range(L):
        result = eye
        if j == 0:
            result = op
        else:
            for i in range(1, j):
                result = backend.sparse_kron(result, eye)
            result = backend.sparse_kron(result, op)
        for i in range(j+1, L):
            result = backend.sparse_kron(result, eye)
        result_list.append(result)
    return result_list

def average_site_expectation_value(L, psi, op):
    """
    Given a wave function psi, the number of sites L, and an operator op,
    this computes the average site expectation value of op:
    1/L sum_{i=0}^{i=L-1} <psi| eye x eye x ... x eye x op_i x eye x ... x eye |psi>

    Parameters
    ----------
    L : int
        number of sites
    psi : backend.array_type of size (L^d, )
        vector representing the wave function
    op : backend.array_type of size (d, d)
        single-site operator

    Returns
    -------
    result : float
        average site expectation value
    """
    op_list = compute_op_list(L, op)
    result = 0.
    for op in op_list:
        result += (psi.T@op@psi)/(psi.T@psi)
    return result / L

def append_to_dict_list(d, key, value):
    """
    appends an object to the list d[key], if the key is already in the dictionary,
    else created a new list d[key] = [value].

    Parameters
    ----------
    d : Dict
        dictionary
    key : str
        key into the dictionary. if key in d, d[key] must be a list.
    value : Any
        value that is appended to the list d[key]. If value itself is a list, it is
        not appended but concatenated to d[key].
    """
    if key in d:
        d[key] = list(d[key])
        if type(value) == list:
            d[key] += value
        else:
            d[key].append(value)
    else:
        if type(value) == list:
            d[key] = value
        else:
            d[key] = [value]

def write_or_change_field_hf(hf, name, value, overwrite=True):
    """
    Helper function for writing into an h5 file
    """
    try:
        if name in hf:
            if overwrite:
                hf[name][...] = value
        else:
            hf[name] = value
    except Exception as e:
        print(f"Encountered exception while trying to write {name} to h5 file: {e}", flush=True)

def dump_dict_into_hf(hf, d):
    """
    dumps a dictionary into a hdf5 file.

    Parameters
    ----------
    hf : h5py.File
        the hdf5 file
    d : dict
        the dictionary
    """
    allowed_types = [int, float, complex, bool, str, backend.array_type, list]
    def _is_allowed_type(value):
        for allowed_type in allowed_types:
            if isinstance(value, allowed_type):
                return True
        return False
    for key, value in d.items():
        if _is_allowed_type(value):
            write_or_change_field_hf(hf, key, value, overwrite=True)
        elif isinstance(value, dict):
            dump_dict_into_hf(hf.create_group(key), d[key])

def load_dict_from_hf(hf):
    """
    Loads a python dictionary from a hdf5 file.

    Parameters
    ----------
    hf : h5py.File
        the hdf5 file

    Returns
    -------
    result : dict
        the resulting dictionary
    """
    result = {}
    for key, value in hf.items():
        if isinstance(value, h5py.Group):
            result[key] = load_dict_from_hf(hf[key])
        else:
            result[key] = hf[key][()]
            if isinstance(result[key], bytes):
                result[key] = result[key].decode('utf8')
            if isinstance(result[key], str) and result[key] == 'null\n...\n':
                result[key] = None
    return result

def turn_lists_to_dicts(d):
    """
    Recursively loops through the contents of the given dictionary, turning all lists [element_1, element_2, ..., element_n] 
    into dictionaries {"1": element_1, "2": element_2, ..., "n": element_n} if element_1, element_2, ..., element_n are lists
    or dicts themselves. This is used for storing lists of elements in h5-files.

    Parameters
    ----------
    d : dict
        the dictionary that should be processed.
    """
    for key in d:
        if type(d[key]) is list:
            # Check if the elements are all of numerical type
            is_dict_or_list = False
            for element in d[key]:
                if type(element) == dict or type(element) == list:
                    is_dict_or_list = True
                    break
            if is_dict_or_list:
                # Turn list into dict
                temp_dict = {}
                for i, item in enumerate(d[key]):
                    temp_dict[str(i)] = item
                d[key] = temp_dict
        if type(d[key]) is dict:
            turn_lists_to_dicts(d[key])

def turn_dicts_to_lists(d):
    """
    Recursively loops through the contents of the given dictionary, turning all dictionaries of the form
    {"1": element_1, "2": element_2, ..., "n": element_n} back to lists lement_1, element_2, ..., element_n].
    This is used for recovering the original structure of the debug_log when loading from file.
    
    Parameters
    ----------
    d : dict
        the dictionary that should be processed.
    """
    # Go through all keys in d
    for key in d:
        if type(d[key]) is dict:
            # Check if d[key] has the expected list format
            i = 0
            while True:
                if str(i) in d[key]:
                    i += 1
                else:
                    i -= 1
                    break
            if i == -1 or len(d[key]) != i+1:
                # The dictionary is not in the expected format. Recursively go down if the element is a dictionary
                turn_dicts_to_lists(d[key])
            else:
                # The dictionary is in the expected format. Turn into a list!
                temp_list = []
                for j in range(i+1):
                    temp_list.append(d[key][str(j)])
                d[key] = temp_list
                # Go through the list and recursively call turn_dicts_to_lists if the elements are dictionaries themselves
                for j in range(i+1):
                    if type(d[key][j]) is dict:
                        turn_dicts_to_lists(d[key][j])
