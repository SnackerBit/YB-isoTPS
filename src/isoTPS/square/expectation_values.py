from ...utility import backend

"""
This file implements the computation of norms and expectation values for one-site and two-site environments for the square isoTPS.
"""

def compute_norm_onesite(T, W, Wp1):
    r"""
    Computes the norm of the one-site wave function

          | /  
          |/    
         Wp1
     \   /|
      \ / |
     --T  |
      / \ |
     /   \|
          W
          |\ 
          | \ 

    by contracting all tensors. The orthogonality center must be 
    either at W or Wp1. Wp1 or W can also be None.

    Parameters
    ----------
    T : backend.array_type of shape (i, ru, rd, ld, lu)
        part of the 1-site wavefunction
    W : backend.array_type of shape (l, u, r, d) or None
        part of the 1-site wavefunction
    Wp1 : backend.array_type of shape (lp1, up1, rp1, dp1) or None
        part of the 1-site wavefunction

    Returns
    -------
    norm : float
    """
    if W is None:
        contr = backend.tensordot(Wp1, backend.conj(Wp1), ([1, 2, 3], [1, 2, 3])) # l [u] [r] [d]; l* [u*] [r*] [d*] -> l l*
        contr = backend.tensordot(T, contr, ([1], [0])) # p [ru] rd ld lu; [l] l* -> p rd ld lu l*
        contr = backend.tensordot(contr, backend.conj(T), ([0, 1, 2, 3, 4], [0, 2, 3, 4, 1])) # [p] [rd] [ld] [lu] [l*]; [p*] [ru*] [rd*] [ld*] [lu*]
    elif Wp1 is None:
        contr = backend.tensordot(W, backend.conj(W), ([1, 2, 3], [1, 2, 3])) # lm1 [um1] [rm1] [dm1]; lm1* [um1*] [rm1*] [dm1*] -> lm1 lm1*
        contr = backend.tensordot(T, contr, ([2], [0])) # p ru [rd] ld lu; [lm1] lm1* -> p ru ld lu lm1*
        contr = backend.tensordot(contr, backend.conj(T), ([0, 1, 2, 3, 4], [0, 1, 3, 4, 2])) # [p] [ru] [ld] [lu] [lm1*]; [p*] [ru*] [rd*] [ld*] [lu*]
    else:
        contr = backend.tensordot(W, backend.conj(W), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1* 
        temp = backend.tensordot(Wp1, backend.conj(Wp1), ([1, 2], [1, 2])) # l [u] [r]] d; l* [u*] [r*] d* -> l d l* d*
        contr = backend.tensordot(contr, temp, ([1, 3], [1, 3])) # lm1 [um1] lm1* [um1*]; l [d] l* [d*] -> lm1 lm1* l l*
        temp = backend.tensordot(T, backend.conj(T), ([0, 3, 4], [0, 3, 4])) # [p] ru rd [ld] [lu]; [p*] ru* rd* [ld*] [lu*] -> ru rd ru* rd*
        contr = backend.tensordot(temp, contr, ([0, 1, 2, 3], [2, 0, 3, 1])) # [ru] [rd] [ru*] [rd*]; [lm1] [lm1*] [l] [l*]
    return contr.item()

def compute_norm_twosite(T1, T2, Wm1, W, Wp1):
    r"""
    Computes the norm of the two-site wave function

        \ | 
         \|
         Wp1
          |\    /
          | \  /
          |  T2--
          | /  \ 
          |/    \ 
          W
    \    /|
     \  / |
    --T1  |
     /  \ |  
    /    \|
         Wm1 
          |\ 
          | \ 

    by contracting all tensors.
    The orthogonality center must be either at Wm1, W, or Wp1.
    Wm1 and/or Wm1 can also be None.

    Parameters
    ----------
    T1 : backend.array_type of shape (i, ru1, rd1, ld1, lu1)
        part of the twosite wavefunction
    T2 : backend.array_type of shape (j, ru2, rd2, ld2, lu2)
        part of the twosite wavefunction
    Wm1 : backend.array_type of shape (lm1, um1, rm1, dm1) or None
        part of the twosite wavefunction
    W : backend.array_type of shape (l, u, r, d)
        part of the twosite wavefunction
    Wp1 : backend.array_type of shape (lp1, up1, rp1, dp1) or None
        part of the twosite wavefunction

    Returns
    -------
    norm : float
    """
    if Wm1 is None:
        if Wp1 is None:
            contr = backend.tensordot(T1, backend.conj(T1), ([0, 2, 3, 4], [0, 2, 3, 4])) # [p1] ru1 [rd1] [ld1] [lu1]; [p1*] ru1* [rd1*] [ld1*] [lu1*] -> ru1 ru1*
            temp = backend.tensordot(W, backend.conj(W), ([1, 3], [1, 3])) # l [u] r [d]; l* [u*] r* [d*] -> l r l* r*
            contr = backend.tensordot(contr, temp, ([0, 1], [0, 2])) # [ru1] [ru1*]; [l] r [l*] r* -> r r*
            temp = backend.tensordot(T2, backend.conj(T2), ([0, 1, 2, 4], [0, 1, 2, 4])) # [p2] [ru2] [rd2] ld2 [lu2]; [p2*] [ru2*] [rd2*] ld2* [lu2*] -> ld2 ld2*
            contr = backend.tensordot(contr, temp, ([0, 1], [0, 1])) # [r] [r*]; [ld2] [ld2*]
        else:
            contr = backend.tensordot(Wp1, backend.conj(Wp1), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
            temp = backend.tensordot(T2, backend.conj(T2), ([0, 1, 2], [0, 1, 2])) # [p2] [ru2] [rd2] ld2 lu2; [p2*] [ru2*] [rd2*] ld2* lu2* -> ld2 lu2 ld2* lu2*
            contr = backend.tensordot(temp, contr, ([1, 3], [0, 2])) # ld2 [lu2] ld2* [lu2*]; [rp1] dp1 [rp1*] dp1* -> ld2 ld2* dp1 dp1*
            contr = backend.tensordot(contr, W, ([0, 2], [2, 1])) # [ld2] ld2* [dp1] dp1*; l [u] [r] d -> ld2* dp1* l d
            contr = backend.tensordot(contr, backend.conj(W), ([0, 1, 3], [2, 1, 3])) # [ld2*] [dp1*] l [d]; l* [u*] [r*] [d*] -> l l*
            temp = backend.tensordot(T1, backend.conj(T1), ([0, 2, 3, 4], [0, 2, 3, 4])) # [p1] ru1 [rd1] [ld1] [lu1]; [p1*] ru1* [rd1*] [ld1*] [lu1*] -> ru1 ru1*
            contr = backend.tensordot(temp, contr, ([0, 1], [0, 1])) # [ru1] [ru1*]; [l] [l*]
    elif Wp1 is None:
        contr = backend.tensordot(Wm1, backend.conj(Wm1), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
        temp = backend.tensordot(T1, backend.conj(T1), ([0, 3, 4], [0, 3, 4])) # [p1] ru1 rd1 [ld1] [lu1]; [p1*] ru1* rd1* [ld1*] [lu1*] -> ru1 rd1 ru1* rd1*
        contr = backend.tensordot(temp, contr, ([1, 3], [0, 2])) # ru1 [rd1] ru1* [rd1*]; [lm1] um1 [lm1*] um1*; -> ru1 ru1* um1 um1* 
        contr = backend.tensordot(contr, W, ([0, 2], [0, 3])) # [ru1] ru1* [um1] um1*; [l] u r [d] -> ru1* um1* u r
        contr = backend.tensordot(contr, backend.conj(W), ([0, 1, 2], [0, 3, 1])) # [ru1*] [um1*] [u] r; [l*] [u*] r* [d*] -> r r*
        temp = backend.tensordot(T2, backend.conj(T2), ([0, 1, 2, 4], [0, 1, 2, 4])) # [p2] [ru2] [rd2] ld2 [lu2]; [p2*] [ru2*] [rd2*] ld2* [lu2*] -> ld2 ld2*
        contr = backend.tensordot(contr, temp, ([0, 1], [0, 1])) # [r] [r*]; [ld2] [ld2*]
    else:
        contr_r = backend.tensordot(Wp1, backend.conj(Wp1), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
        temp = backend.tensordot(T2, backend.conj(T2), ([0, 1, 2], [0, 1, 2])) # [p2] [ru2] [rd2] ld2 lu2; [p2*] [ru2*] [rd2*] ld2* lu2* -> ld2 lu2 ld2* lu2*
        contr_r = backend.tensordot(temp, contr_r, ([1, 3], [0, 2])) # ld2 [lu2] ld2* [lu2*]; [rp1] dp1 [rp1*] dp1* -> ld2 ld2* dp1 dp1*
        contr_r = backend.tensordot(contr_r, W, ([0, 2], [2, 1])) # [ld2] ld2* [dp1] dp1*; l [u] [r] d -> ld2* dp1* l d
        contr_r = backend.tensordot(contr_r, backend.conj(W), ([0, 1], [2, 1])) # [ld2*] [dp1*] l d; l* [u*] [r*] d* -> l d l* d*
        contr_l = backend.tensordot(Wm1, backend.conj(Wm1), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
        temp = backend.tensordot(T1, backend.conj(T1), ([0, 3, 4], [0, 3, 4])) # [p1] ru1 rd1 [ld1] [lu1]; [p1*] ru1* rd1* [ld1*] [lu1*] -> ru1 rd1 ru1* rd1*
        contr_l = backend.tensordot(temp, contr_l, ([1, 3], [0, 2])) # ru1 [rd1] ru1* [rd1*]; [lm1] um1 [lm1*] um1*; -> ru1 ru1* um1 um1*
        contr = backend.tensordot(contr_l, contr_r, ([0, 1, 2, 3], [0, 2, 1, 3])) # [ru1] [ru1*] [um1] [um1*]; [l] [d] [l*] [d*]
    return contr.item()

def expectation_value_onesite(T, Wm1, W, op):
    r"""
    Computes the expectation value of the given one-site
    operator on the wave function

          | /  
          |/    
          W
     \   /|
      \ / |
     --T  |
      / \ |
     /   \|
          Wm1
          |\ 
          | \ 

    by contracting all tensors.
    The orthogonality center must be either at W or Wp1.
    Wp1 and/or W can also be None.

    Parameters
    ----------
    T : backend.array_type of shape (i, ru, rd, ld, lu)
        part of the 1-site wavefunction
    W : backend.array_type of shape (l, u, r, d) or None
        part of the 1-site wavefunction
    Wp1 : backend.array_type of shape (lp1, up1, rp1, dp1) or None
        part of the 1-site wavefunction
    op : backend.array_type of shape (i, i*)
        1-site operator

    Returns
    -------
    result : complex
        the resulting expectation value
    """
    if Wm1 is None:
        contr = backend.tensordot(W, backend.conj(W), ([1, 2, 3], [1, 2, 3])) # l [u] [r] [d]; l* [u*] [r*] [d*] -> l l*
        contr = backend.tensordot(T, contr, ([1], [0])) # p [ru] rd ld lu; [l] l* -> p rd ld lu l*
        contr = backend.tensordot(contr, backend.conj(T), ([1, 2, 3, 4], [2, 3, 4, 1])) # p [rd] [ld] [lu] [l*]; p* [ru*] [rd*] [ld*] [lu*] -> p p*
    elif W is None:
        contr = backend.tensordot(Wm1, backend.conj(Wm1), ([1, 2, 3], [1, 2, 3])) # lm1 [um1] [rm1] [dm1]; lm1* [um1*] [rm1*] [dm1*] -> lm1 lm1*
        contr = backend.tensordot(T, contr, ([2], [0])) # p ru [rd] ld lu; [lm1] lm1* -> p ru ld lu lm1*
        contr = backend.tensordot(contr, backend.conj(T), ([1, 2, 3, 4], [1, 3, 4, 2])) # p [ru] [ld] [lu] [lm1*]; p* [ru*] [rd*] [ld*] [lu*] -> p p*
    else:
        contr = backend.tensordot(Wm1, backend.conj(Wm1), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1* 
        temp = backend.tensordot(W, backend.conj(W), ([1, 2], [1, 2])) # l [u] [r]] d; l* [u*] [r*] d* -> l d l* d*
        contr = backend.tensordot(contr, temp, ([1, 3], [1, 3])) # lm1 [um1] lm1* [um1*]; l [d] l* [d*] -> lm1 lm1* l l*
        temp = backend.tensordot(T, backend.conj(T), ([3, 4], [3, 4])) # p ru rd [ld] [lu]; p* ru* rd* [ld*] [lu*] -> p ru rd p* ru* rd*
        contr = backend.tensordot(temp, contr, ([1, 2, 4, 5], [2, 0, 3, 1])) # p [ru] [rd] p* [ru*] [rd*]; [lm1] [lm1*] [l] [l*] -> p p*
    return backend.tensordot(contr, op, ([0, 1], [1, 0])) / backend.trace(contr) # [p] [p*]; [i] [i*]

def expectation_value_twosite(T1, T2, Wm1, W, Wp1, op):
    r"""
    Computes the expectation value of the given two-site
    operator on the wave function

        \ | 
         \|
         Wp1
          |\    /
          | \  /
          |  T2--
          | /  \ 
          |/    \ 
          W
    \    /|
     \  / |
    --T1  |
     /  \ |  
    /    \|
         Wm1 
          |\ 
          | \ 

    by contracting all tensors.
    The orthogonality center must be either at Wm1, W, or Wp1.
    Wm1 and/or Wm1 can also be None.

    Parameters
    ----------
    T1 : backend.array_type of shape (i, ru1, rd1, ld1, lu1)
        part of the twosite wavefunction
    T2 : backend.array_type of shape (j, ru2, rd2, ld2, lu2)
        part of the twosite wavefunction
    Wm1 : backend.array_type of shape (lm1, um1, rm1, dm1) or None
        part of the twosite wavefunction
    W : backend.array_type of shape (l, u, r, d)
        part of the twosite wavefunction
    Wp1 : backend.array_type of shape (lp1, up1, rp1, dp1) or None
        part of the twosite wavefunction
    op : backend.array_type of shape (i, j, i*, j*)
        twosite operator

    Returns
    -------
    result : complex
        the resulting expectation value
    """
    if Wm1 is None:
        if Wp1 is None:
            contr = backend.tensordot(T1, backend.conj(T1), ([2, 3, 4], [2, 3, 4])) # p1 ru1 [rd1] [ld1] [lu1]; p1* ru1* [rd1*] [ld1*] [lu1*] -> p1 ru1 p1* ru1*
            temp = backend.tensordot(W, backend.conj(W), ([1, 3], [1, 3])) # l [u] r [d]; l* [u*] r* [d*] -> l r l* r*
            contr = backend.tensordot(contr, temp, ([1, 3], [0, 2])) # p1 [ru1] p1* [ru1*]; [l] r [l*] r* -> p1 p1* r r*
            temp = backend.tensordot(T2, backend.conj(T2), ([1, 2, 4], [1, 2, 4])) # p2 [ru2] [rd2] ld2 [lu2]; p2* [ru2*] [rd2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
            contr = backend.tensordot(contr, temp, ([2, 3], [1, 3])) # p1 p1* [r] [r*]; p2 [ld2] p2* [ld2*] -> p1 p1* p2 p2*
        else:
            contr = backend.tensordot(Wp1, backend.conj(Wp1), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
            temp = backend.tensordot(T2, backend.conj(T2), ([1, 2], [1, 2])) # p2 [ru2] [rd2] ld2 lu2; p2* [ru2*] [rd2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
            contr = backend.tensordot(temp, contr, ([2, 5], [0, 2])) # p2 ld2 [lu2] p2* ld2* [lu2*]; [rp1] dp1 [rp1*] dp1* -> p2 ld2 p2* ld2* dp1 dp1*
            contr = backend.tensordot(contr, W, ([1, 4], [2, 1])) # p2 [ld2] p2* ld2* [dp1] dp1*; l [u] [r] d -> p2 p2* ld2* dp1* l d
            contr = backend.tensordot(contr, backend.conj(W), ([2, 3, 5], [2, 1, 3])) # p2 p2* [ld2*] [dp1*] l [d]; l* [u*] [r*] [d*] -> p2 p2* l l*
            temp = backend.tensordot(T1, backend.conj(T1), ([2, 3, 4], [2, 3, 4])) # p1 ru1 [rd1] [ld1] [lu1]; p1* ru1* [rd1*] [ld1*] [lu1*] -> p1 ru1 p1* ru1*
            contr = backend.tensordot(temp, contr, ([1, 3], [2, 3])) # p1 [ru1] p1* [ru1*]; p2 p2* [l] [l*] -> p1 p1* p2 p2*
    elif Wp1 is None:
        contr = backend.tensordot(Wm1, backend.conj(Wm1), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
        temp = backend.tensordot(T1, backend.conj(T1), ([3, 4], [3, 4])) # p1 ru1 rd1 [ld1] [lu1]; p1* ru1* rd1* [ld1*] [lu1*] -> p1 ru1 rd1 p1* ru1* rd1*
        contr = backend.tensordot(temp, contr, ([2, 5], [0, 2])) # p1 ru1 [rd1] p1* ru1* [rd1*]; [lm1] um1 [lm1*] um1*; -> p1 ru1 p1* ru1* um1 um1* 
        contr = backend.tensordot(contr, W, ([1, 4], [0, 3])) # p1 [ru1] p1* ru1* [um1] um1*; [l] u r [d] -> p1 p1* ru1* um1* u r
        contr = backend.tensordot(contr, backend.conj(W), ([2, 3, 4], [0, 3, 1])) # p1 p1* [ru1*] [um1*] [u] r; [l*] [u*] r* [d*] -> p1 p1* r r*
        temp = backend.tensordot(T2, backend.conj(T2), ([1, 2, 4], [1, 2, 4])) # p2 [ru2] [rd2] ld2 [lu2]; p2* [ru2*] [rd2*] ld2* [lu2*] -> p2 ld2 p2* ld2*
        contr = backend.tensordot(contr, temp, ([2, 3], [1, 3])) # p1 p1* [r] [r*]; p2 [ld2] p2* [ld2*] -> p1 p1* p2 p2*
    else:
        contr_r = backend.tensordot(Wp1, backend.conj(Wp1), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
        temp = backend.tensordot(T2, backend.conj(T2), ([1, 2], [1, 2])) # p2 [ru2] [rd2] ld2 lu2; p2* [ru2*] [rd2*] ld2* lu2* -> p2 ld2 lu2 p2* ld2* lu2*
        contr_r = backend.tensordot(temp, contr_r, ([2, 5], [0, 2])) # p2 ld2 [lu2] p2* ld2* [lu2*]; [rp1] dp1 [rp1*] dp1* -> p2 ld2 p2* ld2* dp1 dp1*
        contr_r = backend.tensordot(contr_r, W, ([1, 4], [2, 1])) # p2 [ld2] p2* ld2* [dp1] dp1*; l [u] [r] d -> p2 p2* ld2* dp1* l d
        contr_r = backend.tensordot(contr_r, backend.conj(W), ([2, 3], [2, 1])) # p2 p2* [ld2*] [dp1*] l d; l* [u*] [r*] d* -> p2 p2* l d l* d*
        contr_l = backend.tensordot(Wm1, backend.conj(Wm1), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
        temp = backend.tensordot(T1, backend.conj(T1), ([3, 4], [3, 4])) # p1 ru1 rd1 [ld1] [lu1]; p1* ru1* rd1* [ld1*] [lu1*] -> p1 ru1 rd1 p1* ru1* rd1*
        contr_l = backend.tensordot(temp, contr_l, ([2, 5], [0, 2])) # p1 ru1 [rd1] p1* ru1* [rd1*]; [lm1] um1 [lm1*] um1*; -> p1 ru1 p1* ru1* um1 um1*
        contr = backend.tensordot(contr_l, contr_r, ([1, 3, 4, 5], [2, 4, 3, 5])) # p1 [ru1] p1* [ru1*] [um1] [um1*]; p2 p2* [l] [d] [l*] [d*] -> p1 p1* p2 p2*
    norm = backend.transpose(contr, (0, 2, 1, 3)) # p1, p1*, p2, p2* -> p1, p2, p1*, p2*
    norm = backend.reshape(norm, (norm.shape[0] * norm.shape[1], norm.shape[2] * norm.shape[3])) # p1, p2, p1*, p2* -> (p1, p2), (p1*, p2*)
    return backend.tensordot(contr, op, ([0, 1, 2, 3], [2, 0, 3, 1])) / backend.trace(norm) # [p1] [p1*] [p2] [p2*]; [i] [j] [i*] [j*]