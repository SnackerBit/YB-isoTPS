from ...utility import backend

"""
This file implements the computation of norms and expectation values for one-site and two-site environments for the honeycomb isoTPS.
We need to differentiate between the two different possible onesite/twosite environments in the honeycomb isoTPS.
The following functions thus all carry either the subscript 1 or 2.
"""

def compute_norm_onesite_1(T, W, Wp1):
    r"""
    Computes the norm of the one-site wave function

             | /  
             |/    
            Wp1                      
            /|                   
           / |              
        --T  |                
           \ |             
            \|                      
             W                 
             |\ 
             | \ 

    by contracting all tensors. The orthogonality center must be 
    either at W or Wp1. Wp1 or W can also be None.

    Parameters
    ----------
    T : backend.array_type of shape (i, ru, rd, l)
        part of the 1-site wavefunction
    W: backend.array_type of shape (l, u, r, d) or None
        part of the 1-site wavefunction
    Wp1 : backend.array_type of shape (lp1, up1, rp1, dp1) or None
        part of the 1-site wavefunction

    Returns
    -------
    norm : float
    """
    if W is None:
        contr = backend.tensordot(Wp1, backend.conj(Wp1), ([1, 2, 3], [1, 2, 3])) # l [u] [r] [d]; l* [u*] [r*] [d*] -> l l*
        contr = backend.tensordot(T, contr, ([1], [0])) # p [ru] rd l; [l] l* -> p rd l l*
        contr = backend.tensordot(contr, backend.conj(T), ([0, 1, 2, 3], [0, 2, 3, 1])) # [p] [rd] [l] [l*]; [p*] [ru*] [rd*] [l*]
    elif Wp1 is None:
        contr = backend.tensordot(W, backend.conj(W), ([1, 2, 3], [1, 2, 3])) # lm1 [um1] [rm1] [dm1]; lm1* [um1*] [rm1*] [dm1*] -> lm1 lm1*
        contr = backend.tensordot(T, contr, ([2], [0])) # p ru [rd] l; [lm1] lm1* -> p ru l lm1*
        contr = backend.tensordot(contr, backend.conj(T), ([0, 1, 2, 3], [0, 1, 3, 2])) # [p] [ru] [l] [lm1*]; [p*] [ru*] [rd*] [l*]
    else:
        contr = backend.tensordot(W, backend.conj(W), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1* 
        temp = backend.tensordot(Wp1, backend.conj(Wp1), ([1, 2], [1, 2])) # l [u] [r]] d; l* [u*] [r*] d* -> l d l* d*
        contr = backend.tensordot(contr, temp, ([1, 3], [1, 3])) # lm1 [um1] lm1* [um1*]; l [d] l* [d*] -> lm1 lm1* l l*
        temp = backend.tensordot(T, backend.conj(T), ([0, 3], [0, 3])) # [p] ru rd [l]; [p*] ru* rd* [l*] -> ru rd ru* rd*
        contr = backend.tensordot(temp, contr, ([0, 1, 2, 3], [2, 0, 3, 1])) # [ru] [rd] [ru*] [rd*]; [lm1] [lm1*] [l] [l*]
    return backend.sqrt(contr.item())

def compute_norm_onesite_2(T, W):
    r"""
    Computes the norm of the one-site wave function

            \          
             \ |    |   
              \|    |     
               T----W---- 
              /     |    
             /      |   
            /          

    by contracting all tensors.

    Parameters
    ----------
    T : backend.array_type of shape (i, r, ld, lu) 
        part of the 1-site wavefunction
    W : backend.array_type of shape (l, u, r, d)
        part of the 1-site wavefunction

    Returns
    -------
    norm : float
    """
    contr = backend.tensordot(T, backend.conj(T), ([0, 2, 3], [0, 2, 3])) # [i] r [ld] [lu]; [i*] r* [ld*] [lu*] -> r r*
    contr = backend.tensordot(contr, W, ([0], [0])) # [r] r*; [l] u r d -> r* u r d
    contr = backend.tensordot(contr, backend.conj(W), ([0, 1, 2, 3], [0, 1, 2, 3])) # [r*] [u] [r] [d]; [l*] [u*] [r*] [d*]
    return backend.sqrt(contr.item())

def compute_norm_twosite(T1, T2, Wm1, W, Wp1):
    r"""
    Computes the norm of the two-site wave function

                    \ | 
                     \|
                     Wp1
                      |\  | 
                      | \ |
                      |  T2--                 \                 /
                      | /                      \  |          | / 
                      |/                        \ |          |/ 
                      W              or          T1----W----T2
                  |  /|                         /             \ 
                  | / |                        /               \ 
                --T1  |                       /                 \ 
                    \ |  
                     \|
                     Wm1 
                      |\ 
                      | \ 

    by contracting all tensors.
    The orthogonality center must be either at Wm1, W, or Wp1.
    Wm1 and/or Wm1 can also be None.

    Parameters
    ----------
    T1 : backend.array_type of shape (i, ru, rd, l)
        part of the 1-site wavefunction
    T2 : backend.array_type of shape (j, r, ld, lu)
        part of the 1-site wavefunction
    Wm1 : backend.array_type of shape (lm1, um1, rm1, dm1) or None
        part of the 1-site wavefunction
    W : backend.array_type of shape (l, u, r, d)
        part of the 1-site wavefunction
    Wp1 : backend.array_type of shape (lp1, up1, rp1, dp1) or None
        part of the 1-site wavefunction

    Returns
    -------
    float :
        the norm
    """
    if Wm1 is None:
        if Wp1 is None:
            # We are in the 2nd form!
            contr = backend.tensordot(T1, backend.conj(T1), ([0, 2, 3], [0, 2, 3])) # [p1] r1 [ld1] [lu1]; [p2*] r1* [ld*] [lu*] -> r1 r1*
            contr = backend.tensordot(contr, W, ([0], [0])) # [r1] r1*; [l] u r d -> r1* u r d
            contr = backend.tensordot(contr, backend.conj(W), ([0, 1, 3], [0, 1, 3])) # [r1*] [u] r [d]; [l*] [u*] r* [d*] -> r r*
            temp = backend.tensordot(T2, backend.conj(T2), ([0, 1, 2], [0, 1, 2])) # [p2] [ru2] [rd2] l2; [p2*] [ru2*] [rd2*] l2* -> l2 l2*
            contr = backend.tensordot(contr, temp, ([0, 1], [0, 1])) # [r] [r*]; [l2] [l2*]
        else:
            contr = backend.tensordot(Wp1, backend.conj(Wp1), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
            temp = backend.tensordot(T2, backend.conj(T2), ([0, 1], [0, 1])) # [p2] [r2] ld2 lu2; [p2*] [r2*] ld2* lu2* -> ld2 lu2 ld2* lu2*
            contr = backend.tensordot(temp, contr, ([1, 3], [0, 2])) # ld2 [lu2] ld2* [lu2*]; [rp1] dp1 [rp1*] dp1* -> ld2 ld2* dp1 dp1*
            contr = backend.tensordot(contr, W, ([0, 2], [2, 1])) # [ld2] ld2* [dp1] dp1*; l [u] [r] d -> ld2* dp1* l d
            contr = backend.tensordot(contr, backend.conj(W), ([0, 1, 3], [2, 1, 3])) # [ld2*] [dp1*] l [d]; l* [u*] [r*] [d*] -> l l*
            temp = backend.tensordot(T1, backend.conj(T1), ([0, 2, 3], [0, 2, 3])) # [p1] ru1 [rd1] [l1]; [p1*] ru1* [rd1*] [l1*] -> ru1 ru1*
            contr = backend.tensordot(temp, contr, ([0, 1], [0, 1])) # [ru1] [ru1*]; [l] [l*]
    elif Wp1 is None:
        contr = backend.tensordot(Wm1, backend.conj(Wm1), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
        temp = backend.tensordot(T1, backend.conj(T1), ([0, 3], [0, 3])) # [p1] ru1 rd1 [l1]; [p1*] ru1* rd1* [l1*] -> ru1 rd1 ru1* rd1*
        contr = backend.tensordot(temp, contr, ([1, 3], [0, 2])) # ru1 [rd1] ru1* [rd1*]; [lm1] um1 [lm1*] um1*; -> ru1 ru1* um1 um1* 
        contr = backend.tensordot(contr, W, ([0, 2], [0, 3])) # [ru1] ru1* [um1] um1*; [l] u r [d] -> ru1* um1* u r
        contr = backend.tensordot(contr, backend.conj(W), ([0, 1, 2], [0, 3, 1])) # [ru1*] [um1*] [u] r; [l*] [u*] r* [d*] -> r r*
        temp = backend.tensordot(T2, backend.conj(T2), ([0, 1, 3], [0, 1, 3])) # [p2] [r2] ld2 [lu2]; [p2*] [r2*] ld2* [lu2*] -> ld2 ld2*
        contr = backend.tensordot(contr, temp, ([0, 1], [0, 1])) # [r] [r*]; [ld2] [ld2*]
    else:
        contr_r = backend.tensordot(Wp1, backend.conj(Wp1), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
        temp = backend.tensordot(T2, backend.conj(T2), ([0, 1], [0, 1])) # [p2] [r2] ld2 lu2; [p2*] [r2*] ld2* lu2* -> ld2 lu2 ld2* lu2*
        contr_r = backend.tensordot(temp, contr_r, ([1, 3], [0, 2])) # ld2 [lu2] ld2* [lu2*]; [rp1] dp1 [rp1*] dp1* -> ld2 ld2* dp1 dp1*
        contr_r = backend.tensordot(contr_r, W, ([0, 2], [2, 1])) # [ld2] ld2* [dp1] dp1*; l [u] [r] d -> ld2* dp1* l d
        contr_r = backend.tensordot(contr_r, backend.conj(W), ([0, 1], [2, 1])) # [ld2*] [dp1*] l d; l* [u*] [r*] d* -> l d l* d*
        contr_l = backend.tensordot(Wm1, backend.conj(Wm1), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
        temp = backend.tensordot(T1, backend.conj(T1), ([0, 3], [0, 3])) # [p1] ru1 rd1 [l1]; [p1*] ru1* rd1* [l1*] -> ru1 rd1 ru1* rd1*
        contr_l = backend.tensordot(temp, contr_l, ([1, 3], [0, 2])) # ru1 [rd1] ru1* [rd1*]; [lm1] um1 [lm1*] um1*; -> ru1 ru1* um1 um1*
        contr = backend.tensordot(contr_l, contr_r, ([0, 1, 2, 3], [0, 2, 1, 3])) # [ru1] [ru1*] [um1] [um1*]; [l] [d] [l*] [d*]
    return backend.sqrt(contr.item())

def expectation_value_onesite_1(T, W, Wp1, op):
    r"""
    Computes the norm of the one-site wave function

             | /  
             |/    
            Wp1                      
            /|                   
           / |              
        --T  |                
           \ |             
            \|                      
             W                 
             |\ 
             | \ 

    by contracting all tensors. The orthogonality center must be 
    either at W or Wp1. Wp1 or W can also be None.

    Parameters
    ----------
    T : backend.array_type of shape (i, ru, rd, l)
        part of the 1-site wavefunction
    W : backend.array_type of shape (l, u, r, d) or None
        part of the 1-site wavefunction
    Wp1 : backend.array_type of shape (lp1, up1, rp1, dp1) or None
        part of the 1-site wavefunction
    op : backend.array_type of shape (i, i*)
        part of the 1-site wavefunction

    Returns
    -------
    float:
        the norm
    """
    if W is None:
        contr = backend.tensordot(Wp1, backend.conj(Wp1), ([1, 2, 3], [1, 2, 3])) # l [u] [r] [d]; l* [u*] [r*] [d*] -> l l*
        contr = backend.tensordot(T, contr, ([1], [0])) # p [ru] rd l; [l] l* -> p rd l l*
        contr = backend.tensordot(contr, backend.conj(T), ([1, 2, 3], [2, 3, 1])) # i [rd] [l] [l*]; i* [ru*] [rd*] [l*] -> i i*
    elif Wp1 is None:
        contr = backend.tensordot(W, backend.conj(W), ([1, 2, 3], [1, 2, 3])) # lm1 [um1] [rm1] [dm1]; lm1* [um1*] [rm1*] [dm1*] -> lm1 lm1*
        contr = backend.tensordot(T, contr, ([2], [0])) # p ru [rd] l; [lm1] lm1* -> p ru l lm1*
        contr = backend.tensordot(contr, backend.conj(T), ([1, 2, 3], [1, 3, 2])) # i [ru] [l] [lm1*]; i* [ru*] [rd*] [l*] -> i i*
    else:
        contr = backend.tensordot(W, backend.conj(W), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1* 
        temp = backend.tensordot(Wp1, backend.conj(Wp1), ([1, 2], [1, 2])) # l [u] [r]] d; l* [u*] [r*] d* -> l d l* d*
        contr = backend.tensordot(contr, temp, ([1, 3], [1, 3])) # lm1 [um1] lm1* [um1*]; l [d] l* [d*] -> lm1 lm1* l l*
        contr = backend.tensordot(contr, T, ([0, 2], [2, 1])) # [lm1] lm1* [l] l*; i [ru] [rd] l -> lm1* l* i l
        contr = backend.tensordot(contr, backend.conj(T), ([0, 1, 3], [2, 1, 3])) # [lm1*] [l*] i [l]; i* [ru*] [rd*] [l*] -> i i*
    return backend.tensordot(contr, op, ([0, 1], [1, 0])) / backend.trace(contr) # [i] [i*]; [i] [i*]

def expectation_value_onesite_2(T, W, op):
    r"""
    Computes the norm of the one-site wave function

            \          
             \ |    |   
              \|    |     
               T----W---- 
              /     |    
             /      |   
            /          

    by contracting all tensors.

    Parameters
    ----------
    T : backend.array_type of shape (i, r, ld, lu) 
        part of the 1-site wavefunction
    W : backend.array_type of shape (l, u, r, d)
        part of the 1-site wavefunction
    op : backend.array_type of shape (i, i*)
        1-site operator

    Returns
    -------
    result : complex
        the resulting expectation value
    """
    contr = backend.tensordot(T, backend.conj(T), ([2, 3], [2, 3])) # i r [ld] [lu]; i* r* [ld*] [lu*] -> i r i* r*
    contr = backend.tensordot(contr, W, ([1], [0])) # i [r] i* r*; [l] u r d -> i i* r* u r d
    contr = backend.tensordot(contr, backend.conj(W), ([2, 3, 4, 5], [0, 1, 2, 3])) # i i* [r*] [u] [r] [d]; [l*] [u*] [r*] [d*] -> i i*
    return backend.tensordot(contr, op, ([0, 1], [1, 0])) / backend.trace(contr) # [i] [i*]; [i] [i*]

def expectation_value_twosite_1(T1, T2, Wm1, W, Wp1, op):
    r"""
    Computes the expectation value of the given operator with respect to the two-site wave function

                    \ |                 
                     \|
                     Wp1
                      |\  | 
                      | \ |
                      |  T2-- 
                      | /    
                      |/      
                      W      
                  |  /|       
                  | / |        
                --T1  |       
                    \ |  
                     \|
                     Wm1 
                      |\ 
                      | \ 

    by contracting all tensors.
    The orthogonality center must be either at Wm1, W, or Wp1.
    Wm1 or Wm1 can also be None.

    Parameters
    ----------
    T1 : backend.array_type of shape (i, ru, rd, l)
        part of the twosite wavefunction
    T2 : backend.array_type of shape (j, r, ld, lu)
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
            contr = backend.tensordot(T1, backend.conj(T1), ([2, 3], [2, 3])) # i ru1 [rd1] [l1]; i* ru1* [rd1*] [l1*] -> i ru1 i* ru1*
            contr = backend.tensordot(contr, W, ([1], [0])) # i [ru1] i* ru1*; [l] u r d -> i i* ru1* u r d
            contr = backend.tensordot(contr, backend.conj(W), ([2, 3, 5], [0, 1, 3])) # i i* [ru1*] [u] r [d]; [l*] [u*] r* [d*] -> i i* r r*
            temp = backend.tensordot(T2, backend.conj(T2), ([1, 3], [1, 3])) # j [r2] ld2 [lu2]; j* [r2*] ld2* [lu2*] -> j ld2 j* ld2*
            contr = backend.tensordot(contr, temp, ([2, 3], [1, 3])) #  i i* [r] [r*]; j [ld2] j* [ld2*] -> i i* j j*
        else:
            contr = backend.tensordot(Wp1, backend.conj(Wp1), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
            temp = backend.tensordot(T2, backend.conj(T2), ([1], [1])) # j [r2] ld2 lu2; [p2*] j* ld2* lu2* -> j ld2 lu2 j* ld2* lu2*
            contr = backend.tensordot(temp, contr, ([2, 5], [0, 2])) # j ld2 [lu2] j* ld2* [lu2*]; [rp1] dp1 [rp1*] dp1* -> j ld2 j* ld2* dp1 dp1*
            contr = backend.tensordot(contr, W, ([1, 4], [2, 1])) # j [ld2] j* ld2* [dp1] dp1*; l [u] [r] d -> j j* ld2* dp1* l d
            contr = backend.tensordot(contr, backend.conj(W), ([2, 3, 5], [2, 1, 3])) # j j* [ld2*] [dp1*] l [d]; l* [u*] [r*] [d*] -> j j* l l*
            temp = backend.tensordot(T1, backend.conj(T1), ([2, 3], [2, 3])) # i ru1 [rd1] [l1]; i* ru1* [rd1*] [l1*] -> i ru1 i* ru1*
            contr = backend.tensordot(temp, contr, ([1, 3], [2, 3])) # i [ru1] i* [ru1*]; j j* [l] [l*] -> i i* j j*
    elif Wp1 is None:
        contr = backend.tensordot(Wm1, backend.conj(Wm1), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
        temp = backend.tensordot(T1, backend.conj(T1), ([3], [3])) # i ru1 rd1 [l1]; i* ru1* rd1* [l1*] -> i ru1 rd1 i* ru1* rd1*
        contr = backend.tensordot(temp, contr, ([2, 5], [0, 2])) # i ru1 [rd1] i* ru1* [rd1*]; [lm1] um1 [lm1*] um1*; -> i ru1 i* ru1* um1 um1* 
        contr = backend.tensordot(contr, W, ([1, 4], [0, 3])) # i [ru1] i* ru1* [um1] um1*; [l] u r [d] -> i i* ru1* um1* u r
        contr = backend.tensordot(contr, backend.conj(W), ([2, 3, 4], [0, 3, 1])) # i i* [ru1*] [um1*] [u] r; [l*] [u*] r* [d*] -> i i* r r*
        temp = backend.tensordot(T2, backend.conj(T2), ([1, 3], [1, 3])) # j [r2] ld2 [lu2]; j* [r2*] ld2* [lu2*] -> j ld2 j* ld2*
        contr = backend.tensordot(contr, temp, ([2, 3], [1, 3])) # i i* [r] [r*]; j [ld2] j* [ld2*] -> i i* j j*
    else:
        contr_r = backend.tensordot(Wp1, backend.conj(Wp1), ([0, 1], [0, 1])) # [lp1] [up1] rp1 dp1; [lp1*] [up1*] rp1* dp1* -> rp1 dp1 rp1* dp1*
        temp = backend.tensordot(T2, backend.conj(T2), ([1], [1])) # j [r2] ld2 lu2; j* [r2*] ld2* lu2* -> j ld2 lu2 j* ld2* lu2*
        contr_r = backend.tensordot(temp, contr_r, ([2, 5], [0, 2])) # j ld2 [lu2] j* ld2* [lu2*]; [rp1] dp1 [rp1*] dp1* -> j ld2 j* ld2* dp1 dp1*
        contr_r = backend.tensordot(contr_r, W, ([1, 4], [2, 1])) # j [ld2] j* ld2* [dp1] dp1*; l [u] [r] d -> j j* ld2* dp1* l d
        contr_r = backend.tensordot(contr_r, backend.conj(W), ([2, 3], [2, 1])) # j j* [ld2*] [dp1*] l d; l* [u*] [r*] d* -> j j* l d l* d*
        contr_l = backend.tensordot(Wm1, backend.conj(Wm1), ([2, 3], [2, 3])) # lm1 um1 [rm1] [dm1]; lm1* um1* [rm1*] [dm1*] -> lm1 um1 lm1* um1*
        temp = backend.tensordot(T1, backend.conj(T1), ([3], [3])) # i ru1 rd1 [l1]; i* ru1* rd1* [l1*] -> i ru1 rd1 i* ru1* rd1*
        contr_l = backend.tensordot(temp, contr_l, ([2, 5], [0, 2])) # i ru1 [rd1] i* ru1* [rd1*]; [lm1] um1 [lm1*] um1*; -> i ru1 i* ru1* um1 um1*
        contr = backend.tensordot(contr_l, contr_r, ([1, 3, 4, 5], [2, 4, 3, 5])) # i [ru1] i* [ru1*] [um1] [um1*]; j j* [l] [d] [l*] [d*] -> i i* j j*
    result = backend.tensordot(contr, op, ([0, 1, 2, 3], [2, 0, 3, 1])) # [i] [i*] [j] [j*]; [i] [j] [i*] [j*]
    contr = contr.transpose(0, 2, 1, 3)
    i1, j1, i2, j2 = contr.shape
    return result / backend.trace(contr.reshape(i1*j1, i2*j2))

def expectation_value_twosite_2(T1, T2, W, op):
    r"""
    Computes the expectation value of the given operator wrt the the two-site wave function

                         
                \                 /
                 \  |          | / 
                  \ |          |/ 
                   T1----W----T2
                  /             \ 
                 /               \ 
                /                 \ 
                

    by contracting all tensors.
    The orthogonality center must be either at Wm1, W, or Wp1.
    Wm1 and/or Wm1 can also be None.

    Parameters
    ----------
    T1 : backend.array_type of shape (i, r, ld, lu)
        part of the twosite wavefunction
    T2 : backend.array_type of shape (j, ru, rd, l)
        part of the twosite wavefunction
    W : backend.array_type of shape (l, u, r, d)
        part of the twosite wavefunction
    op : backend.array_type of shape (i, j, i*, j*)
        twosite operator

    Returns
    -------
    result : complex
        the resulting expectation value
    """
    contr = backend.tensordot(T1, backend.conj(T1), ([2, 3], [2, 3])) # i r1 [ld1] [lu1]; i* r1* [ld*] [lu*] -> i r1 i* r1*
    contr = backend.tensordot(contr, W, ([1], [0])) # i [r1] i* r1*; [l] u r d -> i i* r1* u r d
    contr = backend.tensordot(contr, backend.conj(W), ([2, 3, 5], [0, 1, 3])) # i i* [r1*] [u] r [d]; [l*] [u*] r* [d*] -> i i* r r*
    temp = backend.tensordot(T2, backend.conj(T2), ([1, 2], [1, 2])) # j [ru2] [rd2] l2; j* [ru2*] [rd2*] l2* -> j l2 j* l2*
    contr = backend.tensordot(contr, temp, ([2, 3], [1, 3])) # i i* [r] [r*]; j [l2] j* [l2*] -> i i* j j*
    result = backend.tensordot(contr, op, ([0, 1, 2, 3], [2, 0, 3, 1])) # [i] [i*] [j] [j*]; [i] [j] [i*] [j*]
    contr = contr.transpose(0, 2, 1, 3)
    i1, j1, i2, j2 = contr.shape
    return result / backend.trace(contr.reshape(i1*j1, i2*j2))