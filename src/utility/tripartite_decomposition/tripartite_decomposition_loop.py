from .. import backend
import time 
from sklearn.utils.extmath import randomized_svd
from .. import debug_logging

def get_psi(A,S,B):
    psi_p = backend.tensordot(S,B,axes=[2,1]) # i a j b
    psi_p = backend.tensordot(A,psi_p,axes=[(1,2),(0,2)]) # i a j b
    return(psi_p/backend.norm(psi_p))

def rq(A):
    Q,R = backend.qr(backend.flipud(A).T)
    R = backend.flipud(R.T)
    Q = Q.T
    return R[:,::-1],Q[::-1,:]

def svd_truncated(A, chi_max,p_trunc): 
    #X,Y,Z = svd(A,full_matrices=False)
    X,Y,Z = randomized_svd(A,n_components=chi_max, random_state=0)
    chi_new = backend.min([backend.sum(Y/backend.norm(Y)>p_trunc), chi_max])
    Y = Y[:chi_new] / backend.norm(Y[:chi_new])
    
    return X[:,:chi_new], backend.diag(Y), Z[:chi_new,:]
    
def split_psi(psi, dL, dR, truncation_par={'chi_max':10, 'p_trunc':1e-8,}, max_iter=100, eps = 1e-10, init ='psi0', log_iterates=False): 
    
    #     S     l3   B 
    #   j ---\------/--- mR k
    #      l2 \    / l1  
    #          \  /
    #           | A
    #           |
    #            d
    #            i

    chi_max = truncation_par['chi_max']
    d,mL,mR = psi.shape
    
    if dL*dR > d:
        raise ValueError

    if  init == 'random':
        A = 0.5 - backend.rand(d,dL,dR)
        S = 0.5 - backend.rand(dL,mL,chi_max)
        B = 0.5 - backend.rand(dR,chi_max,mR)
    else:
        if d < mL*mR:
            rho = backend.dot(psi.reshape((d, mL*mR)), psi.reshape((d, mL*mR)).T)
            p, X = backend.eigh(rho)
            perm = backend.argsort(-p)
            X = X[:, perm]
            #p = p[perm]
        else:
            X, Y, Z = backend.svd(psi.reshape(d,mL*mR), full_matrices=True)
            #p = Y**2
        
        #opt = backend.sqrt(backend.sum(p[:dL*dR]))
        A = X[:, :dL*dR].reshape(-1,dL,dR)
    
        theta = backend.tensordot(A.conj(),psi,axes=(0,0)).reshape(dL,dR,mL,mR)
        theta = theta.transpose(0,2,1,3).reshape(dL*mL,dR*mR)    
    
        X,Y,Z = svd_truncated(backend.ascontiguousarray(theta),chi_max,truncation_par['p_trunc'])
        
        S = (X@Y).reshape(dL,mL,-1)
        B = Z.reshape(-1,dR,mR).transpose(1,0,2)

    info = {}
    if log_iterates:
        info["As"] = []
        info["Bs"] = []
        info["Ss"] = []
    
    m = 0
    go = True
    S2s = []
    while m < max_iter and go:
        theta = backend.tensordot(psi,S,axes = [1,1])                        # d mR dL a
        theta = backend.tensordot(theta,B,axes = [(1,3),(2,1)])              # d dL dR
        theta = theta.reshape(d,dL*dR)                                  # d dL*dR
    
        X, Y, Z = backend.svd(backend.ascontiguousarray(theta),full_matrices=0)
        A = (X@Z).reshape(d,dL,dR)
    
        theta = backend.tensordot(psi, A, axes=[0,0])                        # mL mR dL dR
        theta = theta.transpose(2,0,3,1)                                # dL mL dR mR
        theta = theta.reshape(dL*mL,dR*mR)

        X,Y,Z = svd_truncated(backend.ascontiguousarray(theta),chi_max,truncation_par['p_trunc'])
        
        S = (X@Y).reshape(dL,mL,-1)
        B = Z.reshape(-1,dR,mR).transpose(1,0,2)

        S2s.append(backend.sum(Y)) 
        m+=1
        if m > 1:
            go = backend.abs(S2s[-1] - S2s[-2]) >= eps

        if log_iterates:
            info["As"].append(A)
            info["Bs"].append(B)
            info["Ss"].append(S)
    
    info['s_Lambda'] = backend.diag(Y)
    return A, S, B, m, info

def tripartite_decomposition(T, D1, D2, chi, N_iters=100, eps=1e-9, p_trunc=1e-8, initialize="psi0", debug_logger=debug_logging.DebugLogger()):
    chi_1, chi_2, chi_3 = T.shape
    T = T.transpose(0, 2, 1) # chi_1 chi_2 chi_3 = d mR mL -> d mL mR
    # D1 D2 = dR dL
    # chi = chi_max
    A, S, B, m, info = split_psi(psi=T, dL=D2, dR=D1, truncation_par={'chi_max':chi, 'p_trunc':p_trunc,}, max_iter=N_iters, eps=eps, init=initialize, log_iterates=debug_logger.tripartite_decomposition_log_iterates)
    A = A.transpose(0, 2, 1) # A: d dL dR = chi_1 D2 D1 -> chi_1 D1 D2
    B = B.transpose(0, 2, 1) # B: dR chi mR = D1 chi chi_2 -> D1 chi_2 chi
    C = S.transpose(0, 2, 1) # S: dL mL chi = D2 chi_3 chi -> D2 chi chi_3
    if debug_logger.tripartite_decomposition_log_iterates:
        iterates = []
        for i in range(len(info["As"])):
            iterates.append({
                "A":info["As"][i].transpose(0, 2, 1), # A: d dL dR = chi_1 D2 D1 -> chi_1 D1 D2
                "B":info["Bs"][i].transpose(0, 2, 1), # B: dR chi mR = D1 chi chi_2 -> D1 chi_2 chi
                "C":info["Ss"][i].transpose(0, 2, 1) # S: dL mL chi = D2 chi_3 chi -> D2 chi chi_3
                })
        debug_logger.append_to_log_list(("tripartite_decomopsition_info", "iterates"), iterates)
    if debug_logger.tripartite_decomposition_log_info:
        debug_logger.append_to_log_list(("tripartite_decomopsition_info", "N_iters"), m)
    return A, B, C



