from .. import backend
from .. import utility

"""
This file implements several iterate classes that are used in the disentangle process optimizing the truncation error.
The optimization algorithm uses instances of iterate classes to represent the current iterate. Iterate classes must implement functionality
to evaluate the cost function, compute the gradient, and optionally compute the hessian vector product (if the iterate class is to be used in
the trust region optimizer). Iterate classes may use caching to minimize the number of necessary computations.
"""

class TruncErrorIterate:
    """
    Base class for the more specialized iterate classes.
    """

    def __init__(self, U, theta, chi, chi_max=None, N_iters_svd=None, eps_svd=0.0, old_iterate=None):
        """
        Initializes new iterate.

        Parameters
        ----------
        U : backend.array_type of shape (i, j, i*, j*)
            disentangling unitary.
        theta : backend.array_type of shape (l, i, j, r)
            wavefunction tensor to be disentangled.
        chi : int 
            Splitting bond dimension for computing the truncation error (cost function).
        chi_max : int or None, optional
            Splitting bond dimension for computing the SVD. If only the gradient computation is needed,
            this should be set to chi, since both the computation of the cost function and the computation
            of the gradient are correct for chi_max = chi. However, the hessian vector product uses derivatives
            of the U and V factors of the SVD, which are not the same for truncated SVD. Thus, the hvp is only
            an approximation if chi_max < min(l*i, j*r). If chi_max is set to None, full SVD is used. Default: None.
        N_iters_svd : int or None, optional
            number of iterations the qr splitting algorithm is run for approximating the SVD.
            If this is set to None, a full SVD is performed instead. Default: None.
        eps_svd : float, optional
            eps parameter passed into split_matrix_iterate_QR(), 
            see src/utility/utility.py for more information. Default: 0.0.
        old_iterate : element of TruncErrorIterate class or None, optional
            old iterate. This is used here for initializing the qr splitting algorithm with the
            old result, which can lead to faster convergence. Default: None.
        """
        self.U = U
        self.theta = theta
        self.chi = chi
        self.chi_max = chi_max
        self.N_iters_svd = N_iters_svd
        self.eps_svd = eps_svd
        self.l, self.D1, self.D2, self.r = self.theta.shape
        self.k = min(self.l*self.D1, self.D2*self.r)
        if self.chi_max is not None:
            self.k = min(self.k, self.chi_max)
        self.Y0 = None
        if self.N_iters_svd is not None and old_iterate is not None:
            self.Y0 = old_iterate.Y0

    def get_iterate(self):
        """
        Returns the actual iterate (the disentangling unitary of shape (i, j, i*, j*)).

        Returns
        -------
        U : backend.array_type of shape (i, j, i*, j*)
            disentangling unitary.
        """
        return self.U

class TruncErrorIterateCG(TruncErrorIterate):
    """
    Class representing a single iterate of the trunc error Conjugate Gradients optimizer.
    Cashes important results such that they do not need to be computed multiple times.
    """

    def evaluate_cost_function(self):
        """
        Computes the truncation error 1 - sum_{i=1}^{chi} s_i**2 with s_i the ith singular value of U@theta.

        Returns
        -------
        cost : float
            value of the cost function
        """
        if self.chi >= min(self.l*self.D1, self.D2*self.r):
            self.cost = 0.0
            return self.cost
        # Compute Utheta
        Utheta = backend.dot(backend.ascontiguousarray(self.U).reshape((self.D1*self.D2, -1)), backend.ascontiguousarray(self.theta.transpose(1, 2, 0, 3)).reshape(self.D1*self.D2, -1)) # l, i, j, r -> i, j, l, r -> (i, j), (l, r)
        Utheta = backend.ascontiguousarray(backend.ascontiguousarray(Utheta).reshape((self.D1, self.D2, self.l, self.r)).transpose((2, 0, 1, 3))).reshape((self.l*self.D1, -1)) # (i, j), (l, r) -> i, j, l, r -> l, i, j, r -> (l, i), (j, r)
        # Renormalize (might not be normalized due to numerical errors)
        Utheta /= backend.norm(Utheta)
        # Perform SVD
        if self.N_iters_svd is None:
            self.X, self.S, self.Y = utility.safe_svd(Utheta, full_matrices=False) # { D^9 }
            idx = backend.argsort(self.S)[::-1][:self.chi]
            self.X, self.S, self.Y = self.X[:, idx], self.S[idx], self.Y[idx, :]
        else:
            self.X, self.Y, _, _ = utility.split_matrix_iterate_QR(Utheta, self.chi, self.N_iters_svd, self.eps_svd, C0=self.Y0, normalize=False) # { N_iters_svd * D^7 }
            self.Y0 = self.Y
            XX, self.S, self.Y = backend.svd(self.Y, full_matrices=False) # { D^5 }
            self.X = self.X@XX
        # Compute cost function.
        # 1 - backend.sum(self.S**2) should be larger than zero, but can be lower than zero due to numerical errors.
        self.cost = backend.sqrt(max(1 - backend.sum(self.S**2), 0.0))
        return self.cost

    def compute_gradient(self):
        """
        Computes the gradient of the truncation error cost function, projected to the tangent space of the iterate.

        Returns
        -------
        grad : backend.array_type of shape (i, j, i*, j*)
            gradient of the cost function
        """
        if self.cost <= 1e-14:
            return backend.zeros(self.U.shape, dtype=self.U.dtype) 
        grad = backend.ascontiguousarray(self.X@backend.diag(self.S)@self.Y) # -> (l, i), (j, r)
        grad = backend.reshape(grad, (self.l, self.D1, self.D2, self.r))
        grad = backend.ascontiguousarray(grad.transpose((0, 3, 1, 2))).reshape(self.l*self.r, self.D1*self.D2) # l, i, j, r -> l, r, i, j -> (l, r), (i, j)
        theta = backend.ascontiguousarray(self.theta.transpose((0, 3, 1, 2))).reshape(self.l*self.r, self.D1*self.D2) # l, i, j, r -> l, r, i, j -> (l, r), (i, j)
        grad = backend.dot(grad.T, backend.conj(theta)) # (i j) [(l r)]; [(l r)*] (i j)* -> (i j) (i j)*
        grad = -grad/self.cost
        U = self.U.reshape(self.D1*self.D2, self.D1*self.D2)
        return (grad - 0.5 * U@(U.conj().T@grad + grad.conj().T@U)).reshape(self.D1, self.D2, self.D1, self.D2)

class TruncErrorIterateTRM(TruncErrorIterate):
    """
    Class representing a single iterate of the trunc error trust region optimizer.
    Cashes important results such that they do not need to be computed multiple times.
    This is especially useful when multiple hessian vector products must be computed.
    """

    def __init__(self, U, theta, chi):
        """
        Initializes new iterate.

        Parameters
        ----------
        See class TruncErrorIterate.
        """
        TruncErrorIterate.__init__(self, U, theta, chi, chi_max=None, N_iters_svd=None, eps_svd=0, old_iterate=None)
        self.computed_gradient = False
        self.computed_hessian = False

    def evaluate_cost_function(self):
        """
         Computes the truncation error 1 - sum_{i=1}^{chi} s_i**2 with s_i the ith singular value of U@theta.

        Returns
        -------
        cost : float
            value of the cost function
        """
        # Compute Utheta
        Utheta = backend.tensordot(self.U, self.theta, ([2, 3], [1, 2])) # i j [i*] [j*]; l [i] [j] r -> i j l r { D^8 }
        Utheta = Utheta.transpose(2, 0, 1, 3) # i, j, l, r -> l, i, j, r
        # Renormalize (might not be normalized due to numerical errors)
        Utheta /= backend.norm(Utheta)
        # Perform SVD
        l, i, j, r = Utheta.shape
        self.X, self.S, self.Y = utility.safe_svd(Utheta.reshape(l*i, j*r), full_matrices=False) # { D^9 }
        self.Y = backend.conj(self.Y.T)
        idx = backend.argsort(self.S)[::-1]
        self.X, self.S, self.Y = self.X[:, idx], self.S[idx], self.Y[:, idx]
        self.X_trunc, self.S_trunc, self.Y_trunc = self.X[:, :self.chi], self.S[:self.chi], self.Y[:, :self.chi]
        # Compute cost function.
        # 1 - backend.sum(self.S**2) should be larger than zero, but can be lower than zero due to numerical errors.
        self.cost = backend.sqrt(max(1 - backend.sum(self.S_trunc**2), 0.0))
        return self.cost

    def compute_gradient(self):
        """
        Computes the gradient of the truncation error cost function, projected to the tangent space of the iterate.

        Returns
        -------
        grad : backend.array_type of shape (i, j, i*, j*)
            gradient of the cost function
        """
        if not self.computed_gradient:
            Xtheta = backend.tensordot(self.X_trunc.reshape(self.l, self.D1, -1), backend.conj(self.theta), ([0], [0])) # [l] i chi; [l*] k* l* r* -> i chi k* l* r* { D^8 }
            XStheta = backend.tensordot(Xtheta, backend.diag(self.S_trunc), ([1], [0])) # i [chi] k* l* r*; [chi] chi -> i k* l* r* chi { D^7 }
            self.XSYtheta = backend.tensordot(XStheta, backend.conj(self.Y_trunc.reshape(self.D2, self.r, -1)), ([3, 4], [1, 2])) # i k* l* [r*] [chi]; j [r] [chi] -> i k* l* j  { D^7 }
            self.XSYtheta = self.XSYtheta.transpose(0, 3, 1, 2) # i, k*, l*, j -> i, j, k*, l*
            self.XSYthetaUU = backend.tensordot(backend.conj(self.XSYtheta), self.U, ([2, 3], [2, 3])) # i' j' [k'] [l']; i j [k'] [l'] -> i' j' i j { D^6 }
            self.XSYthetaUU = backend.tensordot(self.XSYthetaUU, self.U, ([0, 1], [0, 1])) # [i'] [j'] i j; [i'] [j'] k l -> i j k l { D^6 }
            self.computed_gradient = True
        if self.cost == 0:
            # Avoid dividing by zero
            return backend.zeros(self.U.shape)
        else:
            return - (self.XSYtheta - self.XSYthetaUU) / 2 / self.cost
    
    def compute_hessian_vector_product(self, dU):
        """
        Computes the riemannian hessian vector product of dU, which is the projection of D(grad f(U))[dU] to the tangent space of U. 
        When this is called for the first time, several intermediate results are cached for the case that this is called again in the future.

        Parameters
        ----------
        dU : backend.array_type of shape (i, j, i*, j*)
            element from the tangent space of U

        Returns
        -------
        hvp : backend.array_type of shape (i, j, i*, j*)
            hessian vector product H@dU
        """
        if self.cost == 0:
            # Avoid dividing by zero
            return backend.zeros(self.U.shape)
        if not self.computed_hessian:
            # Cache helper results
            self.F = backend.zeros((self.k, self.k))
            for i in range(self.k):
                for j in range(self.k):
                    if i != j:
                        temp = self.S[j]**2 - self.S[i]**2
                        if temp == 0.0:
                            self.F[i, j] = 0.0
                        else:
                            self.F[i, j] = 1/temp
            self.computed_hessian = True
            non_zero_indices = backend.where(self.S != 0.0)
            self.S_inv = backend.zeros(self.S.shape, dtype=self.S.dtype)
            self.S_inv[non_zero_indices] = 1 / self.S[non_zero_indices]
        # Compute dP and dD
        l, D1, D2, r = self.theta.shape
        dUtheta = backend.tensordot(dU, self.theta, ([2, 3], [1, 2])) # i j [i*] [j*]; l [i] [j] r -> i j l r { D^8 }
        dUtheta = dUtheta.transpose(2, 0, 1, 3).reshape(l*D1, D2*r) # i, j, l, r -> l, i, j, r
        dP = backend.conj(self.X.T)@dUtheta@self.Y
        dD = 1.j * backend.imag(backend.diag(dP)) * self.S_inv / 2
        # Compute dX, dS and dY
        dX = self.X@(self.F * (dP*self.S + self.S[:, backend.newaxis]*backend.conj(dP.T)) + backend.diag(dD)) + (backend.eye(l*D1) - self.X@backend.conj(self.X.T))@dUtheta@self.Y * self.S_inv
        dS = backend.real(backend.diag(dP))[:self.chi]
        dY = self.Y@(self.F * (self.S[:, backend.newaxis]*dP + backend.conj(dP.T)*self.S) - backend.diag(dD)) + (backend.eye(D2*r) - self.Y@backend.conj(self.Y.T))@backend.conj(dUtheta.T)@self.X * self.S_inv
        # Compute first term of the product rule
        result = backend.sum(self.S_trunc*dS) * (self.XSYtheta - self.XSYthetaUU) / self.cost**2
        # Compute helper tensor necessary for computing the first two of the remaining four terms
        dXSY = (dX[:, :self.chi]*self.S_trunc)@self.Y_trunc.T.conj()
        dXSY += (self.X_trunc*dS)@self.Y_trunc.T.conj()
        dXSY += (self.X_trunc*self.S_trunc)@dY.T.conj()[:self.chi, :]
        dXSY = dXSY.reshape(l, D1, D2, r)
        # first of four remaining terms
        temp_result = backend.tensordot(dXSY, backend.conj(self.theta), ([0, 3], [0, 3])) # [l] i j [r]; [l*] k* l* [r*] -> i j k* l*
        # second of four remaining terms
        temp = backend.tensordot(backend.conj(temp_result), self.U, ([2, 3], [2, 3])) # i* j* [k*] [l*]; i j [k] [l] -> i* j* i j
        temp = backend.tensordot(temp, self.U, ([0, 1], [0, 1])) # [i*] [j*] i j; [i] [j] k l -> i j k l
        temp_result -= temp
        # third of four remaining terms
        temp = backend.tensordot(backend.conj(self.XSYtheta), dU, ([2, 3], [2, 3])) # i* j* [k*] [l*]; i j [k] [l] -> i* j* i j
        temp = backend.tensordot(temp, self.U, ([0, 1], [0, 1])) # [i*] [j*] i j; [i] [j] k l -> i j k l
        temp_result -= temp
        # fourth of four remaining terms
        temp = backend.tensordot(backend.conj(self.XSYtheta), self.U, ([2, 3], [2, 3])) # i* j* [k*] [l*]; i j [k] [l] -> i* j* i j
        temp = backend.tensordot(temp, dU, ([0, 1], [0, 1])) # [i*] [j*] i j; [i] [j] k l -> i j k l
        temp_result -= temp
        # Compute final result
        result = -(result + temp_result) / 2 / self.cost
        # Project to tangent space and return
        U = self.U.reshape(self.D1*self.D2, -1)
        result = result.reshape(U.shape)
        return (result - 0.5 * U@(U.conj().T@result + result.conj().T@U)).reshape(self.D1, self.D2, self.D1, self.D2)

class TruncErrorIterateApproxTRM(TruncErrorIterate):
    """
    Class representing a single iterate of the approximate trunc error trust region optimizer.
    Cashes important results such that they do not need to be computed multiple times.
    This is especially useful when multiple hessian vector products must be computed.
    """

    def __init__(self, U, theta, chi, chi_max, N_iters_svd=None, eps_svd=0.0, old_iterate=None):
        """
        Initializes new iterate.

        Parameters
        ----------
        See class TruncErrorIterate.
        """
        TruncErrorIterate.__init__(self, U, theta, chi, chi_max=chi_max, N_iters_svd=N_iters_svd, eps_svd=eps_svd, old_iterate=old_iterate)
        self.computed_gradient = False
        self.computed_hessian = False

    def evaluate_cost_function(self):
        """
        Computes the truncation error sqrt(1 - sum_{i=1}^{chi} s_i**2) with s_i the ith singular value of U@theta.

        Returns
        -------
        cost : float
            value of the cost function
        """
        # Compute Utheta
        Utheta = backend.dot(backend.ascontiguousarray(self.U).reshape((self.D1*self.D2, -1)), backend.ascontiguousarray(self.theta.transpose(1, 2, 0, 3)).reshape(self.D1*self.D2, -1)) # l, i, j, r -> i, j, l, r -> (i, j), (l, r)
        Utheta = backend.ascontiguousarray(backend.ascontiguousarray(Utheta).reshape((self.D1, self.D2, self.l, self.r)).transpose((2, 0, 1, 3))).reshape((self.l*self.D1, -1)) # (i, j), (l, r) -> i, j, l, r -> l, i, j, r -> (l, i), (j, r)
        # Renormalize (might not be normalized due to numerical errors)
        Utheta /= backend.norm(Utheta)
        # Perform SVD
        if self.N_iters_svd is None or self.chi_max is None:
            self.X, self.S, self.Y = utility.safe_svd(Utheta, full_matrices=False) # { D^9 }
            if self.chi_max is not None:
                idx = backend.argsort(self.S)[::-1][:self.chi_max]
                self.X, self.S, self.Y = self.X[:, idx], self.S[idx], self.Y[idx, :]
        else:
            self.X, self.Y, _, _ = utility.split_matrix_iterate_QR(Utheta, self.chi_max, self.N_iters_svd, self.eps_svd, C0=self.Y0, normalize=False) # { N_iters_svd * D^7 }
            self.Y0 = self.Y
            XX, self.S, self.Y = backend.svd(self.Y, full_matrices=False) # { D^5 }
            self.X = self.X@XX
        self.Y = backend.conj(self.Y.T)
        if self.chi_max is None or self.chi < self.chi_max:
            self.X_trunc, self.S_trunc, self.Y_trunc = self.X[:, :self.chi], self.S[:self.chi], self.Y[:, :self.chi]
        else:
            self.X_trunc, self.S_trunc, self.Y_trunc = self.X, self.S, self.Y
        if self.chi >= min(self.l*self.D1, self.D2*self.r):
            self.cost = 0.0
            return self.cost
        else:
            # Compute cost function.
            # 1 - backend.sum(self.S**2) should be larger than zero, but can be lower than zero due to numerical errors.
            self.cost = backend.sqrt(max(1 - backend.sum(self.S_trunc**2), 0.0))
        return self.cost

    def compute_gradient(self):
        """
        Computes the approximate gradient of the truncation error cost function, projected to the tangent space of the iterate.

        Returns
        -------
        grad : backend.array_type of shape (i, j, i*, j*)
            gradient of the cost function
        """
        if not self.computed_gradient:
            self.Xtheta = backend.tensordot(self.X.reshape(self.l, self.D1, -1), backend.conj(self.theta), ([0], [0])) # [l] i chi; [l*] k* l* r* -> i chi k* l* r* { D^8 }
            self.XStheta = backend.tensordot(self.Xtheta[:, :self.chi, :, :, :], backend.diag(self.S_trunc), ([1], [0])) # i [chi] k* l* r*; [chi] chi -> i k* l* r* chi { D^7 }
            self.XSYtheta = backend.tensordot(self.XStheta, backend.conj(self.Y_trunc.reshape(self.D2, self.r, -1)), ([3, 4], [1, 2])) # i k* l* [r*] [chi]; j [r] [chi] -> i k* l* j  { D^7 }
            self.XSYtheta = self.XSYtheta.transpose(0, 3, 1, 2) # i, k*, l*, j -> i, j, k*, l*
            self.XSYthetaUU = backend.tensordot(backend.conj(self.XSYtheta), self.U, ([2, 3], [2, 3])) # i' j' [k'] [l']; i j [k'] [l'] -> i' j' i j { D^6 }
            self.XSYthetaUU = backend.tensordot(self.XSYthetaUU, self.U, ([0, 1], [0, 1])) # [i'] [j'] i j; [i'] [j'] k l -> i j k l { D^6 }
            self.computed_gradient = True
        if self.cost == 0:
            # Avoid dividing by zero
            return backend.zeros(self.U.shape)
        else:
            return - (self.XSYtheta - self.XSYthetaUU) / 2 / self.cost

    def compute_hessian_vector_product(self, dU):
        """
        Computes the approximate riemannian hessian vector product of dU, which is the projection of D(grad f(U))[dU] to the tangent space of U. 
        When this is called for the first time, several intermediate results are cached for the case that this is called again in the future.

        Parameters
        ----------
        dU : backend.array_type of shape (i, j, i*, j*)
            element from the tangent space of U

        Returns
        -------
        hvp : backend.array_type of shape (i, j, i*, j*)
            hessian vector product H@dU
        """
        if self.cost == 0:
            # Avoid dividing by zero
            return backend.zeros(self.U.shape)
        if not self.computed_hessian:
            # Cache helper results
            self.F = backend.zeros((self.k, self.k)) # { D^2 }
            for i in range(self.k):
                for j in range(self.k):
                    if i != j:
                        temp = self.S[j]**2 - self.S[i]**2
                        if temp == 0.0:
                            self.F[i, j] = 0.0
                        else:
                            self.F[i, j] = 1/temp
            self.XthetaY = backend.tensordot(self.Xtheta[:, :self.chi, :, :, :], backend.conj(self.Y_trunc.reshape(self.D2, self.r, -1)), ([4], [1])) # i chi k* l* [r*]; j* [r*] chi* -> i chi k* l* j* chi* { D^8 }
            self.XthetaY = self.XthetaY.transpose(0, 4, 2, 3, 1, 5) # i, chi, k*, l*, j*, chi* -> i, j, k, l, chi, chi* { D^6 }
            self.thetaY = backend.conj(backend.tensordot(self.theta, self.Y.reshape(self.D2, self.r, -1), ([3], [1]))) # l* k* l* [r*]; j* [r*] chi* -> l* k* l* j* chi* { D^8 }
            self.thetaY = self.thetaY.transpose(0, 3, 1, 2, 4) # l*, k*, l*, j*, chi* -> l*, j*, k*, l*, chi* { D^6 }
            # Compute modified inverse of S
            self.invS = backend.zeros(self.S.size, dtype=self.S.dtype)
            S_nonzero = backend.where(self.S != 0.0)
            self.invS[S_nonzero] = 1/self.S[S_nonzero]
            # Compute ImXXT and InYYT
            self.ImXXT = backend.eye(self.X.shape[0]) - self.X@backend.conj(self.X.T) # { D^7 }
            self.ImXXT = self.ImXXT.reshape(self.l, self.D1, self.l, self.D1) # { D^6 }
            self.InYYT = backend.eye(self.Y.shape[0]) - self.Y@backend.conj(self.Y.T) # { D^7 }
            self.InYYT = self.InYYT.reshape(self.D2, self.r, self.D2, self.r) # { D^6 }
            self.computed_hessian = True
        # Compute dP and dD
        temp = backend.tensordot(dU, backend.conj(self.thetaY), ([1, 2, 3], [1, 2, 3])) # i [j] [k] [l]; l* [j*] [k*] [l*] chi* -> i l* chi* { D^7 }
        dP = backend.tensordot(backend.conj(self.X).reshape(self.l, self.D1, -1), temp, ([0, 1], [1, 0])) # [l*] [i*] chi; [i] [l*] chi* -> chi chi* { D^5 } 
        dD = backend.diag(1.j*(backend.imag(backend.diag(dP)) * self.invS / 2))
        temp = backend.tensordot(self.ImXXT, temp, ([2, 3], [1, 0])) * self.invS # l i [l] [i]; [i] [l*] chi* -> l i chi* { D^7 }
        temp = temp.reshape(self.X.shape)
        # Compute dX, dS and dY
        dX = self.X@(self.F * (dP*self.S + self.S[:, backend.newaxis]*backend.conj(dP.T)) + dD) + temp # { D^4 }
        dS = backend.real(backend.diag(dP))[:self.chi]
        temp = backend.tensordot(self.Xtheta, backend.conj(dU), ([0, 2, 3], [0, 2, 3])) # [i] chi [k*] [l*] r*; [i] j [k] [l] -> chi r* j { D^7 }
        temp = backend.tensordot(self.InYYT, temp, ([2, 3], [2, 1])) * self.invS # j r [j] [r]; chi [r*] [j] -> j r chi { D^7 }
        temp = temp.reshape(self.Y.shape)
        dY = self.Y@(self.F * (self.S[:, backend.newaxis]*dP + backend.conj(dP.T)*self.S) - dD) + temp # { D^4 }
        # Compute first of five terms
        result = backend.sum(self.S_trunc * dS) * (self.XSYtheta - self.XSYthetaUU) / self.cost**2 # { D^4 }
        # Compute second of five terms
        temp = backend.tensordot(self.XthetaY, backend.diag(dS), ([4, 5], [0, 1])) # i j k l [chi] [chi]; [chi] [chi] -> i j k l { D^6 }
        temp2 = backend.tensordot(self.thetaY[:, :, :, :, :self.chi], backend.diag(self.S_trunc), ([4], [1])) # l* j* k* l* [chi*]; chi [chi] -> l j k l chi { D^6 }
        temp += backend.tensordot(dX.reshape(self.l, self.D1, -1)[:, :, :self.chi], temp2, ([0, 2], [0, 4])) # [l] i [chi]; [l] j k l [chi] -> i j k l { D^7 }
        temp2 = backend.tensordot(self.Xtheta[:, :self.chi, :, :, :], backend.diag(self.S_trunc), ([1], [0])) # i [chi] k* l* r*; [chi] chi -> i k l r chi { D^6 }
        temp += backend.tensordot(temp2, backend.conj(dY.reshape(self.D2, self.r, -1))[:, :, :self.chi], ([3, 4], [1, 2])).transpose(0, 3, 1, 2) # i k l [r] [chi]; j [r] [chi] -> i k l j -> i j k l { D^7 }
        result += temp
        # Compute third of five terms
        temp = backend.tensordot(backend.conj(temp), self.U, ([2, 3], [2, 3])) # i j [k] [l]; i j [k] [l] -> i j i j { D^6 }
        result -= backend.tensordot(temp, self.U, ([0, 1], [0, 1])) # [i] [j] i j; [i] [j] k l -> i j k l { D^6 }
        # Compute fourth of five terms
        temp = backend.tensordot(backend.conj(self.XSYtheta), dU, ([2, 3], [2, 3])) # i j [k] [l]; i j [k] [l] -> i j i j { D^6 }
        result -= backend.tensordot(temp, self.U, ([0, 1], [0, 1])) # [i] [j] i j; [i] [j] k l -> i j k l { D^6 }
        # Compute fifth of five terms
        temp = backend.tensordot(backend.conj(self.XSYtheta), self.U, ([2, 3], [2, 3])) # i j [k] [l]; i j [k] [l] -> i j i j { D^6 }
        result -= backend.tensordot(temp, dU, ([0, 1], [0, 1])) # [i] [j] i j; [i] [j] k l -> i j k l { D^6 }
        # Compute final result
        result = (-result / 2 / self.cost).reshape(self.D1*self.D2, -1)
        # Project to tangent space and return
        U = self.U.reshape(result.shape)
        return (result - 0.5 * U@(U.conj().T@result + result.conj().T@U)).reshape(self.D1, self.D2, self.D1, self.D2)