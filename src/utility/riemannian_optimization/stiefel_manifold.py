from .. import backend

class ComplexStiefelManifold:
    """
    Class implementing the complex stiefel manifold of isometric matrices for Riemannian optimization.
    Sources:
    - P.-A. Absil, Robert Mahony, Rodolphe Sepulchre: "Optimization Algorithms on Matrix Manifolds", https://press.princeton.edu/absil
    - Markus Hauru, Maarten Van Damme, Jutho Haegeman: "Riemannian optimization of isometric tensor networks", https://scipost.org/10.21468/SciPostPhys.10.2.040
    - James Townsend, Niklas Koep, Sebastian Weichwald: "Pymanopt: A Python Toolbox for Optimization on Manifolds using Automatic Differentiation", https://arxiv.org/abs/1603.03236
    """

    def __init__(self, n, p, shape=None):
        """
        Initializes the class.

        Parameters
        ----------
        n, p : int
            shape of the isometries in the manifold: (n, p). It must hold n >= p.
        shape : Tuple of int, arbitrary length, optional
            the actual shape of the tensors we are representing as isometries in this manifold.
            The legs must already be transposed such that reshaping shape into (n, p) is possible.
            If this is None, the default shape (n, m) is used. Default: None.
        """
        assert(n >= p)
        self.n = n
        self.p = p
        if shape is None:
            self.shape = (n, p)
        else:
            self.shape = shape

    def retract(self, x, xi=None):
        """
        Retracts x + xi onto the manifold using the qr decomposition. Extra steps are taken to ensure
        uniqueness of the qr decomposition.
        
        Parameters
        ----------
        x : backend.array_type of shape self.shape, or reshapable into shape (n, m)
            element from the embedding space.
        xi : backend.array_type of shape self.shape, or reshapable into shape (n, m)
            element from the embedding space.
        
        Returns
        -------
        result : backend.array_type of shape self.shape
            the retracted isometric tensor
        """
        x_copy = x.copy()
        temp = x.reshape((self.n, self.p))
        if xi is not None:
            temp = temp + xi.reshape((self.n, self.p))
        assert(backend.allclose(x, x_copy))
        Q, R = backend.qr(temp)
        # Ensure uniqueness of QR decomposition by flipping signs of rows such that
        # all diagonal elements of R are positive
        P = backend.diag(backend.sign(backend.diag(R)))
        return backend.reshape(Q@P, self.shape)

    def project_to_tangent_space(self, x, xi):
        """
        Projects xi to the tangent space of x.

        Parameters
        ----------
        x : backend.array_type of shape self.shape, or reshapable into shape (n, m)
            element of the complex stiefel manifold. Must be an isometry.
            This is not explicitly checked by this function for performance reasons.
        xi : backend.array_type of shape self.shape, or reshapable into shape (n, m)
            element of the embedding space.
        
        Returns
        -------
        result : backend.array_type of shape self.shape
            the tangent vector obtained by projecting xi to the tangent space of x
        """
        temp = x.reshape((self.n, self.p))
        xi = xi.reshape((self.n, self.p))
        return backend.reshape(xi - 0.5 * temp@(backend.conj(temp).T@xi + backend.conj(xi).T@temp), self.shape)

    def inner_product(self, x, y):
        """
        Computes the inner product of two elements from the same tangent space.

        Parameters
        ----------
        x : backend.array_type of shape self.shape, or reshapable into shape (n, m)
            first tangent vector
        y : backend.array_type of shape self.shape, or reshapable into shape (n, m)
            second tangent vector

        Returns
        -------
        result : float
            the inner product of x and y
        """
        return backend.real(backend.tensordot(x.conj(), y, axes=x.ndim))

    def norm(self, x):
        """
        Computes the norm of a tangent vector x, given by the square root of the inner product of x with itself.

        Parameters
        ----------
        x : backend.array_type of shape self.shape, or reshapable into shape (n, m)
            tangent vector

        Returns
        -------
        result : float
            the norm 
        """
        return backend.norm(x)
    
    def transport(self, x, xi):
        """
        Transports a tangent vector xi to the tangent space of x.

        Parameters
        ----------
        x : backend.array_type of shape self.shape, or reshapable into shape (n, m)
            element of the complex stiefel manifold. Must be an isometry.
            This is not explicitly checked by this function for performance reasons.
        xi : backend.array_type of shape self.shape, or reshapable into shape (n, m)
            element of the tangent space at an arbitrary element of the manifold.

        Returns
        -------
        result : backend.array_type of shape self.shape
            the tangent vector transported to the tangent space of x
        """
        return self.project_to_tangent_space(x, xi)

    def zero_vector(self):
        """
        Returns the zero tangent vector of any tangent space. Needed for the initialization of some algorithms.

        Returns
        -------
        result : backend.array_type of shape self.shape
            zero tangent vector
        """
        return backend.zeros(self.shape, dtype=backend.dtype_complex)