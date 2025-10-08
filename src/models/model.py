import scipy.sparse
import scipy.sparse.linalg
from ..utility import utility

def compute_groundstate_info_exact(Hs, L, observables):
    """
    Computes the exact ground state, its energy, and the expectation value of the given local observables in the 
    ground state from the given Hamiltonian or list of Hamiltonians using exact diagonalization.

    Parameters
    ----------
    Hs : backend.array_type of shape (n, n) or list of backend.array_type of shape (n, n)
        Hamiltonian or list of Hamiltonians. The dimension is n = d**L,
        with the local dimension d
    L : int
        number of local systems (sites)
    observables : list of backend.array_type of shape (d, d)
        list of local observables
    
    Returns
    -------
    Es : float or list of float
        energy of the ground state (or list of energies of the ground states)
    psis : backend.array_type of shape (n,) or list of backend.array_type of shape (n,)
        ground state (or list of ground states)
    expectation_values : list
        list containing the per site expectation values of the ground state
    """
    if type(Hs) is list:
        Es = []
        psis = []
        expectation_values = []
        for H in Hs:
            E, psi, expectation_value = compute_groundstate_info_exact(H, L, observables)
            Es.append(E)
            psis.append(psi)
            expectation_values.append(expectation_value)
        return Es, psis, expectation_values
    else:
        psi = scipy.sparse.linalg.eigsh(Hs, 1, which='SA')[1][:, 0]
        E = ((psi.T@Hs@psi)/(psi.T@psi)).item()
        expectation_value = []
        for op in observables:
            expectation_value.append(utility.average_site_expectation_value(L, psi, op))
        return E, psi, expectation_value

class Model:
    """
    Base class of all models
    """

    def get_description(self):
        """
        Returns a human readable description of the model
        """
        raise NotImplementedError()

    def compute_H_bonds_1D(self, L):
        """
        Computes the Hamiltonian as a list of 2D bond operators for a 1D chain.
        """
        raise NotImplementedError()

    def compute_H_1D(self, L):
        """
        Computes the Hamiltonian as a full matrix for a 1D chain.
        """
        raise NotImplementedError()

    def compute_H_bonds_2D_Square(self, Lx, Ly):
        """
        Computes the Hamiltonian as a list of 2D bond operators for a 2D square lattice.
        """
        raise NotImplementedError()

    def compute_H_2D_Square(self, Lx, Ly):
        """
        Computes the Hamiltonian as a full matrix for a 2D square lattice.
        """
        raise NotImplementedError()

    def compute_H_bonds_2D_Honeycomb(self, Lx, Ly):
        """
        Computes the Hamiltonian as a list of 2D bond operators for a 2D honeycomb lattice.
        """
        raise NotImplementedError()

    def compute_H_2D_Honeycomb(self, Lx, Ly):
        """
        Computes the Hamiltonian as a full matrix for a 2D honeycomb lattice.
        """
        raise NotImplementedError()

    def compute_groundstate_info_exact_1D(self, L, observables):
        """
        Computes the numerically exact ground state information of this model on a 1D chain
        """
        H = self.compute_H_1D(L)
        return compute_groundstate_info_exact(H, L, observables)

    def compute_groundstate_info_exact_2D_Square(self, Lx, Ly, observables):
        """
        Computes the numerically exact ground state information of this model on a 2D Square lattice
        """
        H = self.compute_H_2D_Square(Lx, Ly)
        return compute_groundstate_info_exact(H, 2*Lx*Ly, observables)

    def compute_groundstate_info_exact_2D_Honeycomb(self, Lx, Ly, observables):
        """
        Computes the numerically exact ground state information of this model on a 2D Honeycomb lattice
        """
        H = self.compute_H_2D_Honeycomb(Lx, Ly)
        return compute_groundstate_info_exact(H, 2*Lx*Ly, observables)