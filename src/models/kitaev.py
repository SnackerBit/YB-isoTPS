import scipy
from . import model
from ..utility import utility
from ..utility import backend

class Kitaev(model.Model):
    """
    The Kitaev Honeycomb model
    """

    sigma_x = backend.array([[0, 1], [1, 0]])
    sigma_y = backend.array([[0, -1.j], [1.j, 0]])
    sigma_z = backend.array([[1, 0], [0, -1]])
    eye = backend.array([[1, 0], [0, 1]])

    def __init__(self, Kx, Ky, Kz):
        self.Kx = float(Kx)
        self.Ky = float(Ky)
        self.Kz = float(Kz)

    def get_description(self):
        """
        Returns a human readable description of the model
        """
        return {
            "modelType" : "Kitaev",
            "Kx" : self.Kx,
            "Ky" : self.Ky,
            "Kz" : self.Kz
        }

    def compute_H_bonds_2D_Honeycomb(self, Lx, Ly):
        """
        Computes the Hamiltonian as a list of 2D bond operators for a 2D honeycomb lattice.
        """
        xx = - backend.kron(self.sigma_x, self.sigma_x).reshape(2, 2, 2, 2)
        yy = - backend.kron(self.sigma_x, self.sigma_x).reshape(2, 2, 2, 2)
        zz = - backend.kron(self.sigma_x, self.sigma_x).reshape(2, 2, 2, 2)
        H_bonds = []
        for i in range(2 * Lx - 1):
            if i%2 == 0:
                for j in range(2 * Ly - 1):
                    if j%2 == 0:
                        H_bonds.append(xx.copy())
                    else:
                        H_bonds.append(yy.copy())
            else:
                for j in range(2 * Ly - 1):
                    if j%2 == 0:
                        H_bonds.append(zz.copy())
                    else:
                        H_bonds.append(None)
        return H_bonds