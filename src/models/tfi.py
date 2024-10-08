import numpy as np
import scipy
from . import model
from ..utility import utility

class TFI(model.Model):
    """
    Transverse Field Ising model
    """

    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1.j], [1.j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    eye = np.array([[1, 0], [0, 1]])

    def __init__(self, g, J):
        self.g = float(g)
        self.J = float(J)

    def get_description(self):
        """
        Returns a human readable description of the model

        Returns
        -------
        model : dict
            description of the model
        """
        return {
            "modelType" : "TFI",
            "g" : self.g,
            "J" : self.J
        }

    def compute_H_bonds_1D(self, L):
        """
        Computes the Hamiltonian as a list of 2D bond operators for a 1D chain.
        """
        H_bonds = []
        for i in range(L - 1):
            gL = gR = 0.5 * self.g
            if i == 0: # first bond
                gL = self.g
            if i+1 == L-1: # last bond
                gR = self.g
            H_bond = -self.J * np.kron(self.sigma_x, self.sigma_x) - gL * np.kron(self.sigma_z, self.eye) - gR * np.kron(self.eye, self.sigma_z)
            H_bonds.append(np.reshape(H_bond, (2, 2, 2, 2)))
        return H_bonds

    def compute_H_1D(self, L):
        """
        Computes the Hamiltonian as a full matrix for a 1D chain.
        """
        sx_list = utility.compute_op_list(L, self.sigma_x)
        sz_list = utility.compute_op_list(L, self.sigma_z)

        H = sz_list[0]
        for j in range(1, L):
            H += sz_list[j]
        H = -self.g*H
        if L > 1:
            interaction = sx_list[0] @ sx_list[1]
            for j in range(1, L-1):
                interaction += sx_list[j] @ sx_list[j+1]
            H += -self.J * interaction
        return H

    def compute_H_bonds_2D_Square(self, Lx, Ly):
        """
        Computes the Hamiltonian as a list of 2D bond operators for a 2D square lattice.
        """
        # Returns the number of D connections to neighbouring tensors
        def count_connections_Square(x, y, p):
            #print((x, y, p))
            if p == 0:
                if x == 0:
                    if y == 0:
                        return 1
                    else:
                        return 2
                elif y == 0:
                    return 2
                else:
                    return 4
            else: # p == 1
                if x == Lx - 1:
                    if y == Ly - 1:
                        return 1
                    else:
                        return 2
                elif y == Ly - 1:
                    return 2
                else:
                    return 4
        # Creates H_bond with the number of connections of the first and second site
        def create_H_bond(connections_left, connections_right):
            #print(connections_left, connections_right)
            gL = self.g / connections_left
            gR = self.g / connections_right
            H_bond = -self.J * np.kron(self.sigma_x, self.sigma_x) - gL * np.kron(self.sigma_z, self.eye) - gR * np.kron(self.eye, self.sigma_z)
            return np.reshape(H_bond, (2, 2, 2, 2))
        H_bonds = []
        for i in range(2 * Lx - 1):
            if i%2 == 0:
                for j in range(2 * Ly - 1):
                    if j%2 == 0:
                        H_bonds.append(create_H_bond(count_connections_Square(i//2, j//2, 0), count_connections_Square(i//2, j//2, 1)))
                    else:
                        H_bonds.append(create_H_bond(count_connections_Square(i//2, j//2, 1), count_connections_Square(i//2, j//2+1, 0)))
            else:
                for j in range(2 * Ly - 1):
                    if j%2 == 0:
                        H_bonds.append(create_H_bond(count_connections_Square(i//2+1, j//2, 0), count_connections_Square(i//2, j//2, 1)))
                    else:
                        H_bonds.append(create_H_bond(count_connections_Square(i//2, j//2, 1), count_connections_Square(i//2+1, j//2+1, 0)))
        return H_bonds

    def compute_H_2D_Square(self, Lx, Ly):
        """
        Computes the Hamiltonian as a list of 2D bond operators for a 2D square lattice.
        """
        sx_list = utility.compute_op_list(2*Lx*Ly, self.sigma_x)
        sz_list = utility.compute_op_list(2*Lx*Ly, self.sigma_z)
        D = 2**(2*Lx*Ly)
        H = np.zeros((D, D))
        # Add the sigma_z contributions
        for i in range(2*Lx*Ly):
            H -= self.g * sz_list[i]
        # Add the sigma_x contributions
        def compute_bond_contribution(x1, y1, p1, x2, y2, p2):
            return sx_list[(x1 * Ly + y1) * 2 + p1] @ sx_list[(x2 * Ly + y2) * 2 + p2]
        for i in range(2 * Lx - 1):
            if i%2 == 0:
                for j in range(2 * Ly - 1):
                    if j%2 == 0:
                        H -= self.J * compute_bond_contribution(i//2, j//2, 0, i//2, j//2, 1)
                    else:
                        H -= self.J * compute_bond_contribution(i//2, j//2, 1, i//2, j//2+1, 0)
            else:
                for j in range(2 * Ly - 1):
                    if j%2 == 0:
                        H -= self.J * compute_bond_contribution(i//2+1, j//2, 0, i//2, j//2, 1)
                    else:
                        H -= self.J * compute_bond_contribution(i//2, j//2, 1, i//2+1, j//2+1, 0)
        return H

    def compute_H_bonds_2D_Honeycomb(self, Lx, Ly):
        """
        Computes the Hamiltonian as a list of 2D bond operators for a 2D honeycomb lattice.
        """
        # Returns the number of D connections to neighbouring tensors
        def count_connections_Honeycomb(x, y, p):
            #print((x, y, p))
            if p == 0:
                if x == 0:
                    if y == 0:
                        return 1
                    else:
                        return 2
                elif y == 0:
                    return 2
                else:
                    return 3
            else: # p == 1
                if x == Lx - 1:
                    if y == Ly - 1:
                        return 1
                    else:
                        return 2
                elif y == Ly - 1:
                    return 2
                else:
                    return 3
        # Creates H_bond with the number of connections of the first and second site
        def create_H_bond(connections_left, connections_right):
            gL = self.g / connections_left
            gR = self.g / connections_right
            #print(f"Creating bond op with {(connections_left, connections_right)} connections, g = {(gL, gR)}")
            H_bond = -self.J * np.kron(self.sigma_x, self.sigma_x) - gL * np.kron(self.sigma_z, self.eye) - gR * np.kron(self.eye, self.sigma_z)
            return np.reshape(H_bond, (2, 2, 2, 2))
        H_bonds = []
        for i in range(2 * Lx - 1):
            if i%2 == 0:
                for j in range(2 * Ly - 1):
                    if j%2 == 0:
                        H_bonds.append(create_H_bond(count_connections_Honeycomb(i//2, j//2, 0), count_connections_Honeycomb(i//2, j//2, 1)))
                    else:
                        H_bonds.append(create_H_bond(count_connections_Honeycomb(i//2, j//2, 1), count_connections_Honeycomb(i//2, j//2+1, 0)))
            else:
                for j in range(2 * Ly - 1):
                    if j%2 == 0:
                        H_bonds.append(create_H_bond(count_connections_Honeycomb(i//2, j//2, 1), count_connections_Honeycomb(i//2+1, j//2, 0)))
                    else:
                        H_bonds.append(None)
        return H_bonds

    def compute_H_2D_Honeycomb(self, Lx, Ly):
        """
        Computes the Hamiltonian as a full matrix for a 2D honeycomb lattice.
        """
        sx_list = utility.compute_op_list(2*Lx*Ly, self.sigma_x)
        sz_list = utility.compute_op_list(2*Lx*Ly, self.sigma_z)
        D = 2**(2*Lx*Ly)
        H = scipy.sparse.csr_matrix((D, D))
        # Add the sigma_z contributions
        for i in range(2*Lx*Ly):
            H -= self.g * sz_list[i]
        # Add the sigma_x contributions
        def compute_bond_contribution(x1, y1, p1, x2, y2, p2):
            return sx_list[(x1 * Ly + y1) * 2 + p1] @ sx_list[(x2 * Ly + y2) * 2 + p2]

        for i in range(2 * Lx - 1):
            if i%2 == 0:
                for j in range(2 * Ly - 1):
                    if j%2 == 0:
                        H -= self.J * compute_bond_contribution(i//2, j//2, 0, i//2, j//2, 1)
                    else:
                        H -= self.J * compute_bond_contribution(i//2, j//2, 1, i//2, j//2+1, 0)
            else:
                for j in range(0, 2 * Ly - 1, 2):
                    H -= self.J * compute_bond_contribution(i//2, j//2, 1, i//2+1, j//2, 0)
        return H