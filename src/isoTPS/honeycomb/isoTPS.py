import numpy as np
import time
import matplotlib.pyplot as plt
from ...utility import utility
from ...utility import debug_logging
from . import yang_baxter_move
from . import expectation_values
from . import tebd
from .. import isoTPS
from .. import shifting_ortho_center

class isoTPS_Honeycomb(isoTPS.isoTPS):

    def __init__(self, Lx, Ly, D_max=4, D_max_horizontal=None, chi_factor=6, chi_max=24, d=2, shifting_options={ "mode" : "svd" }, yb_options={ "mode" : "svd" }, tebd_options={ "mode" : "svd" }, ordering_mode="edges", debug_level=None): # TODO: FIX debug logging
        """
        Initializes the isoTPS.

        Parameters
        ----------
        Lx, Ly : int
            system size
        D_max : int, optional
            maximal bond dimension of the virtual legs of the physical site tensors ("T-tensors"). Default value: 4.
        D_max_horizontal : int or None, optional
            maximal bond dimension of the horizontal legs of the physical site tensors ("T-tensors"). It may be good to choose a slightly
            larger maximal bond dimension for horizontal legs, since at each intersection two diagonal legs lead into one horizontal leg.
            If this parameter is set to None, horizontal legs have the same maximal bond dimension as diagonal legs. Default : None.
        chi_factor : int or None, optional
            factor with which the maximal bond dimension of the orthogonality hypersurface is computed, chi_max = D_max*chi_factor.
            If this is set to None, the maximal bond dimension of the orthogonality hypersurface is set to the parameter chi_max instead.
            Default: chi_factor = 6.
        chi_max : int, optional
            maximal bond dimension of the orthogonality hypersurface. Is only used if chi_factor is None. Default: 24.
        d : int, optional
            size of the local Hilbert space at each lattice site, e.g. d=2 for spin-1/2 models. Default: 2.
        shifting_options : dict, optional
            options for moving the orthogonality center along the orthogonality hypersurface. See "src/isoTPS/shifting_ortho_center.py"
            for more information. Default: { "mode" : "svd" }.
        yb_options : dict, optional
            options for performing the Yang-Baxter move (moving the orthogonality hypersurface through the lattice). See
            "src/isoTPS/square/yang_baxter_move.py" or src/isoTPS/honeycomb/yang_baxter_move.py" for more information.
            Default: { "mode" : "svd" }.
        tebd_options : dict, optional
            options for performing TEBD. See "src/isoTPS/honeycomb/tebd.py" for more information.
            Default: { "mode" : "svd" }.
        ordering_mode : str, one of {"up", "down", "center", "edges"}, optional
            string specifying the rule for choosing how bond dimension is distributed between the bonds when performing Yang-Baxter moves.
            "up": Larger bond dimension is always chosen for the upper bond. "down": larger bond dimension is always chosen for the lower bond.
            "center": Larger bond dimension is chosen for the upper/lower bond depending on which is closer to the center. "edges": larger bond
            dimension is chosen for the upper bond if we are left of the orthogonality hypersurface, and for the lower bond if we are righ of
            the orthogonality surface. Default : "edges".
        debug_level : enum class DebugLevel, optional
            the debug level decides which information is logged and if warnings/errors are printed to the console. See "utility/debug_levels.py"
            for more information. Default: DebugLevel.NO_DEBUG.
        """
        super().__init__(Lx, Ly, D_max=D_max, chi_factor=chi_factor, chi_max=chi_max, d=d, shifting_options=shifting_options, yb_options=yb_options, tebd_options=tebd_options, ordering_mode=ordering_mode, debug_level=debug_level)
        if D_max_horizontal is None:
            D_max_horizontal = D_max
        self.D_max_horizontal = D_max_horizontal

    def save_to_file(self, filename):
        """
        Saves this isoTPS to a file.
        
        Parameters
        ----------
        filename : str
            file name
        """
        data = { "D_max_horizontal" : self.D_max_horizontal }
        super().save_to_file(filename, data)

    @staticmethod
    def load_from_file(filename):
        tps = isoTPS_Honeycomb(0, 0)
        with h5py.File(filename, "r") as hf:
            data = utility.load_dict_from_hf(hf)
            tps._load_from_dict(data)
        return tps
        
    def copy(self):
        """
        Returns a copy of this isoTPS

        Returns
        -------
        copy : instance if class isoTPS_Honeycomb
            the copied isoTPS
        """
        result = isoTPS_Honeycomb(self.Lx, self.Ly, D_max=self.D_max, D_max_horizontal=self.D_max_horizontal, chi_factor=self.chi_factor, chi_max=self.chi_max, d=self.d, shifting_options=self.shifting_options, yb_options=self.yb_options, tebd_options=self.tebd_options, ordering_mode=self.ordering_mode, debug_level=self.debug_level)
        result._init_as_copy(self)
        return result

    def initialize_product_state(self, states):
        """
        Initializes the isoTPS tensors in the given product state.

        Parameters
        ----------
        states : list of np.ndarray of shape (d,)
            list of local states. The full many-body state is formed by the kronecker product of all states in the list.
        """
        # Temporarily set debug level to zero to avoid printing unnecessary debug information during initialization
        debug_level = self.debug_level
        self.debug_level = debug_levels.DebugLevel.NO_DEBUG
        if self.debug_dict is not None:
            self.debug_dict["debug_level"] = self.debug_level
        def initialize_T_product_state(state=np.array([1.0, 0.0], dtype=np.complex128)):
            T = np.zeros((state.size, 1, 1, 1), dtype=np.complex128)
            T[:, 0, 0, 0] = state[:]
            return T
        # First, initialize the ortho center, which is to the right of all T tensors
        self.ortho_surface = 2 * self.Lx - 1
        for i in range(0, 2 * self.Ly - 1, 2):
            self.Ws[i] = np.array([[[[1.]]]], dtype=np.complex128)
        # We go from right to left, initializing the T tensors in product states and moving the ortho surface left
        for x in range(self.Lx - 1, -1, -1):
            for y in range(self.Ly):
                index = self.get_index(x, y, 1)
                self.Ts[index] = initialize_T_product_state(states[index])
                index = self.get_index(x, y, 0)
                self.Ts[index] = initialize_T_product_state(states[index])
            self.move_ortho_surface_left(force=True)
            self.move_ortho_surface_left(force=True)
        self.move_ortho_surface_right(force=True)
        assert(self.ortho_surface == 0)
        self.debug_level = debug_level
        if self.debug_dict is not None:
            self.debug_dict["debug_level"] = self.debug_level

    def plot(self, T_colors=None, ax=None, figsize_y=8):
        """
        Plots the isoTPS, labeling each leg with its respective bond dimension.
        
        Parameters
        ----------
        T_colors : list of colors, or None, optional
            colors of T tensors, one color per tensor. If this is None, the standard color is used.
            Default: None.
        ax : matplotlib.pyplot.axes or None, optional
            the axis in which the plot is drawn. If this is None, the standard axis is used (plt.gca()).
            Default: None.
        figsize_y : float, optional
            the size of the figure in y direction. Default: 8.0.
        """
        scale = figsize_y / (4 * self.Ly)
        if ax is None:
            fig, ax = plt.subplots(figsize=(2 * np.sqrt(3) * self.Lx * scale, 6 * self.Ly * scale))
        ax.axis("equal")
        #ax.set_xlim(-1, np.sqrt(3) * (self.Lx - 1 + 1/3) + 1)
        #ax.set_ylim(-1, 2 * (self.Ly - 1) + (self.Lx - 1) + 1.5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if T_colors is None:
            # Default color: bright blue
            T_colors = [np.array([31, 119, 180]) / 255] * self.Lx * self.Ly * 2

        # Helper function for drawing labelled arrows
        def construct_labelled_arrow(start, direction, label=None, labelPos="upper left", color="black"):
            dx_2 = direction[0] / 2
            dy_2 = direction[1] / 2
            delta_l = 0.025
            alpha = np.arctan2(dx_2, dy_2)
            ddx = np.sin(alpha) * delta_l
            ddy = np.cos(alpha) * delta_l
            ax.arrow(start[0], start[1], dx_2 + ddx, dy_2 + ddy, head_width=0.05, head_length=0.05, color=color)
            ax.arrow(start[0] + dx_2 + ddx, start[1] + dy_2 + ddy, dx_2 - ddx, dy_2 - ddy, head_width=0, color=color)
            if label is not None:
                labelCoords = (0, 0)
                if labelPos == "upper left":
                    labelCoords = (-5, 5)
                    ha = "right"
                elif labelPos == "upper right":
                    labelCoords = (5, 5)
                    ha = "left"
                elif labelPos == "lower left":
                    labelCoords = (-5, -5)
                    ha = "right"
                elif labelPos == "lower right":
                    labelCoords = (5, -5)
                    ha = "left"
                elif labelPos == "right":
                    labelCoords = (5, 0)
                    ha = "left"
                elif labelPos == "left":
                    ha = "right"
                    labelCoords = (-5, 0)
                elif labelPos == "up":
                    ha = "center"
                    labelCoords = (0, 5)
                elif labelPos == "down":
                    ha = "center"
                    labelCoords = (0, -5)
                ax.annotate(label, (start[0] + dx_2, start[1] + dy_2), xytext=labelCoords, textcoords='offset points', ha=ha)

        def compute_tensor_position_T(x, y, p):
            return (np.sqrt(3) * (x + p/3), 2 * y + x + p)

        def compute_tensor_position_W(x, y):
            p = x%2
            return (np.sqrt(3) * (x//2 + (0.5/3) + p * (1/2)), x/2 + y + 0.5)

        # Draw arrows starting from each p = 0 tensor connecting to p = 1 tensors
        for x in range(self.Lx):
            for y in range(self.Ly):
                position = compute_tensor_position_T(x, y, 0)
                _, ru, rd, l = self.Ts[self.get_index(x, y, 0)].shape
                # "l" leg
                if x > 0:
                    if 2*x > self.ortho_surface:
                        if self.ortho_surface != 2*x-1:
                            construct_labelled_arrow(position, (-2*np.sqrt(3)/3, 0), str(l), labelPos="up")
                    else:
                        construct_labelled_arrow((position[0] - 2*np.sqrt(3)/3, position[1]), (2*np.sqrt(3)/3, 0), str(l), labelPos="up")
                # "ru" leg
                if 2*x < self.ortho_surface:
                    construct_labelled_arrow(position, (np.sqrt(3)/3, 1), str(ru), labelPos="upper left")
                elif 2*x > self.ortho_surface:
                    construct_labelled_arrow((position[0] + np.sqrt(3)/3, position[1] + 1), (-np.sqrt(3)/3, -1), str(ru), labelPos="upper left")
                # "rd" leg
                if y > 0:
                    if 2*x < self.ortho_surface:
                        construct_labelled_arrow(position, (np.sqrt(3)/3, -1), str(rd), labelPos="upper right")
                    elif 2*x > self.ortho_surface:
                        construct_labelled_arrow((position[0] + np.sqrt(3)/3, position[1] - 1), (-np.sqrt(3)/3, 1), str(rd), labelPos="upper right")
                # arrow for physical index
                construct_labelled_arrow((position[0], position[1] + 0.4), (0, -0.4), str(self.Ts[self.get_index(x, y, 0)].shape[0]), labelPos="left", color="green")
                position = compute_tensor_position_T(x, y, 1)
                construct_labelled_arrow((position[0], position[1] + 0.4), (0, -0.4), str(self.Ts[self.get_index(x, y, 1)].shape[0]), labelPos="right", color="green")

        # Draw arrows for each tensor of the orthogonality surface
        if self.ortho_surface % 2 == 0:
            labelPos = "right"
            for y in range(2 * self.Ly - 1):
                if labelPos == "right":
                    labelPos = "left"
                else:
                    labelPos = "right"
                position = compute_tensor_position_W(self.ortho_surface, y)
                l, u, r, _ = self.Ws[y].shape
                # "u" leg
                if y < self.ortho_center:
                    construct_labelled_arrow(position, (0, 1), str(u), labelPos=labelPos, color="red")
                elif y < 2*self.Ly-2:
                    construct_labelled_arrow((position[0], position[1] + 1), (0, -1), str(u), labelPos=labelPos, color="red")
                # "l" leg
                if y%2 == 0:
                    construct_labelled_arrow((position[0]-1/2/np.sqrt(3), position[1] - 0.5), (1/2/np.sqrt(3), 0.5), str(l), labelPos="upper left")
                else:
                    construct_labelled_arrow((position[0]-1/2/np.sqrt(3), position[1] + 0.5), (1/2/np.sqrt(3), -0.5), str(l), labelPos="lower left")
                # "r" leg
                if y%2 == 0:
                    construct_labelled_arrow((position[0]+1/2/np.sqrt(3), position[1] + 0.5), (-1/2/np.sqrt(3), -0.5), str(r), labelPos="lower right")
                else:
                    construct_labelled_arrow((position[0]+1/2/np.sqrt(3), position[1] - 0.5), (-1/2/np.sqrt(3), 0.5), str(r), labelPos="upper right")
        else:
            for y in range(0, 2 * self.Ly - 1, 2):
                position = compute_tensor_position_W(self.ortho_surface, y)
                l, u, r, _ = self.Ws[y].shape
                # "u" leg
                if y < self.ortho_center:
                    construct_labelled_arrow(position, (0, 2), str(u), labelPos="right", color="red")
                elif y < 2*self.Ly-2:
                    construct_labelled_arrow((position[0], position[1] + 2), (0, -2), str(u), labelPos="right", color="red")
                # "l" leg
                construct_labelled_arrow((position[0] - 1/np.sqrt(3), position[1]), (1/np.sqrt(3), 0), str(l), labelPos="up")
                # "r" leg
                construct_labelled_arrow((position[0] + 1/np.sqrt(3), position[1]), (-1/np.sqrt(3), 0), str(r), labelPos="up")

        # Draw actual T tensors
        for y in range(self.Ly):
            for x in range(self.Lx):
                ax.add_patch(plt.Circle(compute_tensor_position_T(x, y, 0), 0.06, color=T_colors[self.get_index(x, y, 0)]))
                ax.add_patch(plt.Circle(compute_tensor_position_T(x, y, 1), 0.06, color=T_colors[self.get_index(x, y, 1)]))

        # Draw actual W tensors
        if self.ortho_surface % 2 == 0:
            for y in range(2 * self.Ly - 1):
                if y == self.ortho_center:
                    ax.add_patch(plt.Circle(compute_tensor_position_W(self.ortho_surface, y), 0.06, color="orange"))
                else:            
                    ax.add_patch(plt.Circle(compute_tensor_position_W(self.ortho_surface, y), 0.06, color="red"))
        else:
            for y in range(0, 2 * self.Ly - 1, 2):
                if y == self.ortho_center:
                    ax.add_patch(plt.Circle(compute_tensor_position_W(self.ortho_surface, y), 0.06, color="orange"))
                else:            
                    ax.add_patch(plt.Circle(compute_tensor_position_W(self.ortho_surface, y), 0.06, color="red"))

    def check_isometry_condition(self):
        """
        If the isometry condition is not satisfied at any tensor, an error
        message will be printed to the console and false is returned.
        If the isometry condition is fullfilled, True is returned instead.

        Returns
        -------
        success : bool
            wether the isoTPS fulfills the isometry condition
        """
        success = True
        # Check T tensors
        for x in range(self.Lx):
            for y in range(self.Ly):
                # p = 0
                T = self.Ts[self.get_index(x, y, 0)]
                if 2*x <= self.ortho_surface: # tensor is to the left of ortho surface
                    T = T.transpose(0, 3, 1, 2) # i, ru, rd, l -> i, l, ru, rd
                    T = T.reshape(T.shape[0]*T.shape[1], T.shape[2]*T.shape[3]) # i, l, ru, rd -> (i, l), (ru, rd)
                else: # tensor is to the right of ortho surface
                    T = T.reshape(T.shape[0]*T.shape[1]*T.shape[2], T.shape[3]) # i, ru, rd, l -> (i, ru, rd), l
                if not utility.check_isometry(T):
                    print(f"T tensor at (x={x}, y={y}, p={0}) is not an isometry ...")
                    success = False
                # p = 1
                T = self.Ts[self.get_index(x, y, 1)]
                if 2*x >= self.ortho_surface: # tensor is to the right of ortho surface
                    T = T.reshape(T.shape[0]*T.shape[1], T.shape[2]*T.shape[3]) # i, r, ld, lu -> (i, r), (ld, lu)
                else:
                    T = T.transpose(0, 2, 3, 1) # i, r, ld, lu -> i, ld, lu, r
                    T = T.reshape(T.shape[0]*T.shape[1]*T.shape[2], T.shape[3]) # i, ld, lu, r -> (i, ld, lu), r)
                if not utility.check_isometry(T):
                    print(f"T tensor at (x={x}, y={y}, p={1}) is not an isometry ...")
                    success = False
        # Check W tensors
        for i in range(0, len(self.Ws), 1 + self.ortho_surface%2):
            W = self.Ws[i]
            if i < self.ortho_center:
                W = np.transpose(W, (0, 2, 3, 1)) # l u r d -> l r d u
            elif i > self.ortho_center:
                W = np.transpose(W, (0, 2, 1, 3)) # l u r d -> l r u d
            else:
                continue
            W = np.reshape(W, (W.shape[0]*W.shape[1]*W.shape[2], W.shape[3]))
            if not utility.check_isometry(W):
                print(f"W tensor at index  {i} is not an isometry ...")
                success = False
        return success

    def check_leg_dimensions(self):
        """
        If the bond dimensions of all legs match up, True is returned.
        If an error is detected, it is printed to the console where the error
        happened and which bond dimensions were expected.

        Returns
        -------
        success : bool
            wether all leg dimensions in the isoTPS match up
        """
        success = True
        # Check all three bonds of p = 0 tensors
        for x in range(self.Lx):
            for y in range(self.Ly):
                _, ru, rd, l = self.Ts[self.get_index(x, y, 0)].shape
                # "ru" bond
                errorMessage = None
                if 2*x == self.ortho_surface:
                    if ru != self.Ws[2 * y].shape[0]:
                        errorMessage = f"Should match with Ws[{2 * y}].shape[0] = {self.Ws[2 * y].shape[0]}."
                else:
                    if ru != self.Ts[self.get_index(x, y, 1)].shape[2]:
                        errorMessage = f"Should match with Ts[({x}, {y}, {1})].shape[2] = {self.Ts[self.get_index(x, y, 1)].shape[2]}."
                if errorMessage is not None:
                    print(f"leg at Ts[({x}, {y}, {0})] in ru direction does not match bond dimension!", errorMessage)
                    success = False
                # "rd" bond
                errorMessage = None
                if y == 0:
                    if rd != 1:
                        errorMessage = "Should be 1."
                elif 2*x == self.ortho_surface:
                    if rd != self.Ws[2 * y - 1].shape[0]:
                        errorMessage = f"Should match with Ws[{2 * y - 1}].shape[0] = {self.Ws[2 * y - 1].shape[0]}."
                else:
                    if rd != self.Ts[self.get_index(x, y-1, 1)].shape[3]:
                        errorMessage = f"Should match with Ts[({x}, {y-1}, {1})].shape[3] = {self.Ts[self.get_index(x, y-1, 1)].shape[3]}."
                if errorMessage is not None:
                    print(f"leg at Ts[({x}, {y}, {0})] in rd direction does not match bond dimension!", errorMessage)
                    success = False
                # "l" bond
                errorMessage = None
                if x == 0:
                    if l != 1:
                        errorMessage = "Should be 1."
                elif 2*x-1 == self.ortho_surface:
                    if l != self.Ws[2 * y].shape[2]:
                        errorMessage = f"Should match with Ws[{2 * y}].shape[2] = {self.Ws[2 * y].shape[2]}."
                else:
                    if l != self.Ts[self.get_index(x-1, y, 1)].shape[1]:
                        errorMessage = f"Should match with Ts[({x-1}, {y}, {1})].shape[1] = {self.Ts[self.get_index(x-1, y, 1)].shape[1]}."
                if errorMessage is not None:
                    print(f"leg at Ts[({x}, {y}, {0})] in l direction does not match bond dimension!", errorMessage)
                    success = False
        # Check "r" bond of all p = 1 tensors at the right boundary
        for y in range(self.Ly):
            if self.Ts[self.get_index(self.Lx-1, y, 1)].shape[1] != 1:
                print(f"leg at Ts[({self.Lx-1}, {y}, {1})] in r direction does not match bond dimension! Should be 1.")
                success = False
        # Check "lu" bond of all p = 1 tensors at the upper boundary
        for x in range(self.Lx-1):
            if self.Ts[self.get_index(x, self.Ly-1, 1)].shape[3] != 1:
                print(f"leg at Ts[({x}, {self.Ly-1}, {1})] in lu direction does not match bond dimension! Should be 1.")
                success = False
        # Check conections of p = 1 tensors with the ortho surface
        if self.ortho_surface % 2 == 0:
            for y in range(self.Ly-1):
                _, _, ld, lu = self.Ts[self.get_index(self.ortho_surface//2, y, 1)].shape
                # "ld" bond
                errorMessage = None
                if ld != self.Ws[2*y].shape[2]:
                    errorMessage = f"Should match with Ws[{2*y}].shape[2] = {self.Ws[2*y].shape[2]}."
                if errorMessage is not None:
                    print(f"leg at Ts[({self.ortho_surface//2}, {y}, {1})] in ld direction does not match bond dimension!", errorMessage)
                    success = False
                # "lu" bond
                errorMessage = None
                if y < self.Ly-2 and lu != self.Ws[2*y+1].shape[2]:
                    errorMessage = f"Should match with Ws[{2*y+1}].shape[2] = {self.Ws[2*y+1].shape[2]}."
                if errorMessage is not None:
                    print(f"leg at Ts[({self.ortho_surface//2}, {y}, {1})] in lu direction does not match bond dimension!", errorMessage)
                    success = False
        else:
            for y in range(self.Ly-1):
                r = self.Ts[self.get_index(self.ortho_surface//2, y, 1)].shape[1]
                # "r" bond
                errorMessage = None
                if r != self.Ws[2*y].shape[0]:
                    errorMessage = f"Should match with Ws[{2*y}].shape[0] = {self.Ws[2*y].shape[0]}."
                if errorMessage is not None:
                    print(f"leg at Ts[({self.ortho_surface//2}, {y}, {1})] in r direction does not match bond dimension!", errorMessage)
                    success = False
        # Check W tensors
        if self.Ws[0].shape[3] != 1:
            print(f"leg at Ws[0] in down direction should have bond dimension of 1.")
            success = False
        if self.ortho_surface % 2 == 0:
            for i in range(len(self.Ws) - 1):
                if self.Ws[i].shape[1] != self.Ws[i+1].shape[3]:
                    print(f"leg at Ws[{i}] in up direction does not match bond dimension! Shoud match with Ws[{i + 1}].shape[3] = {self.Ws[i+1].shape[3]}.")
                    success = False
            if self.Ws[len(self.Ws) - 1].shape[1] != 1:
                print(f"leg at Ws[{len(self.Ws) - 1}] in up direction should have bond dimension of 1.")
                success = False
        else:
            for i in range(0, len(self.Ws) - 1, 2):
                if self.Ws[i].shape[1] != self.Ws[i+2].shape[3]:
                    print(f"leg at Ws[{i}] in up direction does not match bond dimension! Shoud match with Ws[{i + 2}].shape[3] = {self.Ws[i+1].shape[3]}.")
                    success = False
            if self.Ws[len(self.Ws) - 1].shape[1] != 1:
                print(f"leg at Ws[{len(self.Ws) - 2}] in up direction should have bond dimension of 1.")
                success = False
        return success

    def move_ortho_center_up(self):
        """
        Moves the orthogonality center one tensor upwards along the orthogonality surface.
        """
        if self.ortho_center >= 2 * self.Ly - 2:
            return
        if self.Ws[self.ortho_center + 1] is None:
            self.Ws[self.ortho_center], self.Ws[self.ortho_center + 2] = shifting_ortho_center.move_ortho_center_up(self.Ws[self.ortho_center], self.Ws[self.ortho_center + 2], chi_max=self.chi_max, options=self.shifting_options)
            self.ortho_center += 2
        else:
            self.Ws[self.ortho_center], self.Ws[self.ortho_center + 1] = shifting_ortho_center.move_ortho_center_up(self.Ws[self.ortho_center], self.Ws[self.ortho_center + 1], chi_max=self.chi_max, options=self.shifting_options)
            self.ortho_center += 1

    def move_ortho_center_down(self):
        """
        Moves the orthogonality center one tensor downwards along the orthogonality surface.
        """
        if self.ortho_center == 0:
            return
        if self.Ws[self.ortho_center - 1] is None:
            self.Ws[self.ortho_center - 2], self.Ws[self.ortho_center] = shifting_ortho_center.move_ortho_center_down(self.Ws[self.ortho_center - 2], self.Ws[self.ortho_center], chi_max=self.chi_max, options=self.shifting_options)
            self.ortho_center -= 2
        else:
            self.Ws[self.ortho_center - 1], self.Ws[self.ortho_center] = shifting_ortho_center.move_ortho_center_down(self.Ws[self.ortho_center - 1], self.Ws[self.ortho_center], chi_max=self.chi_max, options=self.shifting_options)
            self.ortho_center -= 1

    def perform_yang_baxter_1(self, W1_index, W2_index, T_index, flip):
        """
        Performs a Yang-Baxter move of variant one with the tensors at the given indices.
        The difference between Yang-Baxter moves of variant one and two are explained in
        "src/isoTPS/honeycomb/yang_baxter_move.py".

        Parameters
        ----------
        W1_index, W2_index : int
            indices of the W1 and W2 tensors. The orthogonality center must either be at W1 or W2.
        T_index : int
            index of the T tensor
        flip : bool
            If this is set to true, the one-site wavefunction is mirrored along the y axis. 
            When moving from left to right, flip should be set to False, and when moving from right to left, 
            flip should be set to true.

        Returns
        -------
        error : float
            the error of the YB move. This is only returned if the debug level is equal or larger than
            LOG_PER_SITE_ERROR_AND_WALLTIME. Else, -np.float("inf") is returned as error.
        """
        # Check if W1 or W2 are the ortho_center
        assert(W1_index == self.ortho_center or W2_index == self.ortho_center)
        # Obtain wavefunction tensors. Either W1 or W2 may also be None
        W1 = None
        if W1_index is not None:
            W1 = self.Ws[W1_index]
        W2 = None
        if W2_index is not None:
            W2 = self.Ws[W2_index]
        T = self.Ts[T_index]
        # Flip tensors if necessary
        if flip == True:
            W1 = utility.flip_W(W1)
            W2 = utility.flip_W(W2)
            T = utility.flip_T_honeycomb(T)
        # Save environment tensors (debug)
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_YB_TEBD_ENVIRONMENTS):
            utility.append_to_dict_list(self.debug_dict, "yb_environments", (W1, W2, T))
        # Stop time per YB move (debug)
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_PER_SITE_ERROR_AND_WALLTIME):
            start = time.time()
        # Perform YB move subroutine
        W, T, error = yang_baxter_move.yang_baxter_move_1(W1, W2, T, self.D_max_horizontal, debug_dict=self.debug_dict)
        # Save error and walltime (debug)
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_PER_SITE_ERROR_AND_WALLTIME):
            end = time.time()
            x, y, p = self.get_position(T_index)
            self.debug_dict["errors_yb"][y][x*2+p-1] += error
            self.debug_dict["times_yb"][y][x*2+p-1] += end-start
            if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_CONSECUTIVE_ERROR_AND_WALLTIME):
                utility.append_to_dict_list(self.debug_dict, "errors_yb_consecutive", error)
                utility.append_to_dict_list(self.debug_dict, "times_yb_consecutive", end-start)
        # Flip tensors back if necessary
        if flip == True:
            W = utility.flip_W(W)
            T = utility.flip_T_honeycomb(T)
        # Put updated wavefunction tensors back into the tps
        # and make sure the orthogonality center is at the correct position
        if flip:
            if W1_index is not None:
                self.Ws[W1_index] = None
            self.Ws[W2_index] = W
            self.ortho_center = W2_index
        else:
            if W2_index is not None:
                self.Ws[W2_index] = None
            self.Ws[W1_index] = W
            self.ortho_center = W1_index
        self.Ts[T_index] = T
        # Return the YB move error
        return error
    
    def perform_yang_baxter_2(self, W_index, T_index, flip, arrows_should_point_up=True):
        """
        Performs a Yang-Baxter move of variant two with the tensors at the given indices.
        The difference between Yang-Baxter moves of variant one and two are explained in
        "src/isoTPS/honeycomb/yang_baxter_move.py".

        Parameters
        ----------
        W_index : int
            index of the W tensor. Must be the orthogonality center.
        T_index : int
            index of the T tensor
        flip : bool
            If this is set to true, the one-site wavefunction is mirrored along the y axis. 
            When moving from left to right, flip should be set to False, and when moving from right to left, 
            flip should be set to true.
        arrows_should_point_up : bool, optional
            decides wether W1 (arrows_should_point_up==False) or W2 (arrows_should_point_up==True) will be the 
            new orthogonality center after the YB move. Default: True.

        Returns
        -------
        error : float
            the error of the YB move. This is only returned if the debug level is equal or larger than
            LOG_PER_SITE_ERROR_AND_WALLTIME. Else, -np.float("inf") is returned as error.
        """
        # Check if W1 or W2 are the ortho_center
        assert(W_index == self.ortho_center)
        # Obtain wavefunction tensors.
        W = self.Ws[W_index]
        T = self.Ts[T_index]
        # Flip tensors if necessary
        if flip == True:
            W = utility.flip_W(W)
            T = utility.flip_T_honeycomb(T)
        # Determine the direction of the larger bond dimension in case of uneven splitting
        if self.ordering_mode == "edges":
            if flip:
                larger_bond_direction = "down"
            else:
                larger_bond_direction = "up"
        elif self.ordering_mode == "down":
            larger_bond_direction = "down"
        elif self.ordering_mode == "up":
            larger_bond_direction = "up"
        elif self.ordering_mode == "center":
            if self.is_in_upper_half(T_index):
                larger_bond_direction = "down"
            else: 
                larger_bond_direction = "up"
        else:
            assert False, f'Unknown ordering mode \"{self.ordering_mode}\"'
        # Determine if we have an edge case
        mode = "both"
        if flip == False and W_index == 0:
            mode = "up"
        elif flip == True and W_index == 2*self.Ly-2:
            mode = "down"
        # Save environment tensors (debug)
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_YB_TEBD_ENVIRONMENTS):
            utility.append_to_dict_list(self.debug_dict, "yb_environments", (W, T, mode))
        # Stop time per YB move (debug)
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_PER_SITE_ERROR_AND_WALLTIME):
            start = time.time()
        # Perform YB move subroutine
        W1, W2, T, error = yang_baxter_move.yang_baxter_move_2(W, T, self.D_max, self.chi_max, mode=mode, options=self.yb_options, debug_dict=self.debug_dict, larger_bond_direction=larger_bond_direction)
        # Save error and walltime (debug)
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_PER_SITE_ERROR_AND_WALLTIME):
            end = time.time()
            x, y, p = self.get_position(T_index)
            self.debug_dict["errors_yb"][y][x*2+p-1] += error
            self.debug_dict["times_yb"][y][x*2+p-1]+= end-start
            if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_CONSECUTIVE_ERROR_AND_WALLTIME):
                utility.append_to_dict_list(self.debug_dict, "errors_yb_consecutive", error)
                utility.append_to_dict_list(self.debug_dict, "times_yb_consecutive", end-start)
        # Flip tensors back if necessary
        if flip == True:
            W1 = utility.flip_W(W1)
            W2 = utility.flip_W(W2)
            T = utility.flip_T_honeycomb(T)
        # Put updated wavefunction tensors back into the tps
        if mode == "both":
            if flip:
                self.Ws[W_index] = W1
                self.Ws[W_index+1] = W2
            else:
                self.Ws[W_index-1] = W1
                self.Ws[W_index] = W2
        elif mode == "up":
            self.Ws[W_index] = W2
        elif mode == "down":
            self.Ws[W_index] = W1 
        self.Ts[T_index] = T
        if mode == "both":
            if flip:
                self.ortho_center = W_index + 1
            if not arrows_should_point_up:
                self.move_ortho_center_down()
        # Return the YB move error
        return error
    
    def move_ortho_surface_left(self, force=False, move_upwards=True):
        """
        Moves the orthogonality hypersurface one column to the left.

        Parameters
        ----------
        force : bool, optional
            If force is set to true, the orthogonality hypersurface can be moved to index -1
            (to the left of all tensors). This is used when initializing the isoTPS in a product state.
            Default: False.
        move_upwards : bool, optional
            Controls the order in which the Yang-baxter moves are performed, either from the bottom up
            (move_upwards == True) or from the top down (move_updwards == False). Default: True.
        """
        if self.ortho_surface == 0 and force == False:
            return
        x = self.ortho_surface // 2
        p = self.ortho_surface % 2
        column_error = 0
        if move_upwards:
            # First, move the ortho center to the bottom
            self.move_ortho_center_to(0)
            # Now, use the yang baxter move to shift the ortho surface, one tensor at a time
            if p == 0:
                column_error += self.perform_yang_baxter_1(None, self.ortho_center, self.get_index(x, 0, 0), flip=True)
                for y in range(1, self.Ly):
                    self.move_ortho_center_up()
                    column_error += self.perform_yang_baxter_1(self.ortho_center, self.ortho_center+1, self.get_index(x, y, 0), flip=True)
            else:
                for y in range(self.Ly - 1):
                    column_error += self.perform_yang_baxter_2(self.ortho_center, self.get_index(x, y, 1), flip=True, arrows_should_point_up=True)
                    self.move_ortho_center_up()
                column_error += self.perform_yang_baxter_2(self.ortho_center, self.get_index(x, self.Ly-1, 1), flip=True, arrows_should_point_up=True)
        else:
            # First, move the ortho center to the top
            self.move_ortho_center_to(2*self.Ly - 2)
            # Now, use the yang baxter move to shift the ortho surface, one tensor at a time
            if p == 0:
                for y in range(self.Ly-1, 0, -1):
                    column_error += self.perform_yang_baxter_1(self.ortho_center-1, self.ortho_center, self.get_index(x, y, 0), flip=True)
                    self.move_ortho_center_down()
                column_error += self.perform_yang_baxter_1(self.ortho_center-1, self.ortho_center, self.get_index(x, 0, 0), flip=True)
            else:
                column_error += self.perform_yang_baxter_2(self.ortho_center, self.get_index(x, self.Ly - 1, 1), flip=True, arrows_should_point_up=False)
                for y in range(self.Ly-2, -1, -1):
                    self.move_ortho_center_down()
                column_error += self.perform_yang_baxter_2(self.ortho_center, self.get_index(x, y, 1), flip=True, arrows_should_point_up=False)
        self.ortho_surface -= 1
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_COLUMN_ERRORS):
            utility.append_to_dict_list(self.debug_dict, "column_errors_yb", column_error)

    def move_ortho_surface_right(self, force=False, move_upwards=True):
        """
        Moves the orthogonality hypersurface one column to the right.
        Note: This function should be overwritten in derived classes, as it is lattice specific!

        Parameters
        ----------
        force : bool, optional
            If force is set to true, the orthogonality hypersurface can be moved to index 2*self.Lx - 1
            (to the right of all tensors). This is used when initializing the isoTPS in a product state.
            Default: False.
        move_upwards : bool, optional
            Controls the order in which the Yang-baxter moves are performed, either from the bottom up
            (move_upwards == True) or from the top down (move_updwards == False). Default: True.
        """
        if self.ortho_surface >= 2 * self.Lx - 2 and force == False:
            return
        x = self.ortho_surface // 2
        p = self.ortho_surface % 2
        column_error = 0
        if move_upwards:
            # First, move the ortho center to the bottom
            self.move_ortho_center_to(0)
            # Now, use the yang baxter move to shift the ortho surface, one tensor at a time
            if p == 0:
                for y in range(self.Ly - 1):
                    column_error += self.perform_yang_baxter_1(self.ortho_center, self.ortho_center + 1, self.get_index(x, y, 1), flip=False)
                    self.move_ortho_center_up()
                column_error += self.perform_yang_baxter_1(self.ortho_center, None, self.get_index(x, self.Ly-1, 1), flip=False)
            else:
                column_error += self.perform_yang_baxter_2(self.ortho_center, self.get_index(x + 1, 0, 0), flip=False, arrows_should_point_up=True)
                for y in range(1, self.Ly):
                    self.move_ortho_center_up()
                    column_error += self.perform_yang_baxter_2(self.ortho_center, self.get_index(x + 1, y, 0), flip=False, arrows_should_point_up=True)
        else:
            # First, move the ortho center to the top
            self.move_ortho_center_to(2*self.Ly - 2)
            # Now, use the yang baxter move to shift the ortho surface, one tensor at a time
            if p == 0:
                column_error += self.perform_yang_baxter_1(self.ortho_center, None, self.get_index(x, self.Ly-1, 1), flip=False)
                for y in range(self.Ly-2, -1, -1):
                    self.move_ortho_center_down()
                    column_error += self.perform_yang_baxter_1(self.ortho_center-1, self.ortho_center, self.get_index(x, y, 1), flip=False)
            else:
                for y in range(self.Ly-1, 0, -1):
                    column_error += self.perform_yang_baxter_2(self.ortho_center, self.get_index(x + 1, y, 0), flip=False, arrows_should_point_up=False)
                    self.move_ortho_center_down()
                column_error += self.perform_yang_baxter_2(self.ortho_center, self.get_index(x + 1, 0, 0), flip=False, arrows_should_point_up=True)
        self.ortho_surface += 1
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_COLUMN_ERRORS):
            utility.append_to_dict_list(self.debug_dict, "column_errors_yb", column_error)

    def get_environment_twosite(self):
        r"""
        Returns the environment around the current ortho center,
        consisting of the tensors T1, T2, Wm1, W, Wp1.
        They are already flipped such that they satisfy one of the following two structures:

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

        This structure is also expected in the two-site operations in expectation_values.py
        and tebd.py. Note that one can distinguish between the two options by looking if both
        Wm1 and Wp1 are None, in which case the second structure is used

        Returns
        -------
        T1 : np.ndarray of shape (i, ru1, rd1, ld1, lu1)
            part of the twosite wavefunction
        T2 : np.ndarray of shape (j, ru2, rd2, ld2, lu2)
            part of the twosite wavefunction
        Wm1 : np.ndarray of shape (lm1, um1, rm1, dm1) = (rd1, d, rm1, dm1) or None
            part of the twosite wavefunction
        W : np.ndarray of shape (l, u, r, d) = (ru1, u, ld2, d)
            part of the twosite wavefunction
        Wp1 : np.ndarray of shape (lp1, up1, rp1, dp1) = (lp1, up1, lu2, u) or None
            part of the twosite wavefunction
        """
        x = self.ortho_surface // 2
        px = self.ortho_surface % 2
        y = self.ortho_center // 2
        py = self.ortho_center % 2
        
        T1 = None
        T2 = None
        Wm1 = None
        W = self.Ws[self.ortho_center]
        Wp1 = None
        
        if px == 0:
            if py == 0:
                T1 = self.Ts[self.get_index(x, y, 0)]
                T2 = self.Ts[self.get_index(x, y, 1)]
                flip = False
            else:
                T1 = self.Ts[self.get_index(x, y, 1)]
                T2 = self.Ts[self.get_index(x, y + 1, 0)]
                flip = True
            if self.ortho_center > 0:
                Wm1 = self.Ws[self.ortho_center - 1]
            if self.ortho_center < len(self.Ws) - 1:
                Wp1 = self.Ws[self.ortho_center + 1]
            if flip:
                T1, T2, Wm1, W, Wp1 = utility.flip_twosite_honeycomb(T1, T2, Wm1, W, Wp1)
        else:
            T1 = self.Ts[self.get_index(x, y, 1)]
            T2 = self.Ts[self.get_index(x+1, y, 0)]
        return T1, T2, Wm1, W, Wp1

    def set_environment_twosite(self, T1, T2, Wm1, W, Wp1):
        """
        Sets the environment around the current orthogonality center,
        consisting of the tensors T1, T2, Wm1, W, Wp1.
        The tensors are flipped if necessary. This function expects the same tensor structure
        as is returned by get_environment(). T1 and T2 can also be None, in which case they won't
        be set.

        Parameters
        ----------
        T1, T2, Wm1, W, Wp1: np.ndarray or None
            twosite wavefunction. For the shapes of the individual tensors, see get_environment_twosite().
        """
        x = self.ortho_surface // 2
        px = self.ortho_surface % 2
        y = self.ortho_center // 2
        py = self.ortho_center % 2

        if px == 0:
            if py == 0:
                self.Ts[self.get_index(x, y, 0)] = T1
                self.Ts[self.get_index(x, y, 1)] = T2
            else:
                T1, T2, Wm1, W, Wp1 = utility.flip_twosite_honeycomb(T1, T2, Wm1, W, Wp1)
                self.Ts[self.get_index(x, y, 1)] = T1
                self.Ts[self.get_index(x, y + 1, 0)] = T2
            if self.ortho_center > 0:
                self.Ws[self.ortho_center - 1] = Wm1
            if self.ortho_center < len(self.Ws) - 1:
                self.Ws[self.ortho_center + 1] = Wp1
        else:
            assert(Wm1 is None and Wp1 is None)
            self.Ts[self.get_index(x, y, 1)] = T1
            self.Ts[self.get_index(x+1, y, 0)] = T2

        self.Ws[self.ortho_center] = W

    def get_environment_onesite(self, left_environment=True):
        r"""
        Returns the one-site environment next to the current ortho center.
        Either returns the left or the right environment. Depending on the position of
        the ortho surface, the returned tensors can have two structures.

                      | /  
                      |/    
                     Wp1                    \ 
                     /|                      \ |    |
                    / |                       \|    |
                 --T  |           or           T----W----
                    \ |                       /     |
                     \|                      /      | 
                      W                     / 
                      |\ 
                      | \ 
        
        The tensors are in the same form as expected by compute_norm_onesite() and
        compute_expectation_values_onesite(). The ortho center might be either at W or Wp1.

        Parameters
        ----------
        left_environment: bool, optional
            If set to true, the left environment (T tensor left of the ortho center) is returned.
            Otherwise, the right environment (T tensor right of the ortho center) is returned. Default : True.

        Returns
        -------
        T : np.ndarray of shape (i, ru, rd, ld, lu)
            part of the onesite wavefunction
        W : np.ndarray of shape (l, u, r, d) = (rd, u, r, d) or None
            part of the onesite wavefunction
        Wp1 : np.ndarray of shape (lp1, up1, rp1, dp1) = (ru, up1, rp1, u) or None
            part of the onesite wavefunction
        """
        x = self.ortho_surface // 2
        px = self.ortho_surface % 2
        y = self.ortho_center // 2
        py = self.ortho_center % 2

        T = None
        W = None
        Wp1 = None
        flip = False

        if px == 0:
            if left_environment:
                if py == 0:
                    if self.ortho_center > 0:
                        W = self.Ws[self.ortho_center - 1]
                    Wp1 = self.Ws[self.ortho_center]
                    T = self.Ts[self.get_index(x, y, 0)]
                else:
                    W = self.Ws[self.ortho_center]
                    Wp1 = self.Ws[self.ortho_center + 1]
                    T = self.Ts[self.get_index(x, y+1, 0)]
            else:
                flip = True
                if py == 0:
                    T = self.Ts[self.get_index(x, y, 1)]
                    W = self.Ws[self.ortho_center]
                    if self.ortho_center < 2*self.Ly - 2:
                        Wp1 = self.Ws[self.ortho_center + 1]
                else:
                    W = self.Ws[self.ortho_center - 1]
                    Wp1 = self.Ws[self.ortho_center]
                    T = self.Ts[self.get_index(x, y, 1)]
        else:
            W = self.Ws[self.ortho_center]
            if left_environment:
                T = self.Ts[self.get_index(x, y, 1)]
            else:
                T = self.Ts[self.get_index(x+1, y, 0)]
                flip = True
            
        if flip:
            T, W, Wp1 = utility.flip_onesite_honeycomb(T, W, Wp1)
        
        return T, W, Wp1

    def compute_norm(self):
        """
        Computes the norm of the isoTPS (Should always be 1).

        Returns
        -------
        norm : float
            the computed norm
        """
        return expectation_values.compute_norm_twosite(*self.get_environment_twosite())

    def compute_expectation_values_onesite(self, ops):
        """
        Computes one site expectation values and returns a list with one or more expectation values per site

        Parameters
        ----------
        ops : list of np.ndarray of shape (self.d, self.d)
            list of one-site operators. At each site, an expectation value is computed from each of
            the operators in the list
        
        Returns
        -------
        result : list of list of float
            list containing for each site a list of expecation values. 
            Shape of the list: (N, len(ops)), where N is the number of sites N = 2*self.Lx * self.Ly
        """
        result = []
        for i in range(len(ops)):
            result.append([])
        for x in range(0, 2*self.Lx, 2):
            self.move_to(x, 0)
            T, W, Wp1 = self.get_environment_onesite(left_environment=True)
            for i, op in enumerate(ops):
                result[i].append(expectation_values.expectation_value_onesite_1(T, W, Wp1, op))
            left_environment = False
            for y in range(2*self.Ly - 1):
                T, W, Wp1 = self.get_environment_onesite(left_environment=left_environment)
                for i, op in enumerate(ops):
                    result[i].append(expectation_values.expectation_value_onesite_1(T, W, Wp1, op))
                left_environment = not left_environment
                self.move_ortho_center_up()
        return result

    def compute_expectation_values_twosite(self, ops):
        """
        Computes two site expectation values and returns a list with one expectation value per bond
        
        Parameters
        ----------
        ops: list of np.ndarray of shape (self.d, self.d, self.d, self.d) = (i, j, i, j)
            list of twosite operators. At each bond, the expectation value of the corresponding operator
            is computed. In total, ops should have length len(ops) == (2 * Ly - 1) * (2 * Lx - 1) = N_bonds

        Returns
        result : list of float
            list of bond expectation values. The length of the list is the number of bonds in the isoTPS
        """ 
        result = []
        for xi in range(2 * self.Lx - 1):
            self.move_to(xi, 0)
            for yi in range(2 * self.Ly - 1):
                if xi%2 == 1 and yi%2 == 1:
                    continue
                T1, T2, Wm1, W, Wp1 = self.get_environment_twosite()
                if xi%2 == 0:
                    result.append(expectation_values.expectation_value_twosite_1(T1, T2, Wm1, W, Wp1, ops[xi * (2 * self.Ly - 1) + yi]))
                else:
                    result.append(expectation_values.expectation_value_twosite_2(T1, T2, W, ops[xi * (2 * self.Ly - 1) + yi]))
                self.move_ortho_center_up()
        return result

    def perform_TEBD_at_ortho_center(self, U_bonds):
        """
        Applies the corresponding time evolution operator to the current bond (the bond at the current orthogonality center).

        Parameters
        ----------
        U_bonds : list of np.ndarray of shape (i, j, i*, j*)
            list of time evolution bond operators

        Returns
        -------
        error : float
            the error of the YB move. If the debug level is smaller than LOG_PER_SITE_ERROR_AND_WALLTIME,
            -np.float("inf") is returned instead.
        """
        log_time_and_error = debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_PER_SITE_ERROR_AND_WALLTIME)
        # determine index into U_bonds)
        index = self.ortho_surface * (2 * self.Ly - 1) + self.ortho_center
        # retrieve environment
        T1, T2, Wm1, W, Wp1 = self.get_environment_twosite()
        # Save environment tensors (debug)
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_YB_TEBD_ENVIRONMENTS):
            utility.append_to_dict_list(self.debug_dict, "tebd_environments", (T1, T2, Wm1, W, Wp1))
        # perform TEBD step
        if log_time_and_error:
            start = time.time()
        if self.ortho_surface%2 == 0:
            T1, T2, Wm1, W, Wp1, error = tebd.tebd_step_2(T1, T2, Wm1, W, Wp1, U_bonds[index], self.chi_max, log_error=log_time_and_error, **self.tebd_options)
        else:
            T1, T2, W, error = tebd.tebd_step_1(T1, T2, W, U_bonds[index], log_error=log_time_and_error, **self.tebd_options)
        if log_time_and_error:
            end = time.time()
            self.debug_dict["times_tebd"][self.ortho_center][self.ortho_surface] += end-start
            self.debug_dict["errors_tebd"][self.ortho_center][self.ortho_surface] += error
            if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_CONSECUTIVE_ERROR_AND_WALLTIME):
                utility.append_to_dict_list(self.debug_dict, "errors_tebd_consecutive", error)
                utility.append_to_dict_list(self.debug_dict, "times_tebd_consecutive", end-start)
        # Update environment
        self.set_environment_twosite(T1, T2, Wm1, W, Wp1)
        return error

    def perform_TEBD_along_ortho_surface_brickwall(self, U_bonds):
        """
        Applies TEBD update operators along the current orthogonality hypersurface in a brickwall pattern.
        First updates all even bonds and then all odd bonds.

            |    _|_____|_   _|_____|_    |     
            |    |___U___|   |___U___|    |     
           _|_____|_   _|_____|_   _|_____|_    
           |___U___|   |___U___|   |___U___|    
            |     |     |     |     |     |     
        ----W1----W2----W3----W4----W5----W6----

        Note: this sketch is only qualitatively correct, since the update operators are actually applied to the T-tensors,
        which are arranged in zig-zag pattern.

        Parameters
        ----------
        U_bonds : list of np.ndarray of shape (i, j, i*, j*)
            list of time evolution bond operators
        """
        # apply update on even bonds
        self.move_ortho_center_to(0)
        column_error = 0
        for y in range(0, 2 * self.Ly - 1, 2):
            column_error += self.perform_TEBD_at_ortho_center(U_bonds)
            self.move_ortho_center_up()
            if self.ortho_surface%2 == 0:
                self.move_ortho_center_up()
        # apply update on odd bonds
        if self.Ly > 1 and self.ortho_surface%2 == 0:
            self.move_ortho_center_to(2*self.Ly-3)
            for y in range(2*self.Ly-3, 0, -2):
                column_error += self.perform_TEBD_at_ortho_center(U_bonds)
                self.move_ortho_center_down()
                self.move_ortho_center_down()
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_COLUMN_ERRORS):
            utility.append_to_dict_list(self.debug_dict, "column_errors_tebd", column_error)

    def perform_TEBD_along_ortho_surface_chain(self, U_bonds, move_upwards=True):
        """
        Applies TEBD update operators along the current orthogonality surface in a chain pattern.

            |     |     |     |    _|_____|_    
            |     |     |     |    |___U___|    
            |     |     |    _|_____|_    |     
            |     |     |    |___U___|    |     
            |     |    _|_____|_    |     |     
            |     |    |___U___|    |     |     
            |    _|_____|_    |     |     |     
            |    |___U___|    |     |     |     
           _|_____|_    |     |     |     |     
           |___U___|    |     |     |     |     
            |     |     |     |     |     |     
        ----W1----W2----W3----W4----W5----W6----

        Note: this sketch is only qualitatively correct, since the update operators are actually applied to the T-tensors,
        which are arranged in zig-zag pattern.

        Parameters
        ----------
        U_bonds : list of np.ndarray of shape (i, j, i*, j*)
            list of time evolution bond operators
        move_upwards : bool, optional
            if set to true, operators are applied from the bottom up, otherwise they are applied
            from the top down. Default: True.
        """
        column_error = 0
        if move_upwards:
            self.move_ortho_center_to(0)
            for _ in range(0, 2*self.Ly-1, 1+self.ortho_surface%2):
                column_error += self.perform_TEBD_at_ortho_center(U_bonds)
                self.move_ortho_center_up() 
        else:
            self.move_ortho_center_to(2*self.Ly-2)
            for _ in range(2*self.Ly-1, 0, -1-self.ortho_surface%2):
                column_error += self.perform_TEBD_at_ortho_center(U_bonds)
                self.move_ortho_center_down() 
        if debug_levels.check_debug_level(self.debug_dict, debug_levels.DebugLevel.LOG_COLUMN_ERRORS):
            utility.append_to_dict_list(self.debug_dict, "column_errors_tebd", column_error)

    def perform_TEBD1(self, U_bonds, N_steps):
        """
        Performs N_steps of full first-order TEBD update steps by applying bond operators in a brickwall fashion.
        During one update step, the orthogonality surface is moved once from the left to the right and back.

        Parameters
        ----------
        U_bonds : list of np.ndarray of shape (i, j, i*, j*)
            list of real or imaginary time evolution bond operators of time dtau.
        N_steps : int
            number of TEBD steps performed
        """
        for _ in range(N_steps):
            # apply update on even surfaces
            self.move_ortho_surface_to(0)
            for x in range(0, 2 * self.Lx - 1, 2):
                self.perform_TEBD_along_ortho_surface_brickwall(U_bonds)
                self.move_ortho_surface_right()
                self.move_ortho_surface_right()
            # apply update on odd surfaces
            if self.Lx > 1:
                self.move_ortho_surface_to(2*self.Lx-3)
                for x in range(2*self.Lx-3, 0, -2):
                    self.perform_TEBD_along_ortho_surface_brickwall(U_bonds)
                    self.move_ortho_surface_left()
                    self.move_ortho_surface_left()

    def perform_TEBD2(self, U_bonds, N_steps):
        """
        Performs N_steps of full second-order TEBD update steps by applying bond operators in a chain fashion.
        During one update step, the orthogonality surface is moved once from the left to the right and back.

        Parameters
        ----------
        U_bonds : list of np.ndarray of shape (i, j, i*, j*)
            list of real or imaginary time evolution bond operators of time dtau/2.
        N_steps : int
            number of TEBD steps performed
        """
        for _ in range(N_steps):
            self.move_to(0, 0)
            for _ in range(2*self.Lx-1):
                self.perform_TEBD_along_ortho_surface_chain(U_bonds, move_upwards=True)
                self.move_ortho_surface_right(move_upwards=False)
            for _ in range(2*self.Lx-1):
                self.perform_TEBD_along_ortho_surface_chain(U_bonds, move_upwards=False)
                self.move_ortho_surface_left(move_upwards=True)