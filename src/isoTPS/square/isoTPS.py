import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
from ...utility import utility
from ...utility import debug_logging
from . import yang_baxter_move
from . import expectation_values
from . import tebd
from .. import isoTPS
from . import column_optimization

class isoTPS_Square(isoTPS.isoTPS):

    @staticmethod
    def load_from_file(filename):
        tps = isoTPS_Square(0, 0)
        with h5py.File(filename, "r") as hf:
            data = utility.load_dict_from_hf(hf)
            tps._load_from_dict(data)
        return tps

    def copy(self):
        """
        Returns a copy of this isoTPS

        Returns
        -------
        copy : instance if class isoTPS_Square
            the copied isoTPS
        """
        result = isoTPS_Square(self.Lx, self.Ly, D_max=self.D_max, chi_factor=self.chi_factor, chi_max=self.chi_max, d=self.d, shifting_options=self.shifting_options, yb_options=self.yb_options, tebd_options=self.tebd_options, ordering_mode=self.ordering_mode, perform_variational_column_optimization=self.perform_variational_column_optimization, variational_column_optimization_options=self.variational_column_optimization_options, debug_logger_options=self.debug_logger_options)
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
        # Temporarily set debug level to zero to avoid collecting unnecessary debug information during initialization
        temp_debug_logger = self.debug_logger
        self.debug_logger = debug_logging.DebugLogger()
        def _initialize_T_product_state(state=None, dtype=np.complex128):
            if state is None:
                state = np.zeros((self.d), dtype=np.complex128)
                state[0] = 1.0
            T = np.zeros((state.size, 1, 1, 1, 1), dtype=np.complex128)
            T[:, 0, 0, 0, 0] = state[:]
            return T
        # First, initialize the ortho center, which is to the right of all T tensors
        self.ortho_surface = 2 * self.Lx - 1
        for i in range(2 * self.Ly - 1):
            self.Ws[i] = np.array([[[[1.]]]], dtype=np.complex128)
        # We go from right to left, initializing the T tensors in product states and moving the ortho surface left
        for x in range(self.Lx - 1, -1, -1):
            for y in range(self.Ly):
                index = self.get_index(x, y, 1)
                self.Ts[index] = _initialize_T_product_state(states[index])
                index = self.get_index(x, y, 0)
                self.Ts[index] = _initialize_T_product_state(states[index])
            self.move_ortho_surface_left(force=True)
            self.move_ortho_surface_left(force=True)
        self.move_ortho_surface_right(force=True)
        assert(self.ortho_surface == 0)
        self.debug_logger = temp_debug_logger

    def plot(self, T_colors=None, ax=None, figsize_y=8.0, T_tensor_scale=1.0, W_tensor_scale=1.0, show_bond_dims=True):
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
        T_tensor_scale : float, optional
            size multiplier for T tensors. Default: 1.0
        W_tensor_scale : float, optional
            size multiplier for W tensors. Default: 1.0
        show_bond_dims : boolean, optional
            wether to draw the bond dimensions next to the legs. Default: True.
        """
        scale = figsize_y / (4 * self.Ly)
        if ax is None:
            fig, ax = plt.subplots(figsize=(4 * self.Lx * scale, 4 * self.Ly * scale))
        ax.axis("equal")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if T_colors is None:
            # Default color: black
            T_colors = [np.array([0, 0, 0])] * self.Lx * self.Ly * 2

        # Helper function for drawing labelled arrows
        def construct_labelled_arrow(start, direction, label, labelPos="upper left", color="black"):
            dx_2 = direction[0] / 2
            dy_2 = direction[1] / 2
            delta_l = 0.025
            alpha = np.arctan(dx_2/dy_2)
            ddx = np.sin(alpha) * delta_l
            ddy = np.cos(alpha) * delta_l
            ax.arrow(start[0], start[1], dx_2 + ddx, dy_2 + ddy, head_width=0.05, head_length=0.05, color=color)
            ax.arrow(start[0] + dx_2 + ddx, start[1] + dy_2 + ddy, dx_2 - ddx, dy_2 - ddy, head_width=0, color=color)
            if show_bond_dims:
                labelCoords = (0, 0)
                if labelPos == "upper left":
                    labelCoords = (-5, 5)
                    ha = "right"
                elif labelPos == "upper right":
                    labelCoords = (5, 5)
                    ha = "left"
                elif labelPos == "right":
                    labelCoords = (5, 0)
                    ha = "left"
                elif labelPos == "left":
                    ha = "right"
                    labelCoords = (-5, 0)
                ax.annotate(label, (start[0] + dx_2, start[1] + dy_2), xytext=labelCoords, textcoords='offset points', ha=ha)

        # Draw legs coming from all tensors at p = 0
        for x in range(self.Lx):
            for y in range(self.Ly):
                # arrow to top left
                if x > 0:
                    if 2 * x - 1 < self.ortho_surface:
                        construct_labelled_arrow((2 * x - 1, 2 * y + 1), (1, -1), str(self.Ts[self.get_index(x, y, 0)].shape[4]), labelPos="upper right")
                    elif 2 * x - 1 > self.ortho_surface:
                        construct_labelled_arrow((2 * x, 2 * y), (-1, 1), str(self.Ts[self.get_index(x, y, 0)].shape[4]), labelPos="upper right")
                    else:
                        construct_labelled_arrow((2 * x, 2 * y), (-0.5, 0.5), str(self.Ts[self.get_index(x, y, 0)].shape[4]), labelPos="upper right")
                # arrow to bottom left
                if x > 0 and y > 0:
                    if 2 * x - 1 < self.ortho_surface:
                        construct_labelled_arrow((2 * x - 1, 2 * y - 1), (1, 1), str(self.Ts[self.get_index(x, y, 0)].shape[3]), labelPos="upper left")
                    elif 2 * x - 1 > self.ortho_surface:
                        construct_labelled_arrow((2 * x, 2 * y), (-1, -1), str(self.Ts[self.get_index(x, y, 0)].shape[3]), labelPos="upper left")
                    else:
                        construct_labelled_arrow((2 * x, 2 * y), (-0.5, -0.5), str(self.Ts[self.get_index(x, y, 0)].shape[3]), labelPos="upper left")
                # arrow to bottom right
                if y > 0:
                    if 2 * x < self.ortho_surface:
                        construct_labelled_arrow((2 * x, 2 * y), (1, -1), str(self.Ts[self.get_index(x, y, 0)].shape[2]), labelPos="upper right")
                    elif 2 * x > self.ortho_surface:
                        construct_labelled_arrow((2 * x + 1, 2 * y - 1), (-1, 1), str(self.Ts[self.get_index(x, y, 0)].shape[2]), labelPos="upper right")
                    else:
                        construct_labelled_arrow((2 * x, 2 * y), (0.5, -0.5), str(self.Ts[self.get_index(x, y, 0)].shape[2]), labelPos="upper right")
                # arrow to top right
                if 2 * x < self.ortho_surface:
                    construct_labelled_arrow((2 * x, 2 * y), (1, 1), str(self.Ts[self.get_index(x, y, 0)].shape[1]), labelPos="upper left")
                elif 2 * x > self.ortho_surface:
                    construct_labelled_arrow((2 * x + 1, 2 * y + 1), (-1, -1), str(self.Ts[self.get_index(x, y, 0)].shape[1]), labelPos="upper left")
                else:
                    construct_labelled_arrow((2 * x, 2 * y), (0.5, 0.5), str(self.Ts[self.get_index(x, y, 0)].shape[1]), labelPos="upper left")
                # arrow for physical index
                construct_labelled_arrow((2 * x, 2 * y + 0.4), (0, -0.4), str(self.Ts[self.get_index(x, y, 0)].shape[0]), labelPos="right", color="green")

                # Draw arrows for tensor at p = 1 if they are right next to the orthogonality surface
                if 2 * x == self.ortho_surface:
                    # top left
                    if y < self.Ly - 1:
                        construct_labelled_arrow((2 * x + 1, 2 * y + 1), (-0.5, 0.5), str(self.Ts[self.get_index(x, y, 1)].shape[4]), labelPos="upper right")
                    # bottom left
                    construct_labelled_arrow((2 * x + 1, 2 * y + 1), (-0.5, -0.5), str(self.Ts[self.get_index(x, y, 1)].shape[3]))
                elif 2 * x + 1 == self.ortho_surface:
                    # bottom right
                    construct_labelled_arrow((2 * x + 1, 2 * y + 1), (0.5, -0.5), str(self.Ts[self.get_index(x, y, 1)].shape[2]), labelPos="upper right")
                    # top right
                    if y < self.Ly - 1:
                        construct_labelled_arrow((2 * x + 1, 2 * y + 1), (0.5, 0.5), str(self.Ts[self.get_index(x, y, 1)].shape[1]))
                # arrow for physical index
                construct_labelled_arrow((2 * x + 1, 2 * y + 1.4), (0, -0.4), str(self.Ts[self.get_index(x, y, 1)].shape[0]), labelPos="right", color="green")

        # Draw arrows for each tensor of the orthogonality surface
        labelPos = "right"
        if self.ortho_surface % 2 != 0:
            labelPos = "left"
        for y in range(2 * self.Ly - 2):
            if labelPos == "right":
                labelPos = "left"
            else:
                labelPos = "right"
            if (y < self.ortho_center):
                construct_labelled_arrow((self.ortho_surface + 0.5, 0.5 + y), (0, 1), str(self.Ws[y].shape[1]), labelPos=labelPos, color="red")
            else:
                construct_labelled_arrow((self.ortho_surface + 0.5, 1.5 + y), (0, -1), str(self.Ws[y].shape[1]), labelPos=labelPos, color="red")
        if labelPos == "right":
            labelPos = "left"
        else:
            labelPos = "right"
            
        # Draw actual T tensors
        for y in range(self.Ly):
            for x in range(self.Lx):
                ax.add_patch(plt.Circle((2 * x, 2 * y), 0.06*T_tensor_scale, color=T_colors[self.get_index(x, y, 0)]))
                ax.add_patch(plt.Circle((2 * x + 1, 2 * y + 1), 0.06*T_tensor_scale, color=T_colors[self.get_index(x, y, 1)]))

        # Draw actual W tensors
        for y in range(2 * self.Ly - 1):
            if y == self.ortho_center:
                ax.add_patch(plt.Circle((self.ortho_surface + 0.5, y + 0.5), 0.06*W_tensor_scale, color="orange"))
            else:            
                ax.add_patch(plt.Circle((self.ortho_surface + 0.5, y + 0.5), 0.06*W_tensor_scale, color="red"))
    
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
                for p in [0, 1]:
                    T = self.Ts[self.get_index(x, y, p)]
                    if 2 * x < self.ortho_surface or p == 0 and 2 * x == self.ortho_surface:
                        T = np.transpose(T, (0, 3, 4, 1, 2)) # p ru rd ld lu -> p ld lu ru rd
                    T = np.reshape(T, (T.shape[0]*T.shape[1]*T.shape[2], T.shape[3]*T.shape[4]))
                    if not utility.check_isometry(T):
                        print(f"T tensor at (x={x}, y={y}, p={p}) is not an isometry ...")
                        success = False
        # Check W tensors
        for i in range(len(self.Ws)):
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
        # Check T tensors
        for x in range(self.Lx):
            for y in range(self.Ly):
                for p in [0, 1]:
                    # Check if leg dimension along ru direction matches up
                    ru = self.Ts[self.get_index(x, y, p)].shape[1]
                    errorMessage = None
                    if p == 0:
                        if 2 * x == self.ortho_surface:
                            if ru != self.Ws[2 * y].shape[0]:
                                errorMessage = f"Should match with Ws[{2 * y}].shape[0] = {self.Ws[2 * y].shape[0]}."
                        else:
                            if ru != self.Ts[self.get_index(x, y, 1)].shape[3]:
                                errorMessage = f"Should match with Ts[({x}, {y}, {1})].shape[3] = {self.Ts[self.get_index(x, y, 1)].shape[3]}."
                    else:
                        if x == self.Lx - 1 or y == self.Ly - 1:
                            if ru != 1:
                                errorMessage = "Should be 1."
                        elif 2 * x + 1 == self.ortho_surface:
                            if ru != self.Ws[2 * y + 1].shape[0]:
                                errorMessage = f"Should match with Ws[{2 * y + 1}].shape[0] = {self.Ws[2 * y + 1].shape[0]}."
                        else:
                            if ru != self.Ts[self.get_index(x + 1, y + 1, 0)].shape[3]:
                                errorMessage = f"Should match with Ts[({x + 1}, {y + 1}, {0})].shape[3] = {self.Ts[self.get_index(x + 1, y + 1, 0)].shape[3]}."
                    if errorMessage is not None:
                        print(f"leg at Ts[({x}, {y}, {p})] in ru direction does not match bond dimension!", errorMessage)
                        success = False
                    # Check if leg dimensions along rd direction matches up
                    rd = self.Ts[self.get_index(x, y, p)].shape[2]
                    if p == 0:
                        if y == 0:
                            if rd != 1:
                                errorMessage("Should be 1.")
                        elif 2 * x == self.ortho_surface:
                            if rd != self.Ws[2 * y - 1].shape[0]:
                                errorMessage = f"Should match with Ws[{2 * y - 1}].shape[0] = {self.Ws[2 * y - 1].shape[0]}."
                        else:
                            if rd != self.Ts[self.get_index(x, y-1, 1)].shape[4]:
                                errorMessage = f"Should match with Ts[({x}, {y - 1}, {1})].shape[4] = {self.Ts[self.get_index(x, y-1, 1)].shape[4]}."
                    else:
                        if x == self.Lx - 1:
                            if rd != 1:
                                errorMessage("Should be 1.")
                        elif 2 * x + 1 == self.ortho_surface:
                            if rd != self.Ws[2 * y].shape[0]:
                                errorMessage = f"Should match with Ws[{2 * y}].shape[0] = {self.Ws[2 * y].shape[0]}."
                        else:
                            if rd != self.Ts[self.get_index(x + 1, y, 0)].shape[4]:
                                errorMessage = f"Should match with Ts[({x + 1}, {y}, {0})].shape[4] = {self.Ts[self.get_index(x + 1, y, 0)].shape[4]}."
                    if errorMessage is not None:
                        print(f"leg at Ts[({x}, {y}, {p})] in ru direction does not match bond dimension!", errorMessage)
                        success = False
        # Check W tensors
        if self.Ws[0].shape[3] != 1:
            print(f"leg at Ws[0] in down direction should have bond dimension of 1.")
            success = False
        for i in range(len(self.Ws) - 1):
            if self.Ws[i].shape[1] != self.Ws[i+1].shape[3]:
                print(f"leg at Ws[{i}] in up direction does not match bond dimension! Shoud match with Ws[{i + 1}].shape[3] = {self.Ws[i+1].shape[3]}.")
                success = False
        if self.Ws[len(self.Ws) - 1].shape[1] != 1:
            print(f"leg at Ws[{len(self.Ws) - 1}] in up direction should have bond dimension of 1.")
            success = False
        return success

    def perform_yang_baxter(self, W1_index, W2_index, T_index, flip, arrows_should_point_up=True):
        """
        Performs a Yang-Baxter move with the tensors at the given indices.

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
        arrows_should_point_up : bool, optional
            decides wether W1 (arrows_should_point_up==False) or W2 (arrows_should_point_up==True) will be the 
            new orthogonality center after the YB move. Default: True.

        Returns
        -------
        error : float
            the error of the YB move. This is only returned if debug_logger.log_approximate_column_error_yb is set to true.
        
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
            T = utility.flip_T_square(T)
        # Determine the direction of the larger bond dimension
        # in case of uneven splitting
        if self.ordering_mode == "down":
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
        # Save environment tensors (debug)
        if self.debug_logger.log_yb_move_environments:
            self.debug_logger.append_to_log_list(("yb_environments"), {"W1": None if W1 is None else W1.copy(), "W2": None if W2 is None else W2.copy(), "T": T.copy()})
        # Stop time per YB move (debug)
        if self.debug_logger.log_algorithm_walltimes or self.debug_logger.log_yb_move_walltimes:
            start = time.time()
        # Perform YB move subroutine
        Wm1_prime, W_prime, T_prime, error = yang_baxter_move.yang_baxter_move(W1, W2, T, self.D_max, self.chi_max, options=self.yb_options, debug_logger=self.debug_logger, larger_bond_direction=larger_bond_direction)
        # Save error and walltime (debug)
        if self.debug_logger.log_algorithm_walltimes or self.debug_logger.log_yb_move_walltimes:
            end = time.time()
        if self.debug_logger.log_yb_move_errors:
            self.debug_logger.append_to_log_list(("yb_move_errors"), error)
        if self.debug_logger.log_yb_move_walltimes:
            self.debug_logger.append_to_log_list(("yb_move_walltimes"), end-start)
        if self.debug_logger.log_algorithm_walltimes:
            self.time_counter_yb_move += end-start
        # Flip tensors back if necessary
        if flip == True:
            Wm1_prime = utility.flip_W(Wm1_prime)
            W_prime = utility.flip_W(W_prime)
            T_prime = utility.flip_T_square(T_prime)
        # Put updated wavefunction tensors back into the tps
        if W1_index is not None:
            self.Ws[W1_index] = Wm1_prime
        if W2_index is not None:
            self.Ws[W2_index] = W_prime
        self.Ts[T_index] = T_prime
        # Make sure the ortho center is at the correct position
        if self.ortho_center == W1_index and self.ortho_center < len(self.Ws) - 1:
            self.ortho_center = W2_index
        if arrows_should_point_up == False and W2_index is not None:
            self.move_ortho_center_down()
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
        # If we want to perform a variational column optimization, we need to store the column's tensors before doing any YB moves
        Ts_before_YB = []
        Ws_before_YB = []
        if self.ortho_surface > 0 and (self.perform_variational_column_optimization or self.debug_logger.log_column_error_yb_before_variational_optimization):
            if p == 0:
                Ts_before_YB = [utility.flip_T_square(self.Ts[self.get_index(x, y, 0)].copy()) for y in range(self.Ly)]
                Ws_before_YB = [None] + [utility.flip_W(W.copy()) for W in self.Ws]
            else:
                Ts_before_YB = [utility.flip_T_square(self.Ts[self.get_index(x, y, 1)].copy()) for y in range(self.Ly)]
                Ws_before_YB = [utility.flip_W(W.copy()) for W in self.Ws] + [None]
        if move_upwards:
            # First, move the ortho center to the bottom
            self.move_ortho_center_to(0)
            # Now, use the yang baxter move to shift the ortho surface, one tensor at a time
            if p == 0:
                column_error += self.perform_yang_baxter(None, self.ortho_center, self.get_index(x, 0, 0), flip=True, arrows_should_point_up=True)
                for y in range(1, self.Ly):
                    self.move_ortho_center_up()
                    column_error += self.perform_yang_baxter(self.ortho_center, self.ortho_center + 1, self.get_index(x, y, 0), flip=True, arrows_should_point_up=True)
            else:
                for y in range(self.Ly - 1):
                    column_error += self.perform_yang_baxter(self.ortho_center, self.ortho_center + 1, self.get_index(x, y, 1), flip=True, arrows_should_point_up=True)
                    self.move_ortho_center_up()
                column_error += self.perform_yang_baxter(self.ortho_center, None, self.get_index(x, self.Ly - 1, 1), flip=True, arrows_should_point_up=True)
        else:
            # First, move the ortho center to the top
            self.move_ortho_center_to(2*self.Ly - 2)
            # Now, use the yang baxter move to shift the ortho surface, one tensor at a time
            if p == 0:
                for y in range(self.Ly-1, 0, -1):
                    column_error += self.perform_yang_baxter(self.ortho_center - 1, self.ortho_center, self.get_index(x, y, 0), flip=True, arrows_should_point_up=False)
                    self.move_ortho_center_down()
                column_error += self.perform_yang_baxter(None, self.ortho_center, self.get_index(x, 0, 0), flip=True, arrows_should_point_up=True)
            else:
                column_error += self.perform_yang_baxter(self.ortho_center, None, self.get_index(x, self.Ly - 1, 1), flip=True, arrows_should_point_up=False)
                for y in range(self.Ly-2, -1, -1):
                    self.move_ortho_center_down()
                    column_error += self.perform_yang_baxter(self.ortho_center - 1, self.ortho_center, self.get_index(x, y, 1), flip=True, arrows_should_point_up=False)
        self.ortho_surface -= 1
        if self.debug_logger.log_approximate_column_error_yb:
            self.debug_logger.append_to_log_list("approximate_column_errors", column_error)
        # Optionally perform variational column optimization and/or log the column error
        if self.ortho_surface > 0 and self.perform_variational_column_optimization or self.debug_logger.log_column_error_yb_before_variational_optimization:
            Ts_after_YB = []
            Ws_after_YB = []
            ortho_center = self.ortho_center
            if p == 0:
                Ts_after_YB = [utility.flip_T_square(self.Ts[self.get_index(x, y, 0)].copy()) for y in range(self.Ly)]
                Ws_after_YB = [None] + [utility.flip_W(W.copy()) for W in self.Ws]
                ortho_center += 1
            else:
                Ts_after_YB = [utility.flip_T_square(self.Ts[self.get_index(x, y, 1)].copy()) for y in range(self.Ly)]
                Ws_after_YB = [utility.flip_W(W.copy()) for W in self.Ws] + [None]
            optimizer = column_optimization.variationalColumnOptimizer(Ts_before_YB, Ws_before_YB, Ts_after_YB, Ws_after_YB, ortho_center, self.variational_column_optimization_options["mode"], debug_logger=self.debug_logger)
            if self.perform_variational_column_optimization:
                # Optimize column
                if self.debug_logger.log_algorithm_walltimes:
                    start = time.time()
                optimizer.optimize_column(self.variational_column_optimization_options["N_sweeps"])
                if self.debug_logger.log_algorithm_walltimes:
                    end = time.time()
                    self.time_counter_variational_column_optimization += end-start
                if p == 0:
                    for i in range(2*self.Ly-1):
                        self.Ws[i] = utility.flip_W(optimizer.Ws[i+1])
                    for y in range(self.Ly):
                        self.Ts[self.get_index(x, y, 0)] = utility.flip_T_square(optimizer.Ts[y])
                else:
                    for i in range(2*self.Ly-1):
                        self.Ws[i] = utility.flip_W(optimizer.Ws[i])
                    for y in range(self.Ly):
                        self.Ts[self.get_index(x, y, 1)] = utility.flip_T_square(optimizer.Ts[y])
            else:
                # Just compute the error for debug logging
                self.debug_logger.append_to_log_list("column_errors_yb_before_variational_optimization", optimizer.compute_error())


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
        # If we want to perform a variational column optimization or compute the column error, 
        # we need to store the column's tensors before doing any YB moves
        Ts_before_YB = []
        Ws_before_YB = []
        if self.ortho_surface < 2 * self.Lx - 2 and (self.perform_variational_column_optimization or self.debug_logger.log_column_error_yb_before_variational_optimization):
            if p == 0:
                Ts_before_YB = [self.Ts[self.get_index(x, y, 1)].copy() for y in range(self.Ly)]
                Ws_before_YB = [W.copy() for W in self.Ws] + [None]
            else:
                Ts_before_YB = [self.Ts[self.get_index(x+1, y, 0)].copy() for y in range(self.Ly)]
                Ws_before_YB = [None] + [W.copy() for W in self.Ws]
        if move_upwards:
            # First, move the ortho center to the bottom
            self.move_ortho_center_to(0)
            # Now, use the yang baxter move to shift the ortho surface, one tensor at a time
            if p == 0:
                for y in range(self.Ly - 1):
                    column_error += self.perform_yang_baxter(self.ortho_center, self.ortho_center + 1, self.get_index(x, y, 1), flip=False)
                    self.move_ortho_center_up()
                column_error +=  self.perform_yang_baxter(self.ortho_center, None, self.get_index(x, self.Ly - 1, 1), flip=False)
            else:
                column_error += self.perform_yang_baxter(None, self.ortho_center, self.get_index(x + 1, 0, 0), flip=False)
                for y in range(1, self.Ly):
                    self.move_ortho_center_up()
                    column_error += self.perform_yang_baxter(self.ortho_center, self.ortho_center + 1, self.get_index(x + 1, y, 0), flip=False)
        else:
            # First, move the ortho center to the top
            self.move_ortho_center_to(2*self.Ly - 2)
            # Now, use the yang baxter move to shift the ortho surface, one tensor at a time
            if p == 0:
                column_error += self.perform_yang_baxter(self.ortho_center, None, self.get_index(x, self.Ly - 1, 1), flip=False, arrows_should_point_up=False)
                for y in range(self.Ly-2, -1, -1):
                    self.move_ortho_center_down()
                    column_error += self.perform_yang_baxter(self.ortho_center - 1, self.ortho_center, self.get_index(x, y, 1), flip=False, arrows_should_point_up=False)
            else:
                for y in range(self.Ly-1, 0, -1):
                    column_error += self.perform_yang_baxter(self.ortho_center - 1, self.ortho_center, self.get_index(x + 1, y, 0), flip=False, arrows_should_point_up=False)
                    self.move_ortho_center_down()
                column_error += self.perform_yang_baxter(None, self.ortho_center, self.get_index(x + 1, 0, 0), flip=False, arrows_should_point_up=True)
        self.ortho_surface += 1
        if self.debug_logger.log_approximate_column_error_yb:
            self.debug_logger.append_to_log_list("approximate_column_errors", column_error)
        # Optionally perform variational column optimization and/or log the column error
        if self.ortho_surface < 2*self.Ly - 2 and self.perform_variational_column_optimization or self.debug_logger.log_column_error_yb_before_variational_optimization:
            Ts_after_YB = []
            Ws_after_YB = []
            ortho_center = self.ortho_center
            if p == 0:
                Ts_after_YB = [self.Ts[self.get_index(x, y, 1)].copy() for y in range(self.Ly)]
                Ws_after_YB = [W.copy() for W in self.Ws] + [None]
            else:
                Ts_after_YB = [self.Ts[self.get_index(x+1, y, 0)].copy() for y in range(self.Ly)]
                Ws_after_YB = [None] + [W.copy() for W in self.Ws]
                ortho_center += 1
            optimizer = column_optimization.variationalColumnOptimizer(Ts_before_YB, Ws_before_YB, Ts_after_YB, Ws_after_YB, ortho_center, self.variational_column_optimization_options["mode"], debug_logger=self.debug_logger)
            if self.perform_variational_column_optimization:
                # Optimize column
                if self.debug_logger.log_algorithm_walltimes:
                    start = time.time()
                optimizer.optimize_column(self.variational_column_optimization_options["N_sweeps"])
                if self.debug_logger.log_algorithm_walltimes:
                    end = time.time()
                    self.time_counter_variational_column_optimization += end-start
                if p == 0:
                    for i in range(2*self.Ly-1):
                        self.Ws[i] = optimizer.Ws[i]
                    for y in range(self.Ly):
                        self.Ts[self.get_index(x, y, 1)] = optimizer.Ts[y]
                else:
                    for i in range(2*self.Ly-1):
                        self.Ws[i] = optimizer.Ws[i+1]
                    for y in range(self.Ly):
                        self.Ts[self.get_index(x+1, y, 0)] = optimizer.Ts[y]
            else:
                # Just compute the error for debug logging
                self.debug_logger.append_to_log_list("column_errors_yb_before_variational_optimization", optimizer.compute_error())

    def get_environment_twosite(self):
        r"""
        Returns the environment around the current ortho center,
        consisting of the tensors T1, T2, Wm1, W, Wp1.
        Tensors are flipped such that they satisfy the following structure:
        
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

        This structure is also expected in the two-site operations in expectation_values.py
        and tebd_qr.py.

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
        flip = False
        
        if px == 0:
            if py == 0:
                T1 = self.Ts[self.get_index(x, y, 0)]
                T2 = self.Ts[self.get_index(x, y, 1)]
                flip = False
            else:
                T1 = self.Ts[self.get_index(x, y, 1)]
                T2 = self.Ts[self.get_index(x, y + 1, 0)]
                flip = True
        else:
            if py == 0:
                T1 = self.Ts[self.get_index(x + 1, y, 0)]
                T2 = self.Ts[self.get_index(x, y, 1)]
                flip = True
            else:
                T1 = self.Ts[self.get_index(x, y, 1)]
                T2 = self.Ts[self.get_index(x + 1, y + 1, 0)]
                flip = False
        
        Wm1 = None
        if self.ortho_center > 0:
            Wm1 = self.Ws[self.ortho_center - 1]
        W = self.Ws[self.ortho_center]
        Wp1 = None
        if self.ortho_center < len(self.Ws) - 1:
            Wp1 = self.Ws[self.ortho_center + 1]

        if flip:
            T1, T2, Wm1, W, Wp1 = utility.flip_twosite_square(T1, T2, Wm1, W, Wp1)

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

        # if parity is odd, we need to flip!
        if px != py:
            T1, T2, Wm1, W, Wp1 = utility.flip_twosite_square(T1, T2, Wm1, W, Wp1)
        
        # Set T tensors
        if px == 0:
            if py == 0:
                if T1 is not None:
                    self.Ts[self.get_index(x, y, 0)] = T1
                if T2 is not None:
                    self.Ts[self.get_index(x, y, 1)] = T2
            else:
                if T1 is not None:
                    self.Ts[self.get_index(x, y, 1)] = T1
                if T2 is not None: 
                    self.Ts[self.get_index(x, y + 1, 0)] = T2
        else:
            if py == 0:
                if T1 is not None:
                    self.Ts[self.get_index(x + 1, y, 0)] = T1
                if T2 is not None:
                    self.Ts[self.get_index(x, y, 1)] = T2
            else:
                if T1 is not None:
                    self.Ts[self.get_index(x, y, 1)] = T1
                if T2 is not None:
                    self.Ts[self.get_index(x + 1, y + 1, 0)] = T2

        # Set W tensors
        if self.ortho_center > 0:
            self.Ws[self.ortho_center - 1] = Wm1
        self.Ws[self.ortho_center] = W
        if self.ortho_center < len(self.Ws) - 1:
            self.Ws[self.ortho_center + 1] = Wp1

    def get_environment_onesite(self, left_environment=True):
        r"""
        Returns the one-site environment next to the current ortho center.
        Either returns the left or the right environment.

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

        The tensors are flipped s.t. they are in the form expected by compute_norm_onesite() and
        compute_expectation_values_onesite(). The orthogonality center might be either at W or Wp1.
        
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

        if left_environment:
            if px == 0:
                if py == 0:
                    if self.ortho_center > 0:
                        W = self.Ws[self.ortho_center - 1]
                    Wp1 = self.Ws[self.ortho_center]
                    T = self.Ts[self.get_index(x, y, 0)]
                else:
                    W = self.Ws[self.ortho_center]
                    Wp1 = self.Ws[self.ortho_center + 1]
                    T = self.Ts[self.get_index(x, y + 1, 0)]
            else:
                if py == 0:
                    W = self.Ws[self.ortho_center]
                    if self.ortho_center + 1 < len(self.Ws):
                        Wp1 = self.Ws[self.ortho_center + 1]
                    T = self.Ts[self.get_index(x, y, 1)]
                else:
                    W = self.Ws[self.ortho_center - 1]
                    Wp1 = self.Ws[self.ortho_center]
                    T = self.Ts[self.get_index(x, y, 1)]
        else:
            if px == 0:
                if py == 0:
                    W = self.Ws[self.ortho_center]
                    if self.ortho_center + 1 < len(self.Ws):
                        Wp1 = self.Ws[self.ortho_center + 1]
                    T = self.Ts[self.get_index(x, y, 1)]
                else:
                    W = self.Ws[self.ortho_center]
                    Wp1 = self.Ws[self.ortho_center + 1]
                    T = self.Ts[self.get_index(x, y, 1)]
            else:
                if py == 0:
                    if self.ortho_center > 0:
                        W = self.Ws[self.ortho_center - 1]
                    Wp1 = self.Ws[self.ortho_center]
                    T = self.Ts[self.get_index(x + 1, y, 0)]
                else:
                    W = self.Ws[self.ortho_center]
                    Wp1 = self.Ws[self.ortho_center + 1]
                    T = self.Ts[self.get_index(x + 1, y + 1, 0)]

        if not left_environment:
            T, W, Wp1 = utility.flip_onesite_square(T, W, Wp1)

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
        for x in range(self.Lx):
            self.move_to(2*x, 0)
            T, W, Wp1 = self.get_environment_onesite(left_environment=True)
            for i, op in enumerate(ops):
                result[i].append(expectation_values.expectation_value_onesite(T, W, Wp1, op))
            left_environment = False
            for y in range(2*self.Ly - 1):
                T, W, Wp1 = self.get_environment_onesite(left_environment=left_environment)
                for i, op in enumerate(ops):
                    result[i].append(expectation_values.expectation_value_onesite(T, W, Wp1, op))
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
                T1, T2, Wm1, W, Wp1 = self.get_environment_twosite()
                result.append(expectation_values.expectation_value_twosite(T1, T2, Wm1, W, Wp1, ops[xi * (2 * self.Ly - 1) + yi]))
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
            the error of the local TEBD update. If self.debug_logger.log_approximate_column_error_tebd == False,
            -np.float("inf") is returned instead.
        """
        # determine index into U_bonds
        index = self.ortho_surface * (2 * self.Ly - 1) + self.ortho_center
        # retreive environment
        T1, T2, Wm1, W, Wp1 = self.get_environment_twosite()
        # Save environment tensors (debug)
        if self.debug_logger.log_local_tebd_update_environments:
            self.debug_logger.append_to_log_list("local_tebd_update_environments", {"T1": T1.copy(), "T2": T2.copy(), "Wm1": None if Wm1 is None else Wm1.copy(), "W": None if W is None else W.copy(), "Wp1": None if Wp1 is None else Wp1.copy()})
        # Stop time (debug)
        if self.debug_logger.log_algorithm_walltimes or self.debug_logger.log_local_tebd_update_walltimes:
            start = time.time()
        # perform TEBD step
        T1, T2, Wm1, W, Wp1, error = tebd.tebd_step(T1, T2, Wm1, W, Wp1, U_bonds[index], self.chi_max, debug_logger=self.debug_logger, **self.tebd_options)
        error = np.real_if_close(error)
        # log error and walltimes
        if self.debug_logger.log_algorithm_walltimes or self.debug_logger.log_local_tebd_update_walltimes:
            end = time.time()
        if self.debug_logger.log_local_tebd_update_errors:
            self.debug_logger.append_to_log_list("local_tebd_update_errors", error)
        if self.debug_logger.log_local_tebd_update_walltimes:
            self.debug_logger.append_to_log_list("local_tebd_update_walltimes", end-start)
        if self.debug_logger.log_algorithm_walltimes:
            self.time_counter_tebd_local_update += end-start
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
        for _ in range(0, 2 * self.Ly - 1, 2):
            column_error += self.perform_TEBD_at_ortho_center(U_bonds)
            self.move_ortho_center_up()
            self.move_ortho_center_up()
        # apply update on odd bonds
        if self.Ly > 1:
            self.move_ortho_center_to(2*self.Ly-3)
            for _ in range(2*self.Ly-3, 0, -2):
                column_error += self.perform_TEBD_at_ortho_center(U_bonds)
                self.move_ortho_center_down()
                self.move_ortho_center_down()
        if self.debug_logger.log_approximate_column_error_tebd:
            self.debug_logger.append_to_log_list("approximate_column_error_tebd", column_error)

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
            for _ in range(2*self.Ly-1):
                column_error += self.perform_TEBD_at_ortho_center(U_bonds)
                self.move_ortho_center_up() 
        else:
            self.move_ortho_center_to(2*self.Ly-2)
            for _ in range(2*self.Ly-1):
                column_error += self.perform_TEBD_at_ortho_center(U_bonds)
                self.move_ortho_center_down() 
        if self.debug_logger.log_approximate_column_error_tebd:
            self.debug_logger.append_to_log_list("approximate_column_error_tebd", column_error)

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
            self.reset_time_counters()
            # apply update on even surfaces
            self.move_ortho_surface_to(0)
            for _ in range(0, 2 * self.Lx - 1, 2):
                self.perform_TEBD_along_ortho_surface_brickwall(U_bonds)
                self.move_ortho_surface_right()
                self.move_ortho_surface_right()
            # apply update on odd surfaces
            if self.Lx > 1:
                self.move_ortho_surface_to(2*self.Lx-3)
                for _ in range(2*self.Lx-3, 0, -2):
                    self.perform_TEBD_along_ortho_surface_brickwall(U_bonds)
                    self.move_ortho_surface_left()
                    self.move_ortho_surface_left()
            # Debug logging
            if self.debug_logger.log_algorithm_walltimes:
                self.debug_logger.append_to_log_list(("algorithm_walltimes", "yb_move"), self.time_counter_yb_move)
                self.debug_logger.append_to_log_list(("algorithm_walltimes", "local_tebd_update"), self.time_counter_tebd_local_update)
                self.debug_logger.append_to_log_list(("algorithm_walltimes", "variational_column_optimization"), self.time_counter_variational_column_optimization)


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
            self.reset_time_counters()
            self.move_to(0, 0)
            for _ in range(2*self.Lx-1):
                self.perform_TEBD_along_ortho_surface_chain(U_bonds, move_upwards=True)
                self.move_ortho_surface_right(move_upwards=False)
            for _ in range(2*self.Lx-1):
                self.perform_TEBD_along_ortho_surface_chain(U_bonds, move_upwards=False)
                self.move_ortho_surface_left(move_upwards=True)
            # Debug logging
            if self.debug_logger.log_algorithm_walltimes:
                self.debug_logger.append_to_log_list(("algorithm_walltimes", "yb_move"), self.time_counter_yb_move)
                self.debug_logger.append_to_log_list(("algorithm_walltimes", "local_tebd_update"), self.time_counter_tebd_local_update)
                self.debug_logger.append_to_log_list(("algorithm_walltimes", "variational_column_optimization"), self.time_counter_variational_column_optimization)

    def compute_singular_values_at_bond(self):
        """
        Computes the singular values between the tensor at the orthogonality center and the tensor one above the 
        orthogonality center.

        Returns
        -------
        S : np.ndarray of shape (chi)
            singular values
        """
        assert(self.ortho_center < len(self.Ws) - 1)
        contr = np.tensordot(self.Ws[self.ortho_center], self.Ws[self.ortho_center + 1], ([1], [3])) # l [u] r d; lp1 up1 rp1 [dp1] -> l r d lp1 up1 rp1
        l, r, d, lp1, up1, rp1 = contr.shape
        contr = np.reshape(contr, (l*r*d, lp1*up1*rp1)) # l, r, d, lp1, up1, rp1 -> (l, r, d), (lp1, up1, rp1)
        _, S, _ = utility.safe_svd(contr)
        return S

    def compute_singular_values_at_all_bonds(self):
        """
        Computes the singular values along the orthogonality hypersurface for all possible positions
        of the orthogonality center.

        Returns
        -------
        Ss : list of np.ndarray
            list containing the singular values between the orthogonality center and the tensor one above the 
        orthogonality center for all possible positions. len(Ss) = (2*self.Lx - 1) * (2 * self.Ly - 2).
        """
        Ss = []
        for i in range(2 * self.Lx - 1):
            for j in range(2 * self.Ly - 2):
                self.move_to(i, j)
                S = self.compute_singular_values_at_bond()
                Ss.append(S)
        return Ss

    def evaluate_wave_function(self, state):
        """
        Evaluates a single entry of the wave function vector, specified by state.
        The entry is computed by a full contraction of the tensor network, This costs
        O(L_x * D^(2*L_y + 1)) and is thus only feasable for small networks.

        Parameters
        ----------
        state : list of int
            list selecting the basis vector of the local hilbert space for each T_index.
            the function expects len(list) == 2*self.Lx*self.Ly and state[T_index] in {0, ..., self.d-1}
            for all T_index.

        Returns
        -------
        result : complex
            computed wave function entry.
        """
        # Note: In the following comments, let N = 2*Ly - 1.
        # Move to bottom left
        self.move_to(0, 0)
        # begin building the psi tensor by contracting leftmost T-tensors with the ortho surface
        T_index = self.get_index(0, 0, 0)
        psi = np.tensordot(self.Ts[T_index][state[T_index], :, 0, 0, 0], self.Ws[0][:, :, :, 0], ([0], [0])) # [ru]; [l] u r -> u r_1
        psi = np.tensordot(psi, self.Ws[1], ([0], [3])) # [u] r_1; l u r [d] -> r_1 l u r_2
        for y in range(1, self.Ly - 1):
            T_index = self.get_index(0, y, 0)
            psi = np.tensordot(psi, self.Ts[T_index][state[T_index], :, :, 0, 0], ([-3], [1])) # r_1 ... r_{i-1} [l] u r_i; ru [rd] -> r_1 ... r_{i-1} u r_i ru
            psi = np.tensordot(psi, self.Ws[2*y], ([-3, -1], [3, 0])) #r_1 ... r_{i-1} [u] r_i [ru]; [l] u r [d] -> r_1 ... r_{i-1} r_i u r_{i+1}
            psi = np.tensordot(psi, self.Ws[2*y+1], ([-2], [3])) # r_1 ... r_{i-1} r_i [u] r_{i+1}; l u r [d] -> r_1 ... r_{i+1} l u r_{i+2}
        # Contract psi tensor with final T and W tensor
        T_index = self.get_index(0, self.Ly-1, 0)
        psi = np.tensordot(psi, self.Ts[T_index][state[T_index], :, :, 0, 0], ([-3], [1])) # r_1 ... r_{i-1} [l] u r_i; ru [rd] -> r_1 ... r_{i-1} u r_i ru
        psi = np.tensordot(psi, self.Ws[2*self.Ly-2][:, 0, :, :], ([-3, -1], [2, 0])) #r_1 ... r_{i-1} [u] r_i [ru]; [l] r [d] -> r_1 ... r_{N-2} r_{N-1} r_N
        # Now go from left to right and contract everything with the psi tensor column after column, except for the final column
        x = 0
        for _ in range(self.Lx - 1):
            # Contract with p=1 tensors, going from top to bottom. The first tensor is special
            T_index = self.get_index(x, self.Ly-1, 1)
            psi = np.tensordot(self.Ts[T_index][state[T_index], 0, :, :, 0], psi, ([1], [-1])) # rd [ld]; r_1 ... r_{N-1} [r_{N}] -> r_N r_1 ... r_{N-1}
            for y in range(self.Ly - 2, -1, -1):
                T_index = self.get_index(x, y, 1)
                T = self.Ts[T_index][state[T_index], :, :, :, :].transpose(1, 0, 2, 3) # ru, rd, ld, lu -> rd, ru, ld, lu
                psi = np.tensordot(T, psi, ([2, 3], [-2, -1])) # rd ru [ld] [lu]; r_{i+1} r_{i+2} ... r_{N-1} r_N r_1 ... [r_{i-1}] [r_i] -> r_{i-1} r_i r_{i+1} ... r_N r_1 ... r_{i-2}
            # After this for-loop, the psi tensor has again shape r_1 r_2 ... r_N
            x += 1
            # Now, contract with p=0 tensors, going from top to bottom. The last tensor is special
            for y in range(self.Ly - 1, 0, -1):
                T_index = self.get_index(x, y, 0)
                T = self.Ts[T_index][state[T_index], :, :, :, :].transpose(1, 0, 2, 3) # ru, rd, ld, lu -> rd, ru, ld, lu
                psi = np.tensordot(T, psi, ([2, 3], [-2, -1])) # rd ru [ld] [lu]; r_{i+1} r_{i+2} ... r_{N-1} r_N r_1 ... [r_{i-1}] [r_i] -> r_{i-1} r_i r_{i+1} ... r_N r_1 ... r_{i-2}
            # last tensor is special
            T_index = self.get_index(x, 0, 0)
            T = self.Ts[T_index][state[T_index], :, 0, 0, :] # ru, lu
            psi = np.tensordot(T, psi, ([1], [-1])) # r_1 [l_1]; r_2 ... r_N [r_1] -> r_1 r_2 ... r_N 
        # Contract with the final column of p=1 tensors, from bottom to top
        for y in range(self.Ly - 1):
            T_index = self.get_index(x, y, 1)
            psi = np.tensordot(psi, self.Ts[T_index][state[T_index], 0, 0, :, :], ([0, 1], [0, 1])) # [ld] [lu]; r_{i-1} r_{i} r_{i+1} ... r_N -> r_{i+1} r_{i+2} ... r_N
        # Last tensor is special
        T_index = self.get_index(x, self.Ly - 1, 1)
        psi = np.tensordot(self.Ts[T_index][state[T_index], 0, 0, :, 0], psi, ([0], [0])) # [ld]; [r_N]
        return psi.item()