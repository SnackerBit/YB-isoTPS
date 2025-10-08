from ..utility import debug_logging
from ..utility import utility
from ..utility import backend
from . import shifting_ortho_center
import h5py

class isoTPS:
    """
    Base 2D isoTPS class. Implements functionality that is lattice independant. Sub classes like isoTPS_Square and
    isoTPS_Honeycomb implement remaining lattice dependant functionality. 
    """

    def __init__(self, Lx, Ly, D_max=4, chi_factor=6, chi_max=None, d=2, shifting_options={ "mode" : "svd" }, yb_options={ "mode" : "svd" }, tebd_options={ "mode" : "svd" }, ordering_mode="center", perform_variational_column_optimization=False, variational_column_optimization_options={}, debug_logger_options={}):
        """
        Initializes the isoTPS.

        Parameters
        ----------
        Lx, Ly : int
            system size
        D_max : int, optional
            maximal bond dimension of the virtual legs of the physical site tensors ("T-tensors"). Default value: 4.
        chi_factor : int or None, optional
            factor with which the maximal bond dimension of the orthogonality hypersurface is computed, chi_max = D_max*chi_factor.
            This is only used if chi_max is set to None.
            Default: chi_factor = 6.
        chi_max : int or None, optional
            maximal bond dimension of the orthogonality hypersurface. If this is set to None, chi_max is computed
            according to chi_factor. Default: None.
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
            options for performing TEBD. See "src/isoTPS/square/tebd.py" or "src/isoTPS/honeycomb/tebd.py" for more information.
            Default: { "mode" : "svd" }.
        ordering_mode : str, optional
            string specifying the rule for choosing how bond dimension is distributed between the bonds when performing Yang-Baxter moves.
            For more information on allowed orering_mode strings, see the __init__ method of the sub class of isoTPS. Default: "center". 
        perform_variational_column_optimization : bool, optional
            wether to perform a variational column optimization after shifting the column with YB moves. Default: False.       
        variational_column_optimization_options : dict, optional
            options for performing the variational column optimization. See "src/isoTPS/square/column_optimization.py" for more information.
            Default: {}.
        debug_logger_options: dict, optional
            dictionary containing kwargs that get passed into the constructor of the DebugLogger instance used in this isoTPS.
            This can be used to turn on certain parts of the debug logging. See "utility/debug_logging.py" for more information. Default: {}.
        """
        # System size
        self.Lx = Lx
        self.Ly = Ly
        # Maximal bond dimensions
        self.chi_max = chi_max
        self.chi_factor = chi_factor
        if chi_max is None:
            if chi_factor is None:
                raise ValueError("Both chi_max and chi_factor cannot be None at the same time.")
            self.chi_max = D_max * chi_factor
        self.D_max = D_max
        self.d = d
        # the shifting mode used in shifting_ortho_center.move_ortho_center_up() and shifting_ortho_center.move_ortho_center_down()
        self.shifting_options = shifting_options
        # options for the yang baxter move
        self.yb_options = yb_options
        # options for the tebd step
        self.tebd_options = tebd_options
        # bond ordering
        self.ordering_mode = ordering_mode
        # variational column optimization
        self.perform_variational_column_optimization = perform_variational_column_optimization
        self.variational_column_optimization_options = variational_column_optimization_options
        # debug logger
        self.debug_logger_options = debug_logger_options
        self.debug_logger = debug_logging.DebugLogger(**debug_logger_options)
        # logging variables for keeping track of time spent in subroutines
        self.time_counter_yb_move = 0.0
        self.time_counter_variational_column_optimization = 0.0
        self.time_counter_tebd_local_update = 0.0
        # Initialize internal variables
        self.Ts = [None] * Lx * Ly * 2
        self.Ws = [None] * (Ly * 2 - 1)
        self.ortho_surface = 0
        self.ortho_center = 0

    def reset_time_counters(self):
        """
        Resets the logging variables that keep track of the time spent in subroutines
        """
        self.time_counter_yb_move = 0.0
        self.time_counter_tebd_local_update = 0.0
        self.time_counter_variational_column_optimization = 0.0

    def save_to_file(self, filename, data=None):
        """
        Saves this isoTPS to a file.
        
        Parameters
        ----------
        filename : str
            file name
        data : dict or None, optional
            dictionary in which to save the isoTPS. May already contain additional data, when this function
            is overwritten and called from a derived class. If set to None, a new, empty dictionary will
            be used to save the isoTPS. Default: None.
        """
        # Prepare data dictionary
        if data is None:
            data = {}
        data["Lx"] = self.Lx
        data["Ly"] = self.Ly
        data["chi_max"] = self.chi_max
        data["chi_factor"] = self.chi_factor
        data["D_max"] = self.D_max
        data["d"] = self.d
        data["shifting_options"] = self.shifting_options
        data["yb_obtions"] = self.yb_options
        data["tebd_options"] = self.tebd_options
        data["ordering_mode"] = self.ordering_mode
        data["perform_variational_column_optimization"] = self.perform_variational_column_optimization
        data["variational_column_optimization_options"] = self.variational_column_optimization_options
        data["debug_logger_options"] = self.debug_logger_options
        data["Ts"] = {}
        for i, T in enumerate(self.Ts):
            data["Ts"][f"T{i}"] = T
        data["Ws"] = {}
        for i, W in enumerate(self.Ws):
            data["Ws"][f"W{i}"] = W
        data["ortho_surface"] = self.ortho_surface
        data["ortho_center"] = self.ortho_center
        # Save to .h5 file
        with h5py.File(filename, "w") as hf:
            utility.dump_dict_into_hf(hf, data)
            self.debug_logger.save_to_file_h5(hf)

    def _load_from_dict(self, data, load_debug_log=True):
        """
        Loads an isoTPS from a dictionary. This should only be called by derived classes.

        Parameters
        ----------
        data : dict
            dictionary containing isoTPS data.
        load_debug_log : bool, optional
            wether or not to load the debug log from the file. Default: True 
        """
        self.Lx = data["Lx"]
        self.Ly = data["Ly"]
        self.chi_max = data["chi_max"]
        self.chi_factor = data["chi_factor"]
        self.D_max = data["D_max"]
        self.d = data["d"]
        self.shifting_options = data["shifting_options"]
        self.yb_options = data["yb_obtions"]
        self.tebd_options = data["tebd_options"]
        self.ordering_mode = data["ordering_mode"]
        self.perform_variational_column_optimization = data["perform_variational_column_optimization"]
        self.variational_column_optimization_options = data["variational_column_optimization_options"]
        self.debug_logger = debug_logging.DebugLogger(data["debug_logger_options"])
        self.Ts = [None] * self.Lx * self.Ly * 2
        for i in range(self.Lx * self.Ly * 2):
            self.Ts[i] = data["Ts"][f"T{i}"]
        self.Ws = [None] * (self.Ly * 2 - 1)
        for i in range(self.Ly * 2 - 1):
            self.Ws[i] = data["Ws"][f"W{i}"]
        self.ortho_surface = data["ortho_surface"]
        self.ortho_center = data["ortho_center"]
        if load_debug_log:
            self.debug_logger.load_log_dict(data["debug_log"])

    def _init_as_copy(self, original_tps):
        """
        Helper function that initializes the tensors and orthogonality surface/center of this isoTPS as a copy of the given original tps

        Parameters
        ----------
        original_tps : instance of isoTPS class or sub class
            the tps that is to be copied
        """
        for i in range(len(original_tps.Ts)):
            self.Ts[i] = original_tps.Ts[i].copy()
        for i in range(len(original_tps.Ws)):
            self.Ws[i] = original_tps.Ws[i].copy()
        self.ortho_surface = original_tps.ortho_surface
        self.ortho_center = original_tps.ortho_center

    def get_index(self, x, y, p):
        """
        Returns the index of the T tensor in unit cell (x, y), with p in {0, 1} being the index into the unit cell.

        Parameters
        ----------
        x: int
            horizontal coordinate of the unit cell
        y: int
            vertical coordinate of the unit cell
        p : int
            index into the unit cell
        
        Returns
        -------
        T_index : int
            index into the T tensor list self.Ts
        """
        return (x * self.Ly + y) * 2 + p

    def get_position(self, T_index):
        """
        Returns x, y, and p, specifying the position of the T tensor index, from the index into the T tensor list

        Parameters
        ----------
        T_index : int
            index into the T tensor list self.Ts

        Returns
        -------
        x: int
            horizontal coordinate of the unit cell
        y: int
            vertical coordinate of the unit cell
        p : int
            index into the unit cell
        """
        p = T_index % 2
        temp = (T_index - p) // 2 
        x = temp // self.Ly
        y = temp % self.Ly
        return x, y, p

    def is_in_upper_half(self, T_index):
        """
        Returns true if the given T tensor index is in the upper half of the isoTPS.
        Note: This function may need to be overwritten by some derived classes for isoTPS on different lattices.
        
        Parameters
        ----------
        T_index : int
            index into self.Ts
        
        Returns
        -------
        result : bool
            wether the T tensor at T_index is in the upper half of the isoTPS
        """
        p = T_index % 2
        temp = (T_index - p) // 2 
        y = temp % self.Ly
        if self.Ly % 2 == 0:
            return y >= self.Ly // 2
        else:
            return y >= self.Ly // 2 + 1 or (y == self.Ly // 2 and p == 1)

    def initialize_product_state(self, states):
        """
        Initializes the isoTPS tensors in the given product state
        Note: This function should be overwritten in derived classes, as it is lattice specific!

        Parameters
        ----------
        states : list of backend.array_type of shape (d,)
            list of local states. The full many-body state is formed by the kronecker product of all states in the list.
        """
        raise NotImplementedError("function initialize_product_state() is not implemented in isoTPS base class!")

    def initialize_spinup(self):
        """
        Initializes the isoTPS tensors in a product state consisting of S=1/2 spins pointing up
        """
        assert(self.d == 2)
        self.initialize_product_state([backend.array([1.0, 0.0], dtype=backend.dtype_complex)] * len(self.Ts))

    def initialize_spindown(self):
        """
        Initializes the isoTPS tensors in a product state consisting of S=1/2 spins pointing down
        """
        assert(self.d == 2)
        self.initialize_product_state([backend.array([0.0, 1.0], dtype=backend.dtype_complex)] * len(self.Ts))

    def initialize_spinright(self):
        """
        Initializes the isoTPS tensors in a product state consisting of S=1/2 spins pointing to the right
        """
        assert(self.d == 2)
        self.initialize_product_state([backend.array([1.0, 1.0], dtype=backend.dtype_complex)/backend.sqrt(2.0)] * len(self.Ts))

    def initialize_spinleft(self):
        """
        Initializes the isoTPS tensors in a product state consisting of S=1/2 spins pointing to the left
        """
        assert(self.d == 2)
        self.initialize_product_state([backend.array([1.0, -1.0], dtype=backend.dtype_complex)/backend.sqrt(2.0)] * len(self.Ts))

    def initialize_random_spin_half_product_state(self):
        """
        Initializes the isoTPS tensors in a product state consisting of S=1/2 spins pointing in random directions
        """
        assert(self.d == 2)
        state = []
        for i in range(len(self.Ts)):
            local_state = backend.random(self.d) + 1.j * backend.random(self.d)
            local_state /= backend.norm(local_state)
            state.append(local_state)
        self.initialize_product_state(state)

    def print_shapes(self):
        """
        Prints the shapes of all T and W tensors (used for debugging)
        """
        for i in range(len(self.Ts)):
            if self.Ts[i] is None:
                print(f"T[{i}] = None")
            else:
                print(f"T[{i}] = {self.Ts[i].shape}")
        for i in range(len(self.Ws)):
            if self.Ws[i] is None:
                print(f"W[{i}] = None")
            else:
                print(f"W[{i}] = {self.Ws[i].shape}")

    def check_isometry_condition(self):
        """
        If the isometry condition is not satisfied at any tensor, an error
        message will be printed to the console and false is returned.
        If the isometry condition is fullfilled, True is returned instead.
        Note: This function should be overwritten in derived classes, as it is lattice specific!

        Returns
        -------
        success : bool
            wether the isoTPS fulfills the isometry condition
        """
        raise NotImplementedError("function check_isometry_condition() is not implemented in isoTPS base class!")

    def check_leg_dimensions(self):
        """
        If the bond dimensions of all legs match up, True is returned.
        If an error is detected, it is printed to the console where the error
        happened and which bond dimensions were expected.
        Note: This function should be overwritten in derived classes, as it is lattice specific!

        Returns
        -------
        success : bool
            wether all leg dimensions in the isoTPS match up
        """
        raise NotImplementedError("function check_leg_dimensions() is not implemented in isoTPS base class!")

    def sanity_check(self):
        """
        Checks if the isometry condition is fulfilled for the entire TPS,
        and if the bond dimensions of all legs match. Returns True on success
        and False if an error is found. If an error is found, a message is
        printed to the console specifying what error occurred where in the TPS.

        Returns
        -------
        success : bool
            wether the sanity check was successfull
        """
        success = True
        if not self.check_isometry_condition():
            success = False
        if not self.check_leg_dimensions():
            success = False
        return success

    def move_ortho_center_up(self):
        """
        Moves the orthogonality center one tensor upwards along the orthogonality surface.
        Note: This function may need to be overwritten by some derived classes for isoTPS on different lattices.
        """
        if self.ortho_center >= 2 * self.Ly - 2:
            return
        self.Ws[self.ortho_center], self.Ws[self.ortho_center + 1] = shifting_ortho_center.move_ortho_center_up(self.Ws[self.ortho_center], self.Ws[self.ortho_center + 1], chi_max=self.chi_max, options=self.shifting_options)
        self.ortho_center += 1

    def move_ortho_center_down(self):
        """
        Moves the orthogonality center one tensor downwards along the orthogonality surface.
        Note: This function may need to be overwritten by some derived classes for isoTPS on different lattices.
        """
        if self.ortho_center == 0:
            return
        self.Ws[self.ortho_center - 1], self.Ws[self.ortho_center] = shifting_ortho_center.move_ortho_center_down(self.Ws[self.ortho_center - 1], self.Ws[self.ortho_center], chi_max=self.chi_max, options=self.shifting_options)
        self.ortho_center -= 1

    def move_ortho_surface_left(self, force=False, move_upwards=True):
        """
        Moves the orthogonality hypersurface one column to the left.
        Note: This function should be overwritten in derived classes, as it is lattice specific!

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
        raise NotImplementedError("function move_ortho_surface_left() is not implemented in isoTPS base class!")

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
        raise NotImplementedError("function move_ortho_surface_right() is not implemented in isoTPS base class!")

    def move_ortho_surface_to(self, ortho_surface):
        """
        Moves the orthogonality hypersurface to the given index

        Parameters
        ----------
        ortho_surface : int
            the index the orthogonality hypersurface is to be moved to
        """
        ortho_surface = max(0, min(self.Lx*2-2, ortho_surface))
        while self.ortho_surface < ortho_surface:
            self.move_ortho_surface_right()
        while self.ortho_surface > ortho_surface:
            self.move_ortho_surface_left()

    def move_ortho_center_to(self, ortho_center):
        """
        Moves the orthogonality center to the given index along the current orthogonality hypersurface.

        Parameters
        ----------
        ortho_center : int
            the index the orthogonality center is to be moved to
        """
        ortho_center = max(0, min(self.Ly*2-2, ortho_center))
        while self.ortho_center < ortho_center:
            self.move_ortho_center_up()
        while self.ortho_center > ortho_center:
            self.move_ortho_center_down()

    def move_to(self, ortho_surface, ortho_center):
        """
        Moves the orthoganilty hypersurface and center to the given indices 

        Parameters
        ----------
        ortho_surface : int
            the index the orthogonality hypersurface is to be moved to
        ortho_center : int
            the index the orthogonality center is to be moved to
        """
        self.move_ortho_surface_to(ortho_surface)
        self.move_ortho_center_to(ortho_center)

    def move_to_loc(self, location="upper right"):
        """
        Moves the orthogonality center to the specified location.

        Parameters
        ----------
        location : str, one of {"upper right", "upper left", "lower right", "lower left"}
            string specifying the position the orthogonality hypersurface and 
            orthogonality center should be moved to. 
        """
        if location == "upper right":
            self.move_to(self.Lx*2-2, self.Ly*2-2)
        elif location == "upper left":
            self.move_to(0, self.Ly*2-2)
        elif location == "lower right":
            self.move_to(self.Lx*2-2, 0)
        elif location == "lower left":
            self.move_to(0, 0)