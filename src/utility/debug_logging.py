import copy
from . import utility

class DebugLogger:

    def __init__(self, log_algorithm_walltimes=False, log_approximate_column_error_yb=False, log_approximate_column_error_tebd=False, 
        log_column_error_yb_before_variational_optimization=False, log_column_error_yb_after_variational_optimization=False, 
        log_yb_move_errors=False, log_yb_move_walltimes=False, log_local_tebd_update_errors=False, 
        log_local_tebd_update_walltimes=False, log_iterative_local_tebd_update_info=False, 
        log_iterative_local_tebd_update_errors_per_iteration=False, tripartite_decomposition_log_info=False, 
        tripartite_decomposition_log_info_per_iteration=False, tripartite_decomposition_log_iterates=False, 
        disentangling_log_info=False, disentangling_log_info_per_iteration=False, disentangling_log_iterates=False, 
        variational_column_optimization_log_info=False, variational_column_optimization_log_info_per_iteration=False, 
        log_yb_move_environments=False, log_local_tebd_update_environments=False):

        # The dictionary in which all the logs are saved
        self.log_dict = {}

        # If this is set to true, during each iteration of TEBD, the cumulative walltimes of local TEBD updates,
        # yb moves, and variational column optimization are logged in 
        # self.log_dict["algorithm_walltimes"]["local_tebd_update"], self.log_dict["algorithm_walltimes"]["yb_move"]
        # and self.log_dict["algorithm_walltimes"]["variational_column_optimization"]
        self.log_algorithm_walltimes = log_algorithm_walltimes

        # If this is set to true, the column error of moving the orthogonality hypersurface to the left or to the
        # right is approximated by summing the errors of the YB moves. The approximate column error is saved in
        # self.log_dict["approximate_column_errors_yb"]. the approximate column error approximates the true error
        # before any variational optimization.
        self.log_approximate_column_error_yb = log_approximate_column_error_yb
        # If this is set to true, the column error of applying local TEBD operators along a column is 
        # approximated by summing the errors of local updates. The approximate column error is saved in
        # self.log_dict["approximate_column_errors_tebd"].
        self.log_approximate_column_error_tebd = log_approximate_column_error_tebd
        # If this is set to true, the true column error after moving the orthogonality hypersurface but before
        # performing any variational column optimization is logged. The error is saved in
        # self.log_dict["column_errors_yb_before_variational_optimization"]
        self.log_column_error_yb_before_variational_optimization = log_column_error_yb_before_variational_optimization
        # If this is set to true, the true column error after moving the orthogonality hypersurface and applying
        # variational optimization is logged. The error is saved in
        # self.log_dict["column_errors_yb_after_variational_optimization"]
        self.log_column_error_yb_after_variational_optimization = log_column_error_yb_after_variational_optimization

        # If this is set to true, the error after each YB move is logged in a consecutive list:
        # self.log_dict["yb_move_errors"]
        self.log_yb_move_errors = log_yb_move_errors
        # If this is set to true, the walltime of each YB is logged in a consecutive list:
        # self.log_dict["yb_move_walltimes"]
        self.log_yb_move_walltimes = log_yb_move_walltimes
        # If this is set to true, the error after each local TEBD update is logged in a consecutive list:
        # self.log_dict["local_tebd_update_errors"]
        self.log_local_tebd_update_errors = log_local_tebd_update_errors
        # If this is set to true, the walltime of each local TEBD update is logged in a consecutive list:
        # self.log_dict["local_tebd_update_walltimes"]
        self.log_local_tebd_update_walltimes = log_local_tebd_update_walltimes

        # If this is set to true, information about the iterative local TEBD update algorithm is logged, e.g. the
        # number of iterations the algorithm was run for. This is logged once per call to the algorithm. The
        # information is saved in self.log_dict["iterative_local_tebd_update_info"][key] with e.g. key="N_iters".
        self.log_iterative_local_tebd_update_info = log_iterative_local_tebd_update_info
        # If this is set to true, the error after each iteration of the iterative TEBD update algorithm is logged in
        # self.log_dict["local_tebd_update_errors_per_iteration"]
        self.log_iterative_local_tebd_update_errors_per_iteration = log_iterative_local_tebd_update_errors_per_iteration

        # If this is set to true, information about the tripartite decomposition algorithm is logged, e.g. the number
        # of iterations the algorithm was run for. This is logged once per call to the algorithm. The information is
        # saved in self.log_dict["tripartite_decomopsition_info"][key] with e.g. key="N_iters".
        self.tripartite_decomposition_log_info = tripartite_decomposition_log_info
        # If this is set to true, per-iteration information is logged during any of the iterative tripartite 
        # decomposition algorithms, e.g. the value of the cost function. The information is saved in e.g. 
        # self.log_dict["tripartite_decomopsition_info"]["costs"].
        self.tripartite_decomposition_log_info_per_iteration = tripartite_decomposition_log_info_per_iteration
        # If this is set to true, the iterates A, B and C are logged after each iteration during any of the
        # iterative tripartite decomposition routines. The iterates are saved in 
        # self.log_dict["tripartite_decomopsition_info"]["iterates"].
        self.tripartite_decomposition_log_iterates = tripartite_decomposition_log_iterates
        
        # If this is set to true, information about the disentangling algorithm is logged, e.g. the number
        # of iterations the algorithm was run for. This is logged once per call to the algorithm. The information is
        # saved in self.log_dict["disentangling_info"][key] with e.g. key="N_iters".
        self.disentangling_log_info = disentangling_log_info
        # If this is set to true, per-iteration information is logged during disentangling, e.g. the value of the
        # cost function, or the number of tcG iterations for each step of the TRM, etc. The information is saved in
        # e.g. self.log_dict["disentangling_info"]["N_iters_tCG"].
        self.disentangling_log_info_per_iteration = disentangling_log_info_per_iteration
        # If this is set to true, the iterates U are logged after each iteration furing any of the iterative
        # disentangling routines. The iterates are saved in self.log_dict["disentangling_info"]["iterates"].
        self.disentangling_log_iterates = disentangling_log_iterates

        # If this is set to true, information about the variational column optimization algorithm is logged,
        # e.g. the number of iterations the algorithm was run for. This is logged once per call to the algorithm.
        # The information is saved in self.log_dict["variational_column_optimization_info"][key] with e.g.
        # key="N_iters"
        self.variational_column_optimization_log_info = variational_column_optimization_log_info
        # If this is set to true, per-iteration information is logged during variational column optimization.
        # e.g. the error each iteration. The information is saved in 
        # self.log_dict["variational_column_optimization_info"][key] with e.g. key="costs"
        self.variational_column_optimization_log_info_per_iteration = variational_column_optimization_log_info_per_iteration

        # If this is set to true, the environment tensors W1, W2, T before a yb move are saved in a list at
        # self.log_dict["yb_move_environments"]
        self.log_yb_move_environments = log_yb_move_environments
        # If this is set to true, the environment tensors T1, T2, Wm1, W, Wp1 before a local tebd update are saved in a list 
        # at self.log_dict["local_tebd_update_environments"]
        self.log_local_tebd_update_environments = log_local_tebd_update_environments

        # This suffix will be appended to all keys when calling self.append_to_log_list. This functionality is used
        # for example when using multiple disentanglers for the yb move.
        self.key_suffix = ""


    def append_to_log_list(self, list_name, element):
        """
        If a list with name list_name already exists in self.log_dict, element is appended to it.
        Else, the list will be created first. This function throws an error if the key list_name exists
        in self.log_dict but self.log_dict[list_name] is not a list. list_name can also be a tuple of strings
        describing a nested dictionary structure.

        Parameters
        ----------
        list_name: str or tuple of str
            name of the list, or a tuple of str that describe nested dictionaries leading to a list, e.g. ("dict1", "dict_2", "my_list")
            for the dictionary {"dict1": {"dict2": {"my_list": [...]}, ...}, ...}.
        element: object
            element that should be added to the list
        """
        if type(list_name) is str:
            list_name = (list_name,)
        if type(list_name) is not tuple:
            raise ValueError(f"list_name must be of type str or tuple of str, but is of type {type(list_name)}")
        if len(list_name) == 0:
            raise ValueError(f"list_name must not be empty.")
        current_dict = self.log_dict
        for i, name in enumerate(list_name):
            if type(name) is not str:
                raise ValueError(f"dictionary key must be of type str, but was of type {type(name)}")
            if i < len(list_name)-1:
                # Go deeper into nested dictionary structure
                if name not in current_dict:
                    current_dict[name] = {}
                current_dict = current_dict[name]
            else:
                # Check if the final element is a list and append the element
                name += self.key_suffix
                if name not in current_dict:
                    current_dict[name] = []
                if type(current_dict[name]) is not list:
                    raise ValueError(f"{list_name} already exists as a key in log_dict, but the corresponding value is not a list, but an element of type {type(current_dict[name])}")
                current_dict[name].append(element)

    def save_to_file_h5(self, h5file):
        """
        Saves the logging info into an already opened h5 file.
        """
        temp = {}
        temp["debug_log"] = copy.deepcopy(self.log_dict)
        utility.turn_lists_to_dicts(temp)
        utility.dump_dict_into_hf(h5file, temp)

    def load_log_dict(self, log_dict):
        """
        Loads the log_dict from a dictionary that was generated by loading from a h5file previously saved
        by a call to DebugLogger.save_to_file_h5().
        """
        temp = copy.deepcopy(log_dict)
        utility.turn_dicts_to_lists(temp)
        self.log_dict = temp

    def clear(self):
        """
        Clears all logs
        """
        self.log_dict = {}