import numpy as np
import h5py
import os
import time
from ..isoTPS.square.isoTPS import isoTPS_Square
from ..isoTPS.honeycomb.isoTPS import isoTPS_Honeycomb
from ..models import tfi
from ..utility import utility
from ..utility import debug_logging

def perform_gs_search_run(tps_params, model_params, dtau_l, dtau_r, max_iters_TEBD=50, max_iters_gss=10, min_dtau_diff=1e-9, lattice="square", initialize="spinup", output_filename=None, checkpoints=False):
    """
    Performs a ground state search using TEBD and a golden section search. We start by computing ground state approximations
    with imaginary time TEBD for time step values dtau_l_0 and dtau_r_0 with dtau_l_0 < dtau_r_0. In the next iterations, ground state 
    approximations are computed at intermediate time step values, and the search interval is shrunk according to the value of the
    energy. Intermediate dtau values are chosen according to the golden section search (J. Kiefer. Sequential minimax search for a maximum. 
    Proceedings of the American Mathematical Society, 4(3):502-506, 1953). For computing the ground state approximations, the previous best
    ground state approximation is evolved using imaginary time TEBD, terminating as soon as the energy does not decrease anymore.

    Parameters
    ----------
    tps_params : dict
        dictionary passed as keyword arguments into the constructor of isoTPS, see "src/isoTPS/square/isoTPS.py" or
        "src/isoTPS/honeycomb/isoTPS.py" for more detauls.
    model_params : dict
        dictionary specifying the model, in this case the g and J parameters for the TFI model.
    dtau_l : float
        left border of the search the interval [dtau_l, dtau_r]
    dtau_r : float
        right border of the search the interval [dtau_l, dtau_r]
    max_iters_TEBD : int, optional
        maximum number of TEBD iterations done for computing the approximate ground state at a given dtau. Default: 50.
    max_iters_gss : int, optional
        maximum number of golden-section search performed. Default: 10.
    min_dtau_diff : float, optional
        if the interval size becomes smaller than min_dtau_diff, the algorithm is terminated.
    lattice : str, one of {"square", "honyecomb"}, optional
        The lattice geometry. Default: "square".
    initialize: str, one of {"spinup", "spinright"}, optional
        initialization method. Default: "spinup".
    output_filename : str or None, optional
        the filename for the results of the simulations and the logging file. Do not include the suffix, the suffix ".h5" will be
        added to output file and the suffix ".log" will be added to log file automatically. If this is set to None,
        logging is printed to the console and the results of the run are returned instead. Default: None
    checkpoints: bool, optional
        If this is set to true, the state is saved after each TEBD step. Additionally, the simulation checks if a checkpoint was
        previously saved and starts from the checkpoint if it was found. If this is set to False, no checkpoints are saved/loaded.
        Default: False.

    Returns
    -------
    intervals : list of tuples (dtau_l, dtau_1, dtau_2, dtau_r)
        The hisotry of intervals the algorithm went through. Is only returned if output_filename == None.
    energies : list of tuples (energy_1, energy_2)
        The energies at dtau_1 and dtau_2 of the intervals. Is only returned if output_filename == None.
    """
    # Make sure parameters are in the correct format
    assert("g" in model_params)
    assert("J" in model_params)
    assert("Lx" in tps_params)
    assert("Ly" in tps_params)
    assert(max_iters_TEBD > 0)
    assert(lattice in {"square", "honeycomb"})
    assert(initialize in {"spinup", "spinright"})
    N = 2 * tps_params["Lx"] * tps_params["Ly"]
    assert(dtau_l < dtau_r)

    def append_to_log(text):
        """
        Appends the given text to the log
        """
        if output_filename is None:
            print(text)
        else:
            with open(output_filename + ".log", "a") as file:
                file.write(text + "\n")

    # Prepare output files
    if output_filename is not None:
        # Save parameters in h5 file
        with h5py.File(output_filename + ".h5", "w") as hf:
            parameters = {
                "tps_params" : tps_params,
                "model_params" : model_params,
                "dtau_l" : dtau_l,
                "dtau_r" : dtau_r,
                "max_iters_TEBD" : max_iters_TEBD,
                "max_iters_gss" : max_iters_gss,
                "min_dtau_diff" : min_dtau_diff,
                "lattice": lattice,
                "initialize": initialize, 
                "output_filename" : output_filename,
                "checkpoints": checkpoints
            }
            utility.dump_dict_into_hf(hf, parameters)
            hf["done"] = False
        # Create log file
        with open(output_filename + ".log", "w") as file:
            pass

    golden_ratio = (np.sqrt(5) + 1) / 2
    algorithm_data = {
        "n_start_gss": 2,
        "n_start_TEBD": 0,
        "dtau_l": dtau_l,
        "dtau_r": dtau_r,
        "delta": dtau_r - dtau_l,
        "dtau_1": dtau_l + (dtau_r - dtau_l) / golden_ratio**2,
        "dtau_2": dtau_l + (dtau_r - dtau_l) / golden_ratio,
        "energy_1": None,
        "energy_2": None
    }
    best_tps_data = {} # stores the current best ground state approximation together with its energy

    loaded = False
    if checkpoints: # Try to load checkpoint
        for n in range(max_iters_gss):
            if os.path.isfile(output_filename + f"_checkpoint_n_gss_{n}_n_TEBD_{0}.h5") or os.path.isfile(output_filename + f"_checkpoint_n_gss_{n}_n_TEBD_{max_iters_TEBD}.h5"):
                algorithm_data["n_start_gss"] = n
                loaded = True
            else:
                break
        if loaded:
            for n in range(1, max_iters_TEBD+1):
                temp_n_start_gss = algorithm_data["n_start_gss"]
                if os.path.isfile(output_filename + f"_checkpoint_n_gss_{temp_n_start_gss}_n_TEBD_{n}.h5"):
                    algorithm_data["n_start_TEBD"] = n
            # Load TPS
            temp_n_start_gss = algorithm_data["n_start_gss"]
            temp_n_start_TEBD = algorithm_data["n_start_TEBD"]
            best_tps_data["tps"] = isoTPS_Square.load_from_file(output_filename + f"_checkpoint_tps_n_gss_{temp_n_start_gss}_n_TEBD_{temp_n_start_TEBD}.h5")
            with h5py.File(output_filename + f"_checkpoint_n_gss_{temp_n_start_gss}_n_TEBD_{temp_n_start_TEBD}.h5", "r") as hf:
                algorithm_data = utility.load_dict_from_hf(hf)            
            best_tps_data["E"] = algorithm_data["best_energy"]
            append_to_log(f"Loaded checkpoint from file \"output_filename_checkpoint_n_gss_{temp_n_start_gss}_n_TEBD_{temp_n_start_TEBD}.h5\"")

    if not loaded:
        # Prepare tps
        if lattice == "square":
            best_tps_data["tps"] = isoTPS_Square(**tps_params)
        elif lattice == "honeycomb":
            best_tps_data["tps"] = isoTPS_Honeycomb(**tps_params)
        if initialize == "spinup":
            best_tps_data["tps"].initialize_spinup()
        elif initialize == "spinright":
            best_tps_data["tps"].initialize_spinright()
        best_tps_data["tps"].reset_time_counters()
        if checkpoints:
            append_to_log("Was not able to load checkpoint.")

    # Prepare Hamiltonian
    H_bonds = tfi.TFI(model_params["g"], model_params["J"]).compute_H_bonds_2D_Square(tps_params["Lx"], tps_params["Ly"])
    best_tps_data["E"] = np.sum(best_tps_data["tps"].copy().compute_expectation_values_twosite(H_bonds))

    def save_checkpoint(best_tps_data, n_gss, n_TEBD, algorithm_data):
        algorithm_data["best_energy"] = best_tps_data["E"]
        with h5py.File(output_filename + f"_checkpoint_n_gss_{n_gss}_n_TEBD_{n_TEBD}.h5", "w") as hf:
            utility.dump_dict_into_hf(hf, algorithm_data)
        best_tps_data["tps"].save_to_file(output_filename + f"_checkpoint_tps_n_gss_{n_gss}_n_TEBD_{n_TEBD}.h5")
        append_to_log(f"Saved checkpoint to file \"{output_filename}_checkpoint_n_gss_{n_gss}_n_TEBD_{n_TEBD}.h5\"")

    def compute_energy(dtau, n_gss, best_tps_data, algorithm_data):
        """
        Computes approximate ground state energy using TEBD
        """
        U_bonds = utility.calc_U_bonds(H_bonds, dtau/2)
        for n in range(algorithm_data["n_start_TEBD"], max_iters_TEBD):
            tps_prev = best_tps_data["tps"].copy()
            best_tps_data["tps"].reset_time_counters()
            append_to_log(f"Computing TEBD step {n + 1} with dtau = {dtau} ...")
            start_TEBD = time.time()
            best_tps_data["tps"].perform_TEBD2(U_bonds, 1)
            E_new = np.sum(best_tps_data["tps"].copy().compute_expectation_values_twosite(H_bonds))
            end_TEBD = time.time()
            append_to_log(f"Energy = {E_new}, took {round(end_TEBD-start_TEBD, 3)} seconds.")
            if best_tps_data["tps"].debug_logger.log_yb_move_walltimes:
                append_to_log(f"Total time YB: {np.sum(best_tps_data['tps'].debug_logger.log_dict['yb_move_walltimes'])}")
            if best_tps_data["tps"].debug_logger.log_local_tebd_update_walltimes:
                append_to_log(f"Total time TEBD: {np.sum(best_tps_data['tps'].debug_logger.log_dict['local_tebd_update_walltimes'])}")
            if E_new > best_tps_data["E"]:
                # energy increased -> terminate
                best_tps_data["tps"] = tps_prev
                algorithm_data["n_start_TEBD"] = 0
                return best_tps_data["E"]
            best_tps_data["E"] = E_new
            if checkpoints:
                save_checkpoint(best_tps_data, n_gss, n, algorithm_data)
        append_to_log("[WARNING]: Reached end of TEBD steps. Consider increasing parameter max_iters_TEBD.")
        algorithm_data["n_start_TEBD"] = 0
        return best_tps_data["E"]

    # Run algorithm
    start = time.time()
    if algorithm_data["delta"] <= min_dtau_diff:
        return (algorithm_data["dtau_l"], algorithm_data["dtau_r"]), compute_energy(algorithm_data["dtau_r"], 0, best_tps_data, algorithm_data)
    temp = (algorithm_data["dtau_l"], algorithm_data["dtau_1"], algorithm_data["dtau_2"], algorithm_data["dtau_r"])
    append_to_log(f"interval: {temp}")
    if algorithm_data["energy_2"] is None:
        algorithm_data["energy_2"] = compute_energy(algorithm_data["dtau_2"], 0, best_tps_data, algorithm_data)
    if algorithm_data["energy_1"] is None:
        algorithm_data["energy_1"] = compute_energy(algorithm_data["dtau_1"], 1, best_tps_data, algorithm_data)
    intervals = [(algorithm_data["dtau_l"], algorithm_data["dtau_1"], algorithm_data["dtau_2"], algorithm_data["dtau_r"])]
    energies = [(algorithm_data["energy_1"], algorithm_data["energy_2"])]
    for n_gss in range(algorithm_data["n_start_gss"], max_iters_gss + 2):
        if algorithm_data["energy_1"] < algorithm_data["energy_2"]:
            append_to_log("Shrinking to the left!")
            # Shrink dtau interval to the left
            algorithm_data["dtau_r"] = algorithm_data["dtau_2"]
            algorithm_data["dtau_2"] = algorithm_data["dtau_1"]
            algorithm_data["energy_2"] = algorithm_data["energy_1"]
            algorithm_data["delta"] /= golden_ratio
            algorithm_data["dtau_1"] = algorithm_data["dtau_l"] + algorithm_data["delta"] / golden_ratio**2
            algorithm_data["energy_1"] = compute_energy(algorithm_data["dtau_1"], n_gss, best_tps_data, algorithm_data)
        else:
            append_to_log("Shrinking to the right!")
            # Shrink dtau interval to the right
            algorithm_data["dtau_l"] = algorithm_data["dtau_1"]
            algorithm_data["dtau_1"] = algorithm_data["dtau_2"]
            algorithm_data["energy_1"] = algorithm_data["energy_2"]
            algorithm_data["delta"] /= golden_ratio
            algorithm_data["dtau_2"] = algorithm_data["dtau_l"] + algorithm_data["delta"] / golden_ratio
            algorithm_data["energy_2"] = compute_energy(algorithm_data["dtau_2"], n_gss, best_tps_data, algorithm_data)
        temp = (algorithm_data["dtau_l"], algorithm_data["dtau_1"], algorithm_data["dtau_2"], algorithm_data["dtau_r"])
        intervals.append(temp)
        append_to_log(f"interval: {intervals[-1]}")
        energies.append((algorithm_data["energy_1"], algorithm_data["energy_2"]))
        if checkpoints:
            save_checkpoint(best_tps_data, n_gss, max_iters_TEBD, algorithm_data)
        if algorithm_data["delta"] <= min_dtau_diff:
            break
    end = time.time()
    append_to_log(f"Finished, took {round(end-start, 3)} seconds!")
    if output_filename is None:
        return intervals, energies
    else:
        with h5py.File(output_filename + ".h5", "r+") as hf:
            hf["dtau_ls"] = [temp for (temp, _, _, _) in intervals]
            hf["dtau_1s"] = [temp for (_, temp, _, _) in intervals]
            hf["dtau_2s"] = [temp for (_, _, temp, _) in intervals]
            hf["dtau_rs"] = [temp for (_, _, _, temp) in intervals]
            hf["energies_1"] = [temp for (_, _, _, temp) in intervals]
            hf["energies_2"] = [temp for (_, _, _, temp) in intervals]
            if algorithm_data["energy_1"] < algorithm_data["energy_2"]:
                hf["final_energy"] = algorithm_data["energy_1"]
                hf["final_dtau"] = algorithm_data["dtau_1"]
            else:
                hf["final_energy"] = algorithm_data["energy_2"]
                hf["final_dtau"] = algorithm_data["dtau_2"]
            hf["wall_time"] = end - start
            hf["done"][...] = True
            if best_tps_data['tps'].debug_logger is not None:
                best_tps_data['tps'].debug_logger.save_to_file_h5(hf)        



