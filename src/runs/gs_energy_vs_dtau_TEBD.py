from ..isoTPS.square.isoTPS import isoTPS_Square
from ..isoTPS.honeycomb.isoTPS import isoTPS_Honeycomb
from ..utility import utility
from ..models import tfi
from ..utility import backend
import time
import h5py
import traceback
import copy

def perform_gs_energy_vs_dtau_run(tps_params, model_params, dtaus, dtau_index, N_steps, tebd_order=2, lattice="square", initialize="spinup", L=None, model=tfi.TFI, output_filename=None):
    """
    Computes one data point of a "TEBD ground state energy vs dtau" plot. Because the YB move injects a small error when moving around
    the orthogonality surface, the energy of the ground state we are able to reach is limited. There is a competition between TEBD error
    and YB error: TEBD error is smaller for small dtau, but the YB error rises because more sweeps are necessary for convergence. Thus there
    exists an optimal dtau, for which we can achieve the minimal ground state energy at a given maximum bond dimension. The run works as 
    follows: Given an ordered list of timesteps dtau and an index dtau_index, TEBD is computed for all dtau < dtau[dtau_index], where we 
    skip to the next dtau if the energy increases, but evolve for a maximum of N_steps. Finally, TEBD is computed for N_steps with 
    dtau[dtau_index], where we do not skip.

    Parameters
    ----------
    tps_params : dict
        dictionary passed as keyword arguments into the constructor of isoTPS, see "src/isoTPS/square/isoTPS.py" or
        "src/isoTPS/honeycomb/isoTPS.py" for more detauls.
    model_params : dict
        dictionary specifying the model parameters, e.g. g and J parameters for the TFI model.
    dtaus : list of float
        TEBD time steps.
    dtau_index : int
        specifies up to which time step the algorithm is run
    N_steps : int
        maximal number of TEBD steps per time step
    tebd_order : int, one of {1, 2}, optional
        switch between 1st and 2nd order TEBD. Default: 2.
    lattice : str, one of {"square", "honyecomb"}, optional
        The lattice geometry. Default: "square".
    initialize: str, one of {"spinup", "spinright"}, optional
        initialization method. Default: "spinup".
    L : int or None, optional
        if this is not None, both Lx and Ly of the lattice are set to L.
        Else Lx and Ly are expected in tps_params. Default: None.
    model : Model class, optional
        The model used for generating the Hamiltonian. Default: tfi.TFI (the transverse field Ising model).
    output_filename : str or None, optional
        the filename for the results of the simulations and the logging file. Do not include the suffix, the suffix ".h5" will be
        added to output file and the suffix ".log" will be added to log file automatically. If this is set to None,
        logging is printed to the console and the results of the run are returned instead. Default: None

    Returns
    -------
    Es : list of float
        list of energies computed during the run. Is only returned if output_filename == None.
    dtaus_final : list of float
        list of time steps dtau used at each iteration. Is only returned if output_filename == None.
    walltime : float
        total time the algorithm was run for. Is only returned if output_filename == None.
    """
    # Make sure parameters are in the correct format
    if L is not None:
        tps_params["Lx"] = L
        tps_params["Ly"] = L
    assert("Lx" in tps_params)
    assert("Ly" in tps_params)
    assert(tebd_order == 1 or tebd_order == 2)
    assert(N_steps > 0)
    assert(lattice in {"square", "honeycomb"})
    assert(initialize in {"spinup", "spinright"})
    N = 2 * tps_params["Lx"] * tps_params["Ly"]

    def append_to_log(text):
        """
        Appends the given text to the log
        """
        if output_filename is None:
            print(text)
        else:
            with open(output_filename + ".log", "a") as file:
                file.write(text + "\n")

    if output_filename is not None:
        # Save parameters in h5 file
        with h5py.File(output_filename + ".h5", "w") as hf:
            parameters = {
                "tps_params" : tps_params,
                "model_params" : model_params,
                "dtaus" : dtaus,
                "dtau_index" : dtau_index,
                "N_steps" : N_steps,
                "tebd_order" : tebd_order,
                "lattice" : lattice,
                "output_filename" : output_filename,
                "initialize" : initialize
            }
            utility.dump_dict_into_hf(hf, parameters)
            hf["done"] = False
            hf["success"] = False
        # Create log file
        with open(output_filename + ".log", "w") as file:
            pass

    start = time.time()

    # Initialize TPS
    if lattice == "square":
        tps = isoTPS_Square(**tps_params)
    elif lattice == "honeycomb":
        tps = isoTPS_Honeycomb(**tps_params)
    if initialize == "spinup":
        tps.initialize_spinup()
    elif initialize == "spinright":
        tps.initialize_spinright()
    tps.debug_logger.clear()

    # Initialize Hamiltonian
    if lattice == "square":
        H_bonds = model(**model_params).compute_H_bonds_2D_Square(tps_params["Lx"], tps_params["Ly"])
    elif lattice == "honeycomb":
        H_bonds = model(**model_params).compute_H_bonds_2D_Honeycomb(tps_params["Lx"], tps_params["Ly"])

    # Perform time evolution
    Es = []
    dtaus_final = []
    error = None
    tebd_time = 0.0

    for i in range(dtau_index + 1):
        dtau = dtaus[i]
        if tebd_order == 1:
            U_bonds = utility.calc_U_bonds(H_bonds, dtau)
        elif tebd_order == 2:
            U_bonds = utility.calc_U_bonds(H_bonds, dtau/2)
        for n in range(N_steps):
            if i < dtau_index:
                tps_prev = tps.copy()
            try:
                start_TEBD = time.time()
                if tebd_order == 1:
                    tps.perform_TEBD1(U_bonds, 1)
                elif tebd_order == 2:
                    tps.perform_TEBD2(U_bonds, 1)
                end_TEBD = time.time()
                tebd_time = end_TEBD - start_TEBD
            except Exception as e:
                error = str(e)
                append_to_log(f"An error occurred: \"{error}\"")
                append_to_log(traceback.format_exc())
            if error is not None:
                break
            E = backend.sum(tps.copy().compute_expectation_values_twosite(H_bonds))
            if i < dtau_index and len(Es) > 0 and Es[-1] <= E: 
                # Energy got higher. Go to next dtau!
                tps = tps_prev
                append_to_log(f"energy got larger. Moving on to next timestep.")
                break
            Es.append(E)
            dtaus_final.append(dtau)
            append_to_log(f"dtau = {dtau}, n = {n}, E = {E}")
            append_to_log(f"Total time: {tebd_time}")
            if tps.debug_logger.log_algorithm_walltimes and  "algorithm_walltimes" in tps.debug_logger.log_dict:
                if "local_tebd_update" in tps.debug_logger.log_dict["algorithm_walltimes"]:
                    append_to_log(f"Total time TEBD: {tps.debug_logger.log_dict["algorithm_walltimes"]["local_tebd_update"][-1]}")
                if "yb_move" in tps.debug_logger.log_dict["algorithm_walltimes"]:
                    append_to_log(f"Total time YB: {tps.debug_logger.log_dict["algorithm_walltimes"]["yb_move"][-1]}")
                if "variational_column_optimization" in tps.debug_logger.log_dict["algorithm_walltimes"]:
                    append_to_log(f"Total time variational column optimization: {tps.debug_logger.log_dict["algorithm_walltimes"]["variational_column_optimization"][-1]}")
    end = time.time()
    if len(Es) > 0:
        append_to_log(f"finished simulation after {round(end - start, 4)} seconds with final energy {Es[-1]}.")
    else:
        append_to_log(f"finished simulation after {round(end - start, 4)} seconds. No energy was computed.")
    if output_filename is None:
        return Es, dtaus_final, end-start
    else:
        with h5py.File(output_filename + ".h5", "r+") as hf:
            hf["energies"] = Es
            hf["dtaus_final"] = dtaus_final
            hf["wall_time"] = end - start
            hf["done"][...] = True
            if error is None:
                hf["success"][...] = True
            else:
                hf["error"] = error
            tps.debug_logger.save_to_file_h5(hf)

def perform_gs_energy_vs_dtau_run_sequential(tps_params, model_params, dtaus, N_steps, tebd_order=2, lattice="square", initialize="spinup", L=None, model=tfi.TFI, output_filename=None):
    """
    This is a version of perform_gs_energy_vs_dtau_run(...) that is faster when computing energies for all dtaus sequentially.

    Parameters
    ----------
    tps_params : dict
        dictionary passed as keyword arguments into the constructor of isoTPS, see "src/isoTPS/square/isoTPS.py" or
        "src/isoTPS/honeycomb/isoTPS.py" for more detauls.
    model_params : dict
        dictionary specifying the model parameters, e.g. g and J parameters for the TFI model.
    dtaus : list of float
        TEBD time steps.
    N_steps : int
        maximal number of TEBD steps per time step
    tebd_order : int, one of {1, 2}, optional
        switch between 1st and 2nd order TEBD. Default: 2.
    lattice : str, one of {"square", "honyecomb"}, optional
        The lattice geometry. Default: "square".
    initialize: str, one of {"spinup", "spinright"}, optional
        initialization method. Default: "spinup".
    L : int or None, optional
        if this is not None, both Lx and Ly of the lattice are set to L.
        Else Lx and Ly are expected in tps_params. Default: None.
    model : Model class, optional
        The model used for generating the Hamiltonian. Default: tfi.TFI (the transverse field Ising model).
    output_filename : str or None, optional
        the filename for the results of the simulations and the logging file. Do not include the suffix, the suffix ".h5" will be
        added to output file and the suffix ".log" will be added to log file automatically. If this is set to None,
        logging is printed to the console and the results of the run are returned instead. Default: None

    Returns
    -------
    Es : list of float
        list of energies computed during the run. Is only returned if output_filename == None.
    dtaus_final : list of float
        list of time steps dtau used at each iteration. Is only returned if output_filename == None.
    walltime : float
        total time the algorithm was run for. Is only returned if output_filename == None.
    """
    # Make sure parameters are in the correct format
    if L is not None:
        tps_params["Lx"] = L
        tps_params["Ly"] = L
    assert("Lx" in tps_params)
    assert("Ly" in tps_params)
    assert(tebd_order == 1 or tebd_order == 2)
    assert(N_steps > 0)
    assert(lattice in {"square", "honeycomb"})
    assert(initialize in {"spinup", "spinright"})
    N = 2 * tps_params["Lx"] * tps_params["Ly"]

    def append_to_log(filename, text):
        """
        Appends the given text to the log
        """
        if filename is None:
            print(text)
        else:
            with open(filename + ".log", "a") as file:
                file.write(text + "\n")

    # Initialize TPS
    if lattice == "square":
        best_tps = isoTPS_Square(**tps_params)
    elif lattice == "honeycomb":
        best_tps = isoTPS_Honeycomb(**tps_params)
    if initialize == "spinup":
        best_tps.initialize_spinup()
    elif initialize == "spinright":
        best_tps.initialize_spinright()
    best_tps.debug_logger.clear()

    # Initialize Hamiltonian
    if lattice == "square":
        H_bonds = model(**model_params).compute_H_bonds_2D_Square(tps_params["Lx"], tps_params["Ly"])
    elif lattice == "honeycomb":
        H_bonds = model(**model_params).compute_H_bonds_2D_Honeycomb(tps_params["Lx"], tps_params["Ly"])

    Es = []
    dtaus_final = []

    for dtau_index, dtau in enumerate(dtaus):
        print(f"Computing data point for dtau = {dtau} ...")
        time_dtau_start = time.time()

        current_filename = output_filename
        if current_filename is not None:
            current_filename += f"_dtau_{dtau}"

        if current_filename is not None:
            # Save parameters in h5 file
            with h5py.File(current_filename + ".h5", "w") as hf:
                parameters = {
                    "tps_params" : tps_params,
                    "model_params" : model_params,
                    "dtaus" : dtaus,
                    "dtau_index" : dtau_index,
                    "N_steps" : N_steps,
                    "tebd_order" : tebd_order,
                    "lattice" : lattice,
                    "output_filename" : current_filename,
                    "initialize" : initialize
                }
                utility.dump_dict_into_hf(hf, parameters)
                hf["done"] = False
                hf["success"] = False
            # Create log file
            with open(current_filename + ".log", "w") as file:
                pass

        start = time.time()
        # Perform time evolution
        current_tps = best_tps.copy()
        current_Es = copy.deepcopy(Es)
        current_dtaus_final = copy.deepcopy(dtaus_final)
        error = None
        tebd_time = 0.0

        if tebd_order == 1:
                U_bonds = utility.calc_U_bonds(H_bonds, dtau)
        elif tebd_order == 2:
            U_bonds = utility.calc_U_bonds(H_bonds, dtau/2)
                
        for n in range(N_steps):
            try:
                start_TEBD = time.time()
                if tebd_order == 1:
                    current_tps.perform_TEBD1(U_bonds, 1)
                elif tebd_order == 2:
                    current_tps.perform_TEBD2(U_bonds, 1)
                end_TEBD = time.time()
                tebd_time = end_TEBD - start_TEBD
            except Exception as e:
                error = str(e)
                append_to_log(current_filename, f"An error occurred: \"{error}\"")
                append_to_log(current_filename, traceback.format_exc())
            if error is not None:
                break
            E = np.sum(current_tps.copy().compute_expectation_values_twosite(H_bonds))
            if len(current_Es) > 0 and current_Es[-1] > E:
                # energy got smaller. Save new best tps
                best_tps = current_tps.copy()
                Es.append(E)
                dtaus_final.append(dtau)
            current_Es.append(E)
            current_dtaus_final.append(dtau)
            append_to_log(current_filename, f"dtau = {dtau}, n = {n}, E = {E}")
            append_to_log(current_filename, f"Total time: {tebd_time}")
            if current_tps.debug_logger.log_algorithm_walltimes and  "algorithm_walltimes" in current_tps.debug_logger.log_dict:
                if "local_tebd_update" in current_tps.debug_logger.log_dict["algorithm_walltimes"]:
                    append_to_log(current_filename, f"Total time TEBD: {current_tps.debug_logger.log_dict["algorithm_walltimes"]["local_tebd_update"][-1]}")
                if "yb_move" in current_tps.debug_logger.log_dict["algorithm_walltimes"]:
                    append_to_log(current_filename, f"Total time YB: {current_tps.debug_logger.log_dict["algorithm_walltimes"]["yb_move"][-1]}")
                if "variational_column_optimization" in current_tps.debug_logger.log_dict["algorithm_walltimes"]:
                    append_to_log(current_filename, f"Total time variational column optimization: {current_tps.debug_logger.log_dict["algorithm_walltimes"]["variational_column_optimization"][-1]}")
        
        end = time.time()
        if len(current_Es) > 0:
            append_to_log(current_filename, f"finished simulation after {round(end - start, 4)} seconds with final energy {Es[-1]}.")
        else:
            append_to_log(current_filename, f"finished simulation after {round(end - start, 4)} seconds. No energy was computed.")
        if current_filename is not None:
            with h5py.File(current_filename + ".h5", "r+") as hf:
                hf["energies"] = current_Es
                hf["dtaus_final"] = current_dtaus_final
                hf["wall_time"] = end - start
                hf["done"][...] = True
                if error is None:
                    hf["success"][...] = True
                else:
                    hf["error"] = error
                current_tps.debug_logger.save_to_file_h5(hf)

        time_dtau_end = time.time()
        print(f"Took {round(time_dtau_end-time_dtau_start, 3)} seconds.")

    if output_filename is None:
                return current_Es, current_dtaus_final, end-start