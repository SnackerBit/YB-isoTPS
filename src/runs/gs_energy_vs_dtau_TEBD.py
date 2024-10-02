import numpy as np
from ..isoTPS.square.isoTPS import isoTPS_Square
from ..isoTPS.honeycomb.isoTPS import isoTPS_Honeycomb
from ..utility import utility
from ..utility import debug_levels
from ..models import tfi
import time
import h5py
import hdfdict
import traceback

def perform_gs_energy_vs_dtau_run(tps_params, model_params, dtaus, dtau_index, N_steps, tebd_order=2, lattice="square", initialize="spinup", L=None, output_filename=None):
    """
    Computes one data point of a "TEBD ground state energy vs dtau" plot. Because the YB move injects a small error when moving around
    the orthogonality surface, the energy of the ground state we are able to reach is limited. There is a "competition" between TEBD error
    and YB error: TEBD error is smaller for small dtau, but the YB error rises because more sweeps are necessary for convergence. Thus there
    exists an optimal dtau, for which we can achieve the minimal ground state energy. The run works as follows: Given an ordered list of 
    timesteps dtau and an index dtau_index, TEBD is computed for all dtau < dtau[dtau_index], where we skip to the next dtau if the energy 
    increases, but evolve for a maximum of N_steps. Finally, TEBD is computed for N_steps with dtau[dtau_index], where we do not skip.
    The model used is the transverse field Ising model.

    Parameters
    ----------
    tps_params : dict
        dictionary passed as keyword arguments into the constructor of isoTPS, see "src/isoTPS/square/isoTPS.py" or
        "src/isoTPS/honeycomb/isoTPS.py" for more detauls.
    model_params : dict
        dictionary specifying the model, in this case the g and J parameters for the TFI model.
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
    assert("g" in model_params)
    assert("J" in model_params)

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
            hdfdict.dump(parameters, hf)
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
    tps.reset_debug_dict()

    # Initialize Hamiltonian
    if lattice == "square":
        H_bonds = tfi.TFI(model_params["g"], model_params["J"]).compute_H_bonds_2D_Square(tps_params["Lx"], tps_params["Ly"])
    elif lattice == "honeycomb":
        H_bonds = tfi.TFI(model_params["g"], model_params["J"]).compute_H_bonds_2D_Honeycomb(tps_params["Lx"], tps_params["Ly"])

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
            tps.reset_debug_errors_and_times()
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
            E = np.sum(tps.copy().compute_expectation_values_twosite(H_bonds))
            if i < dtau_index and len(Es) > 0 and Es[-1] <= E: 
                # Energy got higher. Go to next dtau!
                tps = tps_prev
                append_to_log(f"energy got larger. Moving on to next timestep.")
                break
            Es.append(E)
            dtaus_final.append(dtau)
            append_to_log(f"dtau = {dtau}, n = {n}, E = {E}")
            append_to_log(f"Total time: {tebd_time}")
            if debug_levels.check_debug_level(tps.debug_dict, debug_levels.DebugLevel.LOG_PER_SITE_ERROR_AND_WALLTIME):
                append_to_log(f"Total time YB: {np.sum(tps.debug_dict['times_yb'])}")
                append_to_log(f"Total time TEBD: {np.sum(tps.debug_dict['times_tebd'])}")
                append_to_log(f"Error density YB: {np.sum(tps.debug_dict['errors_yb']) / N}")
                append_to_log(f"Error density TEBD: {np.sum(tps.debug_dict['errors_tebd']) / N}")

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
            if tps.debug_dict is not None:
                tps.dump_debug_dict(hf)