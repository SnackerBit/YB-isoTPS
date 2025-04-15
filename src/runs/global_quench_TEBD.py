
import h5py
import hdfdict
import time
import numpy as np
import os
from ..isoTPS.square.isoTPS import isoTPS_Square
from ..isoTPS.honeycomb.isoTPS import isoTPS_Honeycomb
from ..models import tfi
from ..utility import utility

def perform_global_quench_run(tps_params, model_params, dt, N_steps, output_folder, tebd_order=2, lattice="square", initialize="spinup", initial_state=None, load_checkpoint=False):
    # Make sure parameters are in the correct format
    assert("g" in model_params)
    assert("J" in model_params)
    assert("Lx" in tps_params)
    assert("Ly" in tps_params)
    assert(tebd_order == 1 or tebd_order == 2)
    assert(N_steps > 0)
    assert(lattice in {"square", "honeycomb"})
    assert(initialize in {"spinup", "spinright", "product"})
    N = 2 * tps_params["Lx"] * tps_params["Ly"]

    def append_to_log(text):
        """
        Appends the given text to the log
        """
        with open(output_folder + "/log.log", "a") as file:
            file.write(text + "\n")

    # Save parameters in h5 file
    with h5py.File(output_folder + "/simulation.h5", "w") as hf:
        parameters = {
            "tps_params" : tps_params,
            "model_params" : model_params,
            "dt" : dt,
            "N_steps" : N_steps,
            "tebd_order" : tebd_order,
            "lattice" : lattice,
            "output_folder" : output_folder,
            "initialize" : initialize,
            "load_checkpoint" : load_checkpoint,
        }
        hdfdict.dump(parameters, hf)
        hf["done"] = False
        hf["success"] = False
    # Create log file
    with open(output_folder + "/log.log", "w") as file:
        pass

    start = time.time()

    n_start = None
    if load_checkpoint:
        # Try to load latest checkpoint
        for n in range(N_steps + 1):
            if os.path.isfile(output_folder + f"/tps_{n}.h5"):
                n_start = n
            else:
                break
    
    if n_start is None:
        if load_checkpoint:
            append_to_log(f"Wasn't able to load checkpoint. Starting from n = {0}.")
        n_start = 0
        # Initialize TPS
        if lattice == "square":
            tps = isoTPS_Square(**tps_params)
        elif lattice == "honeycomb":
            tps = isoTPS_Honeycomb(**tps_params)
        if initialize == "spinup":
            tps.initialize_spinup()
        elif initialize == "spinright":
            tps.initialize_spinright()
        elif initialize == "product":
            tps.initialize_product_state(np.array(initial_state))
        tps.save_to_file(output_folder + "/tps_0.h5", )
    else:
        append_to_log(f"Continuing from checkpoint n = {n_start}.")
        tps = isoTPS_Square.load_from_file(output_folder + f"/tps_{n_start}.h5")

    # Initialize Hamiltonian
    H_bonds = tfi.TFI(model_params["g"], model_params["J"]).compute_H_bonds_2D_Square(tps_params["Lx"], tps_params["Ly"])
    if tebd_order == 1:
        U_bonds = utility.calc_U_bonds(H_bonds, 1.j*dt)
    elif tebd_order == 2:
        U_bonds = utility.calc_U_bonds(H_bonds, 1.j*dt/2)
    # Perform time evolution
    #Es = [np.sum(tps.copy().compute_expectation_values_twosite(H_bonds))]
    for n in range(n_start, N_steps):
        append_to_log(f"Performing time step {n} ...")
        start_TEBD = time.time()
        if tebd_order == 1:
            tps.perform_TEBD1(U_bonds, 1)
        elif tebd_order == 2:
            tps.perform_TEBD2(U_bonds, 1)
        end_TEBD = time.time()
        tebd_time = end_TEBD - start_TEBD
        #E = np.sum(tps.copy().compute_expectation_values_twosite(H_bonds))
        #Es.append(E)
        append_to_log(f"Took {round(tebd_time, 3)} seconds.")#, energy = {E}.")
        if tps.debug_logger.log_algorithm_walltimes and "algorithm_walltimes" in tps.debug_logger.log_dict:
            if "local_tebd_update" in tps.debug_logger.log_dict["algorithm_walltimes"]:
                temp = tps.debug_logger.log_dict["algorithm_walltimes"]["local_tebd_update"][-1]
                append_to_log(f"Total time TEBD: {temp}")
            if "yb_move" in tps.debug_logger.log_dict["algorithm_walltimes"]:
                temp = tps.debug_logger.log_dict["algorithm_walltimes"]["yb_move"][-1]
                append_to_log(f"Total time YB: {temp}")
            if "variational_column_optimization" in tps.debug_logger.log_dict["algorithm_walltimes"]:
                temp = tps.debug_logger.log_dict["algorithm_walltimes"]["variational_column_optimization"][-1]
                append_to_log(f"Total time variational column optimization: {temp}")
        # save tps to file
        tps.save_to_file(output_folder + f"/tps_{n+1}.h5")
    end = time.time()
    append_to_log(f"finished simulation after {round(end - start, 4)} seconds.")

    # Save output
    with h5py.File(output_folder + "/simulation.h5", "r+") as hf:
        #hf["energies"] = Es
        hf["wall_time"] = end - start
        hf["done"][...] = True

def compute_onsesite_expectation_values(N_steps, output_folder):

    def append_to_log(text):
        """
        Appends the given text to the log
        """
        with open(output_folder + "/expectation_values.log", "a") as file:
            file.write(text + "\n")

    expectation_values = []
    sigma_x = tfi.TFI(0.0, 0.0).sigma_x
    sigma_y = tfi.TFI(0.0, 0.0).sigma_y
    sigma_z = tfi.TFI(0.0, 0.0).sigma_z
    for n in range(N_steps+1):
        append_to_log(f"Computing measurement for tps {n} ...")
        start = time.time()
        tps = isoTPS_Square.load_from_file(f"{output_folder}/tps_{n}.h5")
        expectation_values.append(tps.compute_expectation_values_onesite([sigma_x, sigma_y, sigma_z]))
        end = time.time()
        append_to_log(f"Took {round(end-start, 4)} seconds.")
    with h5py.File(output_folder + "/expectation_values.h5", "w") as hf:
        hf["expectation_values"] = np.array(expectation_values)