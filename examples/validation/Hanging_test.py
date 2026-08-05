from pathlib import Path
from multiprocessing import Pool, cpu_count
import copy

from kite_fem.FEMStructure import FEM_structure
from kite_fem.Plotting import (
    plot_structure,
    plot_structure_with_strain,
    plot_convergence
)

from kite_fem.Functions import relaxbridles, fix_nodes,set_pressure, adapt_stiffnesses, check_element_strain
from kite_fem.saveload import load_fem_structure,save_fem_structure

import matplotlib.pyplot as plt
import numpy as np
import time
import csv

PROJECT_DIR = Path(__file__).resolve().parents[2]
kite_path = (
    Path(PROJECT_DIR)
    / "data"
    / "TUDELFT_V3_KITE"
)
m_arr = np.loadtxt(kite_path / "mass_hanging_test.csv", delimiter=',')

def create_kite():
    """Factory function to create a fresh kite instance"""
    kite = load_fem_structure(kite_path / "hanging_test_initial.npz")
    return kite

def loading(N,m_arr,tip_load,point_load):
    fe = np.zeros(N)
    gravity = m_arr*9.81
    fe[2::6] = gravity 
    fe[2*6+1] += tip_load*9.81
    fe[58*6+1] += -tip_load*9.81
    fe[29*6+2] += point_load*9.81
    return fe

def solve_single_case(args):
    """Worker function to solve a single load case"""
    pressure, tip_load, point_load, load_case = args
    
    # Create fresh kite instance for this process
    kite = create_kite()
    kite = set_pressure(kite, pressure)
    fe = loading(kite.N, m_arr, tip_load, point_load)
    max_strain = 100
    iteration = 1
    print("set up kite for case",load_case)
    while max_strain >1 and iteration <5:
        start_time = time.time()
        print("load case",load_case, "iteration",iteration)
        kite.solve(fe=fe, max_iterations=15000, tolerance=0.01, step_limit=.005, 
                relax_init=.25, relax_min=0.00, relax_update=0.9998, k_update=1, I_stiffness=15)
        end_time = time.time()
        strain_data = check_element_strain(kite, False)
        all_strains = strain_data['spring_strains'] + strain_data['beam_strains']
        max_strain = max(all_strains)
        adapt_stiffnesses(kite)
        iteration += 1
    elapsed_time = end_time - start_time
    # Save results for this case
    result_dir = Path(__file__).resolve().parent / "results"
    result_dir.mkdir(exist_ok=True)
    save_path = result_dir / f"load_case_{load_case}.npz"
    save_fem_structure(kite,save_path)
    csv_path = result_dir / "timing.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([load_case, elapsed_time])

def get_load_cases():
    pressures = [0.15, 0.25]
    tip_loads = [2, 5]           #kg
    point_loads = [9.7, 25.2]     #kg
    # Build list of all load cases
    load_cases = []
    load_case = 0
    for pressure in pressures:
        # Base case (no additional loads)
        load_case += 1
        load_cases.append((pressure, 0, 0, load_case))
        # Point load cases
        for point_load in point_loads:
            load_case += 1
            load_cases.append((pressure, 0, point_load, load_case))
        # Tip load cases
        for tip_load in tip_loads:
            load_case += 1
            load_cases.append((pressure, tip_load, 0, load_case))
    return load_cases

if __name__ == '__main__':
    all_load_cases = get_load_cases()
    
    load_cases_to_run = [1,2,3,4,5,6,7,8,9,10] #adapt to only run certain cases

    # load_cases_to_run = [1] #adapt to only run certain cases

    load_cases = [all_load_cases[i-1] for i in load_cases_to_run]
    # Run simulations in parallel
    n_cores = cpu_count()
    print(f"Running {len(load_cases)} load cases on {n_cores} CPU cores...")
    
    with Pool(processes=n_cores) as pool:
        results = pool.map(solve_single_case, load_cases)
    
    print("All simulations complete!")






