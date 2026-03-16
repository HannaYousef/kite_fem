import numpy as np
from pathlib import Path
from kite_fem.saveload import load_fem_structure
from kitesim.utils import load_yaml
from kitesim import read_struc_geometry_yaml_level_2
from kite_fem.Functions import adapt_stiffnesses
import csv

PROJECT_DIR = Path(__file__).resolve().parents[2]
kite_name = "TUDELFT_V3_KITE"  
struc_geometry_path = (
    Path(PROJECT_DIR)
    / "data"
    / f"{kite_name}"
    / "struc_geometry_hanging_test.yaml"
)
struc_geometry = load_yaml(struc_geometry_path)

(
    # node level
    struc_nodes,
    m_arr,
    struc_node_le_indices,
    struc_node_te_indices,
    power_tape_index,
    steering_tape_indices,
    pulley_node_indices,
    canopy_sections,
    strut_sections,
    simplified_bridle_points,
    # element level
    kite_connectivity_arr,
    bridle_connectivity_arr,
    bridle_diameter_arr,
    l0_arr,
    k_arr,
    c_arr,
    linktype_arr,
    pulley_line_indices,
    pulley_line_to_other_node_pair_dict,
) = read_struc_geometry_yaml_level_2.main(struc_geometry)

def extract_lengths_validation(kite,strut_sections):
    phi = []
    strut_sections = np.array(strut_sections)
    #extract lengths billowing segments
    coords = kite.coords_current.reshape(-1,3)
    te_ids = strut_sections[:,-1]
    le_ids = strut_sections[:,0]
    te_ids = te_ids[::-1]
    te_id1 = te_ids[0:-1]
    te_id2 = te_ids[1:]
    for id1,id2 in zip(te_id1,te_id2):
        coord1 = coords[id1]
        coord2 = coords[id2]
        length = np.linalg.norm(coord1-coord2)
        phi.append(length)
    #span
    spanid1 = te_ids[0]
    spanid2 = te_ids[-1]
    coord1 = coords[spanid1]
    coord2 = coords[spanid2]
    span = np.linalg.norm(coord1-coord2)
    phi.append(span)
    #tip_leading_edge distance
    middle_idx = len(le_ids) // 2
    # For uneven length, middle_idx is the exact center
    # Get the points on either side of the center
    left_le_id = le_ids[middle_idx]
    right_le_id = le_ids[middle_idx-1]
    rightids = [right_le_id, te_ids[-1]]
    leftids = [left_le_id, te_ids[0]]
    right = np.linalg.norm(coords[rightids[0]]-coords[rightids[1]])
    left = np.linalg.norm(coords[leftids[0]]-coords[leftids[1]])
    phi.append(right)
    phi.append(left)
    return phi

result_dir = Path(__file__).resolve().parent / "results"
result_dir.mkdir(exist_ok=True)
for load_case in range(1,11):
    result_path = result_dir / f"load_case_{load_case}.npz"
    kite = load_fem_structure(result_path)
    output = extract_lengths_validation(kite,strut_sections)
    tolerance = kite.crisfield_history[-1]
    output.append(tolerance)
    output.insert(0, load_case)
    max_strain = adapt_stiffnesses(kite, False)  
    output.append(max_strain)
    csv_path = Path(__file__).parent / "model_results.csv"
    with open(csv_path, 'w' if load_case == 1 else 'a', newline='') as f:
        writer = csv.writer(f)
        if load_case == 1:
            writer.writerow(["Load case",'La','Lb','Lc','Ld','Le','Lf','Lg','Lh','Li','b','LcsTL','LcsTR',"Tolerance","max strain"])
        writer.writerow(output)




# Import each row of model_results.csv as arrays
model_results_csv = Path(__file__).parent / "model_results.csv"
model_results_data = np.loadtxt(model_results_csv, delimiter=',', skiprows=1)[:,1:]
tolerance = model_results_data[:, -2]
model_results_data = model_results_data[:, :-2]


# Import each row of validation_data.csv as arrays
validation_csv = Path(__file__).parent / "validation_data.csv"
validation_data = np.loadtxt(validation_csv, delimiter=',', skiprows=1)[:,1:]
data_header = np.loadtxt(validation_csv, delimiter=',', max_rows=1, dtype=str)


def shapecorrelation(phi_exp,phi_mod):
    phi_exp = np.asarray(phi_exp).reshape(-1, 1)
    phi_mod = np.asarray(phi_mod).reshape(-1, 1)
    numerator = (phi_exp.T @ phi_mod)[0, 0] ** 2
    denominator = (phi_exp.T @ phi_exp)[0, 0] * (phi_mod.T @ phi_mod)[0, 0]
    return numerator / denominator

for test_case in range(1,11):
    results = model_results_data[test_case-1]
    validation = validation_data[test_case-1]
    SC = shapecorrelation(results,validation)
    meanabsolutedeviation = 1/len(results) * np.sum(np.abs(results-validation))
    print(f"Test case {test_case}: SC = {SC:.3f},Mean absolute deviation = {meanabsolutedeviation:.3f}")

