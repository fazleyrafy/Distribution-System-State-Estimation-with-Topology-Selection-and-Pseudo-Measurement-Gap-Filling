import os
import sys
import pprint
import json
import logging
from datetime import datetime
from typing import List, Optional, Union
import numpy as np
import scipy.sparse
import pytest
from data_types import (
    Topology,
    PowersReal,
    PowersImaginary,
    VoltagesMagnitude,
    VoltagesReal,
    VoltagesImaginary,
    Complex,
    VoltagesAngle,
    AdmittanceMatrix,
    AdmittanceSparse
)
import csv
import json
import os
from datetime import datetime



def parse_dates(date_str):
    try:
        # Handle the format without seconds and AM/PM
        return datetime.strptime(date_str, '%m/%d/%Y %H:%M')
    except ValueError:
        print(f"Error parsing date: {date_str}")
        return None

# CSV file paths
p_real_csv = "testing_data_small_system/measured_active_power.csv"
p_img_csv = "testing_data_small_system/measured_reactive_power.csv"
v_mag_csv = "testing_data_small_system/measured_voltage_magnitudes.csv"
v_ang_csv = "testing_data_small_system/measured_voltage_angles.csv"

# Function to read CSV files using the csv module
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Get the headers
        data = [row for row in reader]  # Read all rows into a list
    return headers, data

# Read the CSV files
try:
    p_real_headers, p_real_data = read_csv(p_real_csv)
    p_img_headers, p_img_data = read_csv(p_img_csv)
    v_mag_headers, v_mag_data = read_csv(v_mag_csv)
    v_ang_headers, v_ang_data = read_csv(v_ang_csv)
except Exception as e:
    print(f"Error reading CSV files: {e}")
    exit(1)

# Key list for JSON conversion
key_list = ['values', 'ids', 'units', 'accuracy', 'bad_data_threshold', 'time', 'equipment_ids']

# Generate JSON format for a given row index
def create_json(headers, data, units, index):
    json_data = {}
    for key in key_list:
        json_data[key] = None
    if units == 'degrees':
        matching_columns = headers[1:]  # Skip the first header (assumed to be the date/time)
        matching_values = [float(val) for val in data[index][1:]]
        json_data = {
            'units': units,
            'equipment_ids': [],
            'ids': matching_columns,
            'values': matching_values,
            'time': parse_dates(data[index][0]).strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        json_data = {
            'units': units,
            'equipment_ids': [],
            'ids': headers[1:],  # Skip the first header (assumed to be the date/time)
            'values': [float(val) for val in data[index][1:]],
            'time': parse_dates(data[index][0]).strftime('%Y-%m-%d %H:%M:%S')
        }
    return json_data

# Generate all JSONs
def generate_all_jsons():
    for i in range(len(p_real_data)):
        p_real_json = create_json(p_real_headers, p_real_data, 'kW', i)
        p_img_json = create_json(p_img_headers, p_img_data, 'kVAR', i)
        v_mag_json = create_json(v_mag_headers, v_mag_data, 'kV', i)
        v_ang_json = create_json(v_ang_headers, v_ang_data, 'degrees', i)
        yield p_real_json, p_img_json, v_mag_json, v_ang_json


# make sure all the inputs are in the current folder
current_directory = os.getcwd()
TEST_DIR=current_directory
sys.path.append(os.path.dirname(TEST_DIR))

target_directory = r"testing_data_small_1_system"
os.makedirs(target_directory, exist_ok=True)


# modified functions with voltage angle measurements

# jacobian with angles
def calculate_jacobian(X0, z, num_node, knownP, knownQ, knownV, knownA, Y):
    # needs knownA
    # z has voltageMAg...P..Q..VoltageAngles
    """Calculate the Jacobian matrix for the weighted least squares algorithm.

    Called H in literature."""
    deltaK, VabsK = X0[:num_node], X0[num_node:]

    # gradients for voltage magnitude measurements
    num_knownV = len(knownV)
    # Calculate original H1
    H11, H12 = np.zeros((num_knownV, num_node)), np.zeros(num_knownV * num_node)
    H12[np.arange(num_knownV) * num_node + knownV] = 1
    # print(knownV)
    # pprint.pprint(H12, width=300)
    # np.savetxt('array_output.csv', H12, delimiter=',', fmt='%s')
    H1 = np.concatenate((H11, H12.reshape(num_knownV, num_node)), axis=1)

    # gradients for angle measurements
    num_knownA = len(knownA)
    # Calculate original H4

    if num_knownA != 0:
        H41, H42 = np.zeros(num_knownA * num_node), np.zeros((num_knownA, num_node))
        H41[np.arange(num_knownA) * num_node + knownA] = 1
        # np.savetxt('array_output_angleJAcobian2.csv', H41, delimiter=',', fmt='%s')
        H4 = np.concatenate((H41.reshape(num_knownA, num_node), H42), axis=1)
    else:
        H4 = np.empty((0, num_node * 2))

    # gradients for power injections
    Vp = VabsK * np.exp(1j * deltaK)
    ##### S = np.diag(Vp) @ Y.conjugate() @ Vp.conjugate()
    ######  Take gradient with respect to V
    H_pow2 = scipy.sparse.diags_array(Vp) @ Y.conjugate() @ scipy.sparse.diags_array(
        np.exp(-1j * deltaK)
    ) + scipy.sparse.diags_array(np.exp(1j * deltaK) * (Y.conjugate() @ Vp.conjugate()))
    # Take gradient with respect to delta
    H_pow1 = (
            1j
            * scipy.sparse.diags_array(Vp)
            @ (
                    scipy.sparse.diags_array(Y.conjugate() @ Vp.conjugate())
                    - Y.conjugate() @ scipy.sparse.diags_array(Vp.conjugate())
            )
    )

    if isinstance(Y, scipy.sparse.sparray):  # never actually enters in this condition, can delete later
        H2 = scipy.sparse.hstack((H_pow1.real, H_pow2.real))[knownP, :]
        H3 = scipy.sparse.hstack((H_pow1.imag, H_pow2.imag))[knownQ, :]
        assert isinstance(H2, scipy.sparse.sparray), f"H2 has type {type(H2)}"
        assert isinstance(H3, scipy.sparse.sparray), f"H3 has type {type(H3)}"
        H = scipy.sparse.vstack((H1, H2, H3, H4))
    else:
        H2 = np.concatenate((H_pow1.real, H_pow2.real), axis=1)[knownP, :]
        H3 = np.concatenate((H_pow1.imag, H_pow2.imag), axis=1)[knownQ, :]
        H = np.concatenate((H1, H2, H3, H4), axis=0)
    return -H


# calculates residuals including angles
def residual(X0, z, num_node, knownP, knownQ, knownV, knownA, Y):
    delta, Vabs = X0[:num_node], X0[num_node:]
    hx = estimated_pqva(knownP, knownQ, knownV, knownA, Y, delta, Vabs, num_node)
    #     logger.debug("X0")
    #     logger.debug(X0)
    #     logger.debug("z")
    #     logger.debug(z)
    #     logger.debug("h")
    #     logger.debug(h)
    #     print("HI")
    #    print(z-hx)
    return z - hx


# as is
def get_y(admittance: Union[AdmittanceMatrix, AdmittanceSparse], ids: List[str]):
    # def get_y(admittance, ids):
    if type(admittance) == AdmittanceMatrix:
        assert ids == admittance.ids
        return matrix_to_numpy(admittance.admittance_matrix)
    elif type(admittance) == AdmittanceSparse:
        node_map = {name: i for (i, name) in enumerate(ids)}
        return scipy.sparse.coo_array(
            (
                [v[0] + 1j * v[1] for v in admittance.admittance_list],
                (
                    [node_map[r] for r in admittance.from_equipment],
                    [node_map[c] for c in admittance.to_equipment],
                ),
            )
        )


# as is
def matrix_to_numpy(admittance: List[List[Complex]]):
    "Convert list of list of our Complex type into a numpy matrix"
    return np.array([[x[0] + 1j * x[1] for x in row] for row in admittance])


# as is
def get_indices(topology, measurement):
    "Get list of indices in the topology for each index of the input measurement"
    inv_map = {v: i for i, v in enumerate(topology.base_voltage_magnitudes.ids)}
    return [inv_map[v] for v in measurement.ids]


# ----------------------------
# PSEUDO-MEASUREMENT INJECTION
# ----------------------------

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def add_pseudo_pq_measurements(
    input_data_dir: str,
    topology: Topology,
    prev_p: dict,
    prev_q: dict,
    fill_mode: str = "prev_then_mean",   # "prev_then_mean" | "mean" | "zero"
):
    """
    Adds pseudo P/Q injections for any topology buses missing from power_real.json / power_imag.json.

    Returns:
      pseudoP_ids (set[str]), pseudoQ_ids (set[str])  -> IDs that were injected as pseudo this timestep
      prev_p, prev_q updated with latest available values
    """

    all_bus_ids = list(topology.base_voltage_magnitudes.ids)

    p_path = os.path.join(input_data_dir, "power_real.json")
    q_path = os.path.join(input_data_dir, "power_imag.json")

    Pj = _load_json(p_path)
    Qj = _load_json(q_path)

    # Existing measured IDs
    measP_ids = list(Pj.get("ids", []))
    measQ_ids = list(Qj.get("ids", []))

    measP_vals = list(Pj.get("values", []))
    measQ_vals = list(Qj.get("values", []))

    measP_set = set(measP_ids)
    measQ_set = set(measQ_ids)

    # Compute "typical" fill values if needed
    meanP = float(np.mean(measP_vals)) if len(measP_vals) else 0.0
    meanQ = float(np.mean(measQ_vals)) if len(measQ_vals) else 0.0

    pseudoP_ids, pseudoQ_ids = set(), set()

    # Update prev_* with the current measured values (so next timestep can use them)
    for _id, _val in zip(measP_ids, measP_vals):
        prev_p[_id] = float(_val)
    for _id, _val in zip(measQ_ids, measQ_vals):
        prev_q[_id] = float(_val)

    # Decide the fill function
    def _fill(prev_dict, mean_val, bus_id):
        if fill_mode == "prev_then_mean":
            return float(prev_dict.get(bus_id, mean_val))
        elif fill_mode == "mean":
            return float(mean_val)
        elif fill_mode == "zero":
            return 0.0
        else:
            raise ValueError(f"Unknown fill_mode={fill_mode}")

    # Add pseudo P for missing buses
    for bus_id in all_bus_ids:
        if bus_id not in measP_set:
            measP_ids.append(bus_id)
            measP_vals.append(_fill(prev_p, meanP, bus_id))
            pseudoP_ids.add(bus_id)

    # Add pseudo Q for missing buses
    for bus_id in all_bus_ids:
        if bus_id not in measQ_set:
            measQ_ids.append(bus_id)
            measQ_vals.append(_fill(prev_q, meanQ, bus_id))
            pseudoQ_ids.add(bus_id)

    # Write back
    Pj["ids"] = measP_ids
    Pj["values"] = measP_vals
    Qj["ids"] = measQ_ids
    Qj["values"] = measQ_vals

    _save_json(p_path, Pj)
    _save_json(q_path, Qj)

    return pseudoP_ids, pseudoQ_ids, prev_p, prev_q


# ----------------------------
# WEIGHTED RESIDUAL + WEIGHTED JACOBIAN
# ----------------------------

def residual_weighted(X0, z, sigma, num_node, knownP, knownQ, knownV, knownA, Y):
    """Weighted residual: r_w = (z - h(x)) / sigma"""
    delta, Vabs = X0[:num_node], X0[num_node:]
    hx = estimated_pqva(knownP, knownQ, knownV, knownA, Y, delta, Vabs, num_node)
    r = z - hx
    return r / sigma

def calculate_jacobian_weighted(X0, z, sigma, num_node, knownP, knownQ, knownV, knownA, Y):
    """Weighted Jacobian for weighted residual: J_w = (∂r/∂x)/sigma = (-H)/sigma"""
    H = calculate_jacobian(X0, z, num_node, knownP, knownQ, knownV, knownA, Y)  # returns -H already
    # scale each measurement row by 1/sigma
    if scipy.sparse.issparse(H):
        inv_sigma = scipy.sparse.diags(1.0 / sigma)
        return inv_sigma @ H
    else:
        return H / sigma.reshape(-1, 1)

def estimated_pqva(knownP, knownQ, knownV, knownA, Y, deltaK, VabsK, num_node):
    """Calculate estimated P, Q, and V and A."""
    h1 = (VabsK[knownV]).reshape(-1, 1)
    Vp = VabsK * np.exp(1j * deltaK)
    S = Vp * (Y.conjugate() @ Vp.conjugate())
    P, Q = S.real, S.imag
    h2, h3 = P[knownP].reshape(-1, 1), Q[knownQ].reshape(-1, 1)
    h4 = (deltaK[knownA]).reshape(-1, 1)
    #     print("inside_estimatedpqva")##########################################################
    #     print(h4)
    h = np.concatenate((h1, h2, h3, h4), axis=0)
    return h.reshape(-1)



#gives measurement
def get_measurements(directory):
    return (
        PowersReal.parse_file(os.path.join(directory, "power_real.json")),
        PowersImaginary.parse_file(os.path.join(directory, "power_imag.json")),
        VoltagesMagnitude.parse_file(os.path.join(directory, "voltage_magnitude.json")),
        VoltagesAngle.parse_file(os.path.join(directory, "voltage_angle.json")),
    )

def get_topology(directory,topotype):
    if topotype==1:
        return Topology.parse_file(os.path.join(directory, "topology1.json"))
    else:
        return Topology.parse_file(os.path.join(directory, "topology2.json"))

# for validation later
def get_actuals(directory):
    return (
        VoltagesReal.parse_file(os.path.join(directory, "voltage_real.json")),
        VoltagesImaginary.parse_file(os.path.join(directory, "voltage_imaginary.json")),
    )

# provides the z, indices and Ybus in pu form
def inner_args(parameters, topology, measurements, pseudoP_ids=None, pseudoQ_ids=None):
    P, Q, V, A = measurements

    pseudoP_ids = pseudoP_ids or set()
    pseudoQ_ids = pseudoQ_ids or set()

    knownP = get_indices(topology, P)
    knownQ = get_indices(topology, Q)
    knownV = get_indices(topology, V)
    knownA = get_indices(topology, A)

    base_voltages = np.array(topology.base_voltage_magnitudes.values)
    num_node = len(topology.base_voltage_magnitudes.ids)
    base_power = parameters.base_power

    # Building z
    z = np.concatenate(
        (
            V.values / base_voltages[knownV],
            -np.array(P.values) / base_power,
            -np.array(Q.values) / base_power,
            A.values,
        ),
        axis=0,
    )

    # Building sigma vector (weights)
    # Smaller sigma => stronger trust; larger sigma => weaker trust (pseudo)
    sigma_V = np.full(len(knownV), getattr(parameters, "sigma_v_meas", 0.01))
    sigma_A = np.full(len(knownA), getattr(parameters, "sigma_a_meas", 0.01))

    # For P/Q we set sigma based on whether ID is pseudo
    sigma_P = []
    for mid in P.ids:
        if mid in pseudoP_ids:
            sigma_P.append(getattr(parameters, "sigma_p_pseudo", 5.0))
        else:
            sigma_P.append(getattr(parameters, "sigma_p_meas", 0.5))
    sigma_P = np.array(sigma_P)

    sigma_Q = []
    for mid in Q.ids:
        if mid in pseudoQ_ids:
            sigma_Q.append(getattr(parameters, "sigma_q_pseudo", 5.0))
        else:
            sigma_Q.append(getattr(parameters, "sigma_q_meas", 0.5))
    sigma_Q = np.array(sigma_Q)

    sigma = np.concatenate((sigma_V, sigma_P, sigma_Q, sigma_A), axis=0)

    # Ybus in pu
    Y = get_y(topology.admittance, topology.base_voltage_magnitudes.ids)
    Y = (
        scipy.sparse.diags_array(base_voltages)
        @ Y
        @ scipy.sparse.diags_array(base_voltages)
    ) / (base_power * 1000)

    initial_ang = np.array(topology.base_voltage_angles.values)
    X0 = np.concatenate((initial_ang, np.full(num_node, 1)))

    return X0, z, sigma, num_node, knownP, knownQ, knownV, knownA, Y



# functions related to error calculation

#error for voltages
def get_mean_relative_error(topology, solution, actuals):
    vmagestDecen, vangestDecen = (
        solution[len(solution) // 2 :],
        solution[: len(solution) // 2],
    )

    slack_id = topology.base_voltage_magnitudes.ids.index(topology.slack_bus[0])
    vangestDecen = vangestDecen - vangestDecen[slack_id]

    voltage_mag = vmagestDecen * np.array(topology.base_voltage_magnitudes.values)
    voltage_ang = vangestDecen

    voltage_real, voltage_imag = actuals
    true_voltage = np.array(voltage_real.values) + 1j * np.array(voltage_imag.values)
    estimated_voltage = voltage_mag * np.exp(1j * voltage_ang)

    return np.abs(
        (estimated_voltage - true_voltage)
        / np.array(topology.base_voltage_magnitudes.values)
    ).mean()

#error for angles
def get_mean_angle_error(topology, solution, actuals):
    vmagestDecen, vangestDecen = (
        solution[len(solution) // 2 :],
        solution[: len(solution) // 2],
    )

    slack_id = topology.base_voltage_magnitudes.ids.index(topology.slack_bus[0])
    vangestDecen = vangestDecen - vangestDecen[slack_id]

    voltage_mag = vmagestDecen * np.array(topology.base_voltage_magnitudes.values)
    voltage_ang = vangestDecen

    voltage_real, voltage_imag = actuals
    true_voltage = np.array(voltage_real.values) + 1j * np.array(voltage_imag.values)
    estimated_voltage = voltage_mag * np.exp(1j * voltage_ang)

    return np.abs(np.angle(estimated_voltage * true_voltage.conj())).mean()

# functions related to error calculation


########################################################################################################################

#downwards are all test functions


# function that runs wls

### least square call
# def test_mean_absolute_error_least_squares(parameters, input_data, topotype):
def test_mean_absolute_error_least_squares(parameters, input_data, topotype, pseudoP_ids=None, pseudoQ_ids=None):
    print(topotype)

    # to track optimization history
    # Variables to track cost at each iteration

    class LeastSquaresTracker:
        def __init__(self, residual_func):
            self.residual_func = residual_func
            self.cost_history = []
            self.residual_history = []
            self.iteration = 0

        def wrapped_residual(self, *args):
            residuals = self.residual_func(*args)
            current_cost = 0.5 * np.sum(residuals ** 2)

            # Calculate cost reduction if this is not the first iteration
            if self.cost_history:
                cost_reduction = self.cost_history[-1] - current_cost
            else:
                cost_reduction = None

            self.cost_history.append(current_cost)
            self.iteration += 1
            print(f"Iteration {self.iteration}: Cost = {current_cost}, Cost Reduction = {cost_reduction}")

            return residuals

    # Instantiate the tracker for debugging
    tracker = LeastSquaresTracker(residual)

    topology = get_topology(input_data, topotype)
    measurements = get_measurements(input_data)
    X0, z, sigma, num_node, knownP, knownQ, knownV, knownA, Y = inner_args(
        parameters, topology, measurements, pseudoP_ids=pseudoP_ids, pseudoQ_ids=pseudoQ_ids
    )
    # tracker with weighted residual
    tracker = LeastSquaresTracker(residual_weighted)
    #     print(knownA)
    ls_result = scipy.optimize.least_squares(
        tracker.wrapped_residual,
        x0=X0,
        jac=calculate_jacobian,
        # bounds=(low_limit, up_limit),
        method="trf",
        # method="lm",
        verbose=2,
        ftol=parameters.ftol1,
        xtol=parameters.xtol1,
        gtol=parameters.gtol1,
        #         max_nfev=parameters.nfev1,
        args=(z, num_node, knownP, knownQ, knownV, knownA, Y),
    )
    assert ls_result.success, f"Least squares failed: {ls_result.message}"

    solution = ls_result.x
    # solution
    # print(solution)

    vmagestDecen, vangestDecen = (
        solution[len(solution) // 2:],
        solution[: len(solution) // 2],
    )

    # count = 0
    base_voltages = np.array(topology.base_voltage_magnitudes.values)

    # checking residuals
    # print(X0)
    # final_residual = residual(solution, z, num_node, knownP, knownQ, knownV, knownA, Y)
    final_residual_w = residual_weighted(solution, z, sigma, num_node, knownP, knownQ, knownV, knownA, Y)
    final_residual_unw = (final_residual_w * sigma)  # converting back to raw residual

    # print(final_residual)

    final_cost = tracker.cost_history[-1]
    print(f"Least Squares completed")
    return vmagestDecen * base_voltages, vangestDecen, final_residual_w, final_cost


# new functions

def delete_measurement(input_data, meas_type, index_to_delete):
    #     print(index_to_delete)
    if meas_type == 1:  # voltage_meas
        filename = os.path.join(input_data, "voltage_magnitude.json")
    elif meas_type == 2:  # P_meas
        filename = os.path.join(input_data, "power_real.json")
    elif meas_type == 3:  # Q_meas
        filename = os.path.join(input_data, "power_imag.json")
    elif meas_type == 4:  # Angle_meas
        filename = os.path.join(input_data, "voltage_angle.json")

        # Load the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)
    values = data['values']
    ids = data['ids']
    print(f"BD_key")
    print(ids[index_to_delete])
    del values[index_to_delete]
    del ids[index_to_delete]
    #     print(f"Deleted row at index {index_to_delete}.")
    data['values'] = values
    data['ids'] = ids
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Modified data saved back to {filename}.")

# main

class AlgorithmParameters():
    def __init__(self):
        self.base_power = 100  # Initialize the 'value' attribute
        self.tol = 5e-7
        self.ftol1=.0075
        self.xtol1=.001
        self.gtol1=.001
        self.nfev1=50
        # measurement sigmas (smaller = trust more)
        self.sigma_v_meas = 0.01
        self.sigma_a_meas = 0.01

        self.sigma_p_meas = 0.5
        self.sigma_q_meas = 0.5

        # pseudo sigmas (larger = trust less)
        self.sigma_p_pseudo = 5.0
        self.sigma_q_pseudo = 5.0

prev_p, prev_q = {}, {}
parameters = AlgorithmParameters()

INPUT_DATA=os.path.join(TEST_DIR, "testing_data_small_system")
topology = get_topology('testing_data_small_system',1)
csv_file_path = "est_v_mag.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestep"] + topology.base_voltage_magnitudes.ids)
topo_file_path = "topology_changes.csv"
with open(topo_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestep"] + ["Topology"])

## looping the values start here
for p_real_json, p_img_json, v_mag_json, v_ang_json in generate_all_jsons():

    recorded_time = [v_mag_json['time']]
# Save the variables in JSON file
    dicts = [p_real_json, p_img_json, v_mag_json, v_ang_json]
    file_names = [
        'power_real_original.json',
        'power_imag_original.json',
        'voltage_magnitude_original.json',
        'voltage_angle_original.json'
    ]

    for file_name, data in zip(file_names, dicts):
        file_path = os.path.join(target_directory, file_name)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)

    ##################################### first load from original measurement data

    # real_power
    input_filename = os.path.join(INPUT_DATA, "power_real_original.json")
    with open(input_filename, 'r') as infile: data = json.load(infile)
    output_filename = os.path.join(INPUT_DATA, "power_real.json")
    with open(output_filename, 'w') as outfile: json.dump(data, outfile, indent=4)

    # reactive_power
    input_filename = os.path.join(INPUT_DATA, "power_imag_original.json")
    with open(input_filename, 'r') as infile: data = json.load(infile)
    output_filename = os.path.join(INPUT_DATA, "power_imag.json")
    with open(output_filename, 'w') as outfile: json.dump(data, outfile, indent=4)

    # voltage_magnitude
    input_filename = os.path.join(INPUT_DATA, "voltage_magnitude_original.json")
    with open(input_filename, 'r') as infile: data = json.load(infile)
    output_filename = os.path.join(INPUT_DATA, "voltage_magnitude.json")
    with open(output_filename, 'w') as outfile: json.dump(data, outfile, indent=4)

    # voltage_angle
    input_filename = os.path.join(INPUT_DATA, "voltage_angle_original.json")
    with open(input_filename, 'r') as infile: data = json.load(infile)
    output_filename = os.path.join(INPUT_DATA, "voltage_angle.json")
    with open(output_filename, 'w') as outfile: json.dump(data, outfile, indent=4)
    ## Straightforward topo checker
    parameters.xtol1 = .01

    # using pseudo-measurements for missing P/Q
    topology_tmp = get_topology(INPUT_DATA, 1)
    pseudoP_ids, pseudoQ_ids, prev_p, prev_q = add_pseudo_pq_measurements(INPUT_DATA, topology_tmp, prev_p, prev_q,
                                                                          fill_mode="prev_then_mean")

    topotype = 1
    estVmag, estVang, final_residual, fcost1 = test_mean_absolute_error_least_squares(parameters, INPUT_DATA, topotype,
                                                                                      pseudoP_ids, pseudoQ_ids)

    topotype = 2
    estVmag, estVang, final_residual, fcost2 = test_mean_absolute_error_least_squares(parameters, INPUT_DATA, topotype,
                                                                                      pseudoP_ids, pseudoQ_ids)


    if fcost1 < fcost2:
        topotype = 1
    else:
        topotype = 2  ##############dump the result from here

    print(f"finalized topology:{topotype}")

    # BDD checker
    #################################################################################
    topology = get_topology(INPUT_DATA, topotype)
    measurements = get_measurements(INPUT_DATA)
    X0, z, num_node, knownP, knownQ, knownV, knownA, Y = inner_args(
        parameters, topology, measurements)

    n_states = num_node * 2
    num_knownP, num_knownQ, num_knownV, num_knownA = len(knownP), len(knownQ), len(knownV), len(knownA)
    n_meas = len(z)
    n_BD = 0

    parameters.xtol1 = 0.0008  ###################need sth automated to change for large and small system &finetune
    parameters.cost_threshold = .01  ###################need sth automated to change for large and small system &finetune #large==5, small =-01

    # start loop

    print("Iteration starts!")
    while True:
        estVmag, estVang, final_residual, fcost = test_mean_absolute_error_least_squares( parameters, INPUT_DATA, topotype, pseudoP_ids, pseudoQ_ids)
            # test_mean_absolute_error_least_squares(parameters, INPUT_DATA, topotype)
        topology = get_topology(INPUT_DATA, topotype)
        measurements = get_measurements(INPUT_DATA)
        X0, z, num_node, knownP, knownQ, knownV, knownA, Y = inner_args(
            parameters, topology, measurements)
        num_knownP, num_knownQ, num_knownV, num_knownA = len(knownP), len(knownQ), len(knownV), len(knownA)

        print(fcost)
        if fcost < parameters.cost_threshold:
            print(f"no BAD DATA, good to go")
            break
        else:
            print(f"Possible BDD exits")
            n_BD += 1
            indices = np.argsort(abs(final_residual))  # Get indices of the sorted array in ascending order
            worst_measurement = indices[-1]

            if (worst_measurement >= 0) & (worst_measurement < num_knownV):
                print(f"Its a Bad voltage measurement")
                delete_measurement(INPUT_DATA, 1, worst_measurement)

            if (worst_measurement >= num_knownV) & (worst_measurement < num_knownV + num_knownP):
                print(f"Its a Bad real power measurement")
                delete_measurement(INPUT_DATA, 2, worst_measurement - num_knownV)

            if (worst_measurement >= num_knownV + num_knownP) & (worst_measurement < num_knownV + num_knownP + num_knownQ):
                print(f"Its a Bad reactive power measurement")
                delete_measurement(INPUT_DATA, 3, worst_measurement - num_knownV - num_knownP)

            if (worst_measurement >= num_knownV + num_knownP + num_knownQ) & (
                    worst_measurement < num_knownV + num_knownP + num_knownQ + num_knownA):
                print(f"Its a Bad angle measurement")
                delete_measurement(INPUT_DATA, 4, worst_measurement - num_knownV - num_knownP - num_knownQ)
    recorded_time = [v_mag_json['time']]
    pu_estVmag = estVmag / topology.base_voltage_magnitudes.values

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(v_mag_json['time'] + estVmag)
        writer.writerow([v_mag_json['time']] + pu_estVmag.tolist())
    with open(topo_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([v_mag_json] + [topotype])  # Convert topotype to a list


