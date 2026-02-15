
AC distribution system state estimation (DSSE) over time series data to recover full bus voltage magnitudes and angles from sparse V P Q and angle measurements using nonlinear least squares. Includes topology selection via minimum cost and residual-based bad data detection. Optional pseudo P and Q fill gaps to improve observability.

# Distribution System State Estimation with Topology Selection and Pseudo Measurement Gap Filling

## Overview
This repository provides a reproducible distribution system state estimation workflow for a three phase distribution feeder represented using a bus admittance model. The implementation performs nonlinear alternating current state estimation from time series measurements of voltage magnitudes, voltage angles, and net active and reactive power injections. The estimator is executed at each timestamp to generate a sequence of static state estimates.

The workflow includes topology selection between two candidate feeder models by comparing the final least squares objective value under each network admittance matrix. A residual driven bad data detection routine is included to identify and remove the most inconsistent measurement when the objective value exceeds a user defined threshold. Optional pseudo measurement injection can be enabled to fill gaps in sparse metering by creating weakly weighted active and reactive power pseudo measurements for unmeasured buses using recent history and population level statistics from the current measurement set.

## Key Capabilities
1. Nonlinear distribution system state estimation using an alternating current power flow based measurement function  
2. Topology selection between two candidate admittance models based on minimum final cost  
3. Residual based bad data detection and iterative measurement removal  
4. Optional pseudo measurement generation for missing power injections with configurable strength via uncertainty parameters  
5. Batch processing of time series data from comma separated value files and automatic conversion to the measurement json format used by the solver  
6. Export of estimated voltage magnitude profiles per timestamp to a results file for downstream analysis

## Mathematical Formulation
### State Vector
For a system with N buses the estimated state is

x = [ δ1 δ2 … δN  |V1| |V2| … |VN| ]ᵀ

where δi is the voltage phase angle at bus i and |Vi| is the voltage magnitude at bus i.

### Network Model
Let Y denote the complex bus admittance matrix. The complex bus voltage vector is

V = |V| ⊙ exp(jδ)

where ⊙ denotes element wise multiplication. The net complex power injection at each bus is computed as

S = V ⊙ ( Y* V* )

where * denotes complex conjugation. Active and reactive injections follow

P = Re{S}
Q = Im{S}

### Measurement Model
At each timestamp a measurement vector z is constructed by stacking available measurements

z = [ |V|m  Pm  Qm  δm ]ᵀ

The corresponding nonlinear measurement function is

h(x) = [ |V(x)|m  P(x)m  Q(x)m  δ(x)m ]ᵀ

The residual vector is

r(x) = z − h(x)

### Objective Function
The estimator solves a nonlinear least squares problem

minimize over x   J(x) = 1/2  r(x)ᵀ r(x)

If pseudo measurements are enabled the implementation can be used in a weighted form by scaling each residual component by its standard deviation σk

rw,k(x) = rk(x) / σk

and the objective becomes

minimize over x   Jw(x) = 1/2  rw(x)ᵀ rw(x)

### Jacobian
The solver uses an analytical Jacobian for improved convergence. The Jacobian of the residual is

Jr(x) = ∂r(x) / ∂x = − ∂h(x) / ∂x

Voltage magnitude and angle measurement rows include identity gradients with respect to |V| and δ at the measured indices. Power injection gradients are formed using complex derivatives of S with respect to δ and |V| and are assembled into the full Jacobian used by the trust region reflective solver.

## Topology Selection
Two candidate feeder topologies are represented by two Y matrices derived from topology1 json and topology2 json. For each timestamp the estimator is executed twice

J1 = final cost using topology 1
J2 = final cost using topology 2

The selected topology is

topology selected = argmin over t in {1,2}  Jt

This procedure supports switch status uncertainty and model mismatch studies in distribution feeders.

## Bad Data Detection
After topology selection the estimator is executed iteratively. If the final cost exceeds a threshold Jthr the measurement with the largest residual magnitude is identified and removed from its measurement json file. The estimator then re runs until the final cost satisfies

J ≤ Jthr

This routine provides practical resilience against gross measurement errors and communication anomalies.

## Pseudo Measurement Gap Filling
Distribution feeders are commonly under metered. If pseudo measurement injection is enabled the script augments P and Q injection lists to ensure each bus has a power injection constraint. Missing buses receive pseudo values computed from recent history or from the mean of currently available measurements depending on the configured mode.

Pseudo measurements are assigned larger uncertainty so that they guide the solution without overpowering real sensor data. For example

σP,pseudo ≫ σP,meas
σQ,pseudo ≫ σQ,meas

This design enforces observability while preserving the dominance of real measurements in the objective.

## Repository Structure
data_types.py  
Typed parsers for measurement and topology json files including Topology, PowersReal, PowersImaginary, VoltagesMagnitude, VoltagesAngle, and admittance matrix containers

testing_data_small_system  
Input dataset directory containing time series measurement csv files and the json templates used by the solver

testing_data_small_1_system  
Generated json outputs per timestamp that mirror the measurement format expected by the solver

main_script.py  
Primary executable script that reads csv time series measurements, writes per timestamp json measurement files, performs topology selection, performs state estimation, executes bad data detection, and exports estimated voltage profiles

est_v_mag.csv  
Output file containing per timestamp estimated voltage magnitudes in per unit for all buses

topology_changes.csv  
Output file containing per timestamp topology selection results

## Data Format
### Measurement CSV
Each measurement csv file contains a timestamp column followed by bus id columns. Supported inputs include

measured_voltage_magnitudes.csv  
measured_voltage_angles.csv  
measured_active_power.csv  
measured_reactive_power.csv  

Timestamps are expected in month day year hour minute format.

### Measurement JSON
At runtime each timestamp is converted into json objects with fields

ids  
values  
units  
time  

The ids correspond to bus identifiers consistent with the topology base voltage ids.

### Topology JSON
The topology json defines

base voltage magnitudes per bus  
base voltage angles per bus  
slack bus id  
admittance representation as dense or sparse form  

## Installation
Python 3.9 or later is recommended.

Create an environment and install required packages

pip install numpy scipy pytest

Additional dependencies may be required if your local data_types implementation imports extra packages.

## Running the Workflow
Place input csv files in the input directory and ensure topology1 json and topology2 json exist.

Run the main script

python main_script.py

The script will iterate through all timestamps found in the input csv files and will produce estimated voltage magnitudes and selected topology per timestamp.

## Configuration Parameters
AlgorithmParameters includes key tunable values

base_power  
Per unit base power in kVA or kW consistent with the measurement unit scaling used to form z

ftol1 xtol1 gtol1  
Nonlinear least squares termination criteria used by the trust region reflective optimizer

cost_threshold  
Bad data detection threshold for the final objective value

sigma settings when pseudo measurements are enabled  
sigma_v_meas  
sigma_a_meas  
sigma_p_meas  
sigma_q_meas  
sigma_p_pseudo  
sigma_q_pseudo  

Larger sigma values represent lower confidence.

## Engineering Notes
Ensure consistent units for voltage angles. The state vector uses δ within exp(jδ) which expects radians. If input angles are provided in degrees they should be converted to radians before forming z and before initializing X0.

Sign convention for P and Q should match the injection model. The script currently applies a negative sign when stacking P and Q into z which corresponds to a load positive convention when the model uses injection positive.

## Citation and Attribution
If you use this repository in academic work cite it as a distribution system state estimation implementation featuring topology selection and pseudo measurement based observability support.


