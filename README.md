# net-ev-accum
Simulations for models of evidence accumulation on a network.

## Basic Models
* ddm_fp_sim.py = Fokker-Planck approximation
* evAccum.py = Direct Simulation 
* nonDecDiscrete.py = the equilibration process 

## General Networks
* directed_line.py = simulation for a directed line of agents
* first_exit.py = the first exit times for large clique (OLD)
* exit_comp.py = updated first_exit.py
* clique_sim.py = some pseudo-code for the clique process
* CliqueInfo.py = simulation to get information for larger cliques

## One Agent
Mostly obsolete code used for early exploration:
* survivalProb.py = computes the survival probability for a single agent accumulating evidence. 

## Plotting
Some test code for plotting. Not needed since plots are produced in the other scripts.

## Two Agents
* threshOpt.py = code to test what update amount is optimal
* nonDecUpdate.py = some skeleton code which is refined in nonDecDiscrete.py

