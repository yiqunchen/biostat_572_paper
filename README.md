# biostat_572_paper
We document code and procedures to reproduce the paper "Exact spike train inference  via $\ell_0$ optimization"

- example_notebook.ipynb demonstrates the use of our code by reproducing figure 2 in the expository piece

- code/
	- ell_0_spike.py reproduces the timing result in figure 3
	- cv_ell_0_spike.py reproduces the simulation result in figure 4
	- cv_ell_0_spike_mis_tau.py generates the sensitivity analysis result in the appendix 
	- chen_data_l0.py generates the figure for Chen et al. data
	- allen_data_l0.py generates the figure for the Allen Institute data

- stat_572_report.pdf: the actual report