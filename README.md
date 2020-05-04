# Bayesian Link Adaptation under a BLER Target

This respository contains the simulation code for running the numerical experiments reported in the paper:  
**Vidit Saxena and Joakim Jaldén,"Bayesian Link Adaptation under a BLER Target", In 2020 IEEE 21st International Workshop on Signal Processing Advances in Wireless Communications (SPAWC) on May 26-29, 2020.** 

This simulation code is written in Python3. Running each of the cells in the corresponding [`Jupyter Notebooks`](https://github.com/jupyter/notebook) will execute the experiments, generate a results file, and plot the results.

The simulations make extensive use of the [`py-itpp`](https://github.com/vidits-kth/py-itpp), [`Numpy`](https://github.com/numpy/numpy), and [`Matplotlib`](https://github.com/matplotlib/matplotlib) packages.

Additionally, to speed up the generation of results, the simulations are parallezlized using the [`Ray`](http://ray.readthedocs.io/en/latest/index.html) package. It is possible to run single-threaded simulations at the cost of slowness, by commenting out the Ray-specific lines in the notebook - this is indicated in the appropriate sections of the code.

## Files  
`Link Adaptation - OLLA and BayesLA.ipynb` contains the code for running the experiments and saving the results to disk.  
`Plot Results.ipynb` contains the code for plotting experiment results from the result file read from the disk.  
`source.py` contains helper code for simulating a Rayleigh fading wireless channel and for the OLLA and BayesLA algorithms.  
`AWGN_DATASET.npy` contains offline lookup data for mapping between instantaneous channel SNR and CQI values for each MCS.  

