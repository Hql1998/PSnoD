# PSnoD
PSnoD codes for predicting snoRNA-disease association using matrix completion technique

########################
before running the program, be sure your system installed python version 3.6 plus
and packages:
pip install cvxpy
pip install matrix_completion
and sklearn, numpy, pandas, matplotlib, scipy
if you are only interested in BNNR, you could ignore matrix_completion and cvxpy installations.

#######################
all codes lie in PSnoD_WorkFlow directory , you could import it
the program relies input_data, output_csv, output_images, the three directories.
check file main.py to find the usage

#######################
if you are only interested in the BNNR completion method, and want to complete a matrix, please directly use:

from PSnoD_WorkFlow.BNNR import bnnr
completed_matrix, iterations = bnnr(matrix.to_numpy(), mask.to_numpy(), alpha=param_a, beta=param_b)

1. where matrix is the a pandas dataframe object, and composed of disease matrix at left upper corner,
sonRNA matrix at right lower corner and relation matrix at right upper corner.
2. the mask matrix is as the same size with the matrix you need to be completed, but the element within it
is only 0 or 1, which 0 represent the relation is unknown, and 1 represent known
3. alpha and beta are hyperparameters

