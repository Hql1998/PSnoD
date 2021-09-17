# PSnoD <br />
PSnoD codes for predicting snoRNA-disease association using matrix completion technique<br />

########################<br />
before running the program, be sure your system installed python version 3.6 plus<br />
and packages:<br />
pip install cvxpy<br />
pip install matrix_completion<br />
and sklearn, numpy, pandas, matplotlib, scipy<br />
if you are only interested in BNNR, you could ignore matrix_completion and cvxpy installations.<br />

#######################<br />
all codes lie in PSnoD_WorkFlow directory , you could import it<br />
the program relies input_data, output_csv, output_images, the three directories.<br />
check file main.py to find the usage<br />

#######################<br />
if you are only interested in the BNNR completion method, and want to complete a matrix, please directly use:<br />

from PSnoD_WorkFlow.BNNR import bnnr<br />
completed_matrix, iterations = bnnr(matrix.to_numpy(), mask.to_numpy(), alpha=param_a, beta=param_b)<br />

1. where matrix is the a pandas dataframe object, and composed of disease matrix at left upper corner,<br />
sonRNA matrix at right lower corner and relation matrix at right upper corner.<br />
2. the mask matrix is as the same size with the matrix you need to be completed, but the element within it<br />
is only 0 or 1, which 0 represent the relation is unknown, and 1 represent known<br />
3. alpha and beta are hyperparameters<br />

