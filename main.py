# if you are only interested in the BNNR completion method, and want to complete a matrix, please directly use

# from PSnoD_WorkFlow.BNNR import bnnr
# completed_matrix, iterations = bnnr(matrix.to_numpy(), mask.to_numpy(), alpha=param_a, beta=param_b)

# 1. where matrix is the a pandas dataframe object, and composed of disease matrix at left upper corner,
# sonRNA matrix at right lower corner and relation matrix at right upper corner.
# 2. the mask matrix is as the same size with the matrix you need to be completed, but the element within it
# is only 0 or 1, which 0 represent the relation is unknown, and 1 represent known
# 3. alpha and beta are hyperparameters


from PSnoD_WorkFlow import Training_Model
from PSnoD_WorkFlow import get_color
from matplotlib.colors import LinearSegmentedColormap

# color map for heatmap
color1 = get_color((122, 31, 46))
color2 = get_color((250, 163, 178))
cmap_heatmap = LinearSegmentedColormap.from_list('my_color', [color1, color2], N=100)

# color map for matrix
color1 = get_color((68, 109, 169)) #dark blue
color_base = get_color((255, 255, 255)) #whitw
cmap_matrix = LinearSegmentedColormap.from_list('my_color', [color_base, color1], N=50)

# colors for ROC
pink = get_color((239, 129, 145))
blue = get_color((86, 108, 179))
colors_metrics = [pink, blue, "orange"]

method_list = ["BNNR",] #"SVT", "Candès and Recht's method"

###### attention ######
# using this program, you need provide 3 kind of files:
# the A similarity csv files (in this article, it's disease similarity matrix)
# the B similarity csv files (in this article, it's snoRNA similarity matrix)
# the relation matrix, where row labels are matrix A's row or column labels and column labels are matrix B's row or column labels

# the program will iterate all matrix completion method in the method_list variable and all snoRNA sequence similarity csv files,
# and use cross validation to test the performance, ploting ROC and AUCs

# disease_sim_path is the disease_similarity csv files lies in
disease_sim_path = "./input_data/disease_sim_graph_filtered.csv"

# relationship_matrix_path variable is the path where the relation matrix lies in,
# the relation_matrix's row name or row label should be disease name or mesh id
# the relation_matrix's column name or label should be snoRNA name or mesh id
# the value in the matrix is either 1 or 0, showing existing relation or not
relationship_matrix_path = "./input_data/relationship_matrix_filtered.csv"

# snoRNA sequence simliarity csv file path are composed from 2 varible,
# first seq_sim_dir_path is the directory where these sim files in,
# second similarity csv file you choosed to include,
# all sim files should computed from the same set of snoRNAs.
seq_sim_dir_path = "./input_data/snoRNA_sim"
seq_simlarity_list = ["snoRNA_4mer_similarity.csv"]

# initialize the training model, like cv_fold >= 2, test_set_size,
# method list, if you are only interested in BNNR, you cold try method_list=["BNNR"] (upper case only)
# seq_simlarity_list you defined before
# fig_dir: directory path where result figures save, the last slash / is required
# csv_dir: directory path where result csv files save, the last slash / is required
# to change the hyperparameters, check class Training_Model definition -> init_hyperparameter(self)
train_model = Training_Model(cv_fold=5,
                             test_size=0.2,
                             method_list=method_list,
                             seq_simlarity_list=seq_simlarity_list,
                             fig_dir=r"./output_images/",
                             csv_dir=r"./output_csv/",
                             colors=colors_metrics,
                             cmap=cmap_heatmap,
                             result_cmp=cmap_matrix)

train_model.run(relationship_matrix_path, disease_sim_path, seq_sim_dir_path)

