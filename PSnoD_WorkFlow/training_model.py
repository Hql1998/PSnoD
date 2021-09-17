import time
import os

import pandas as pd
from numpy import interp
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sn

from matrix_completion import svt_solve, nuclear_norm_solve
from PSnoD_WorkFlow.BNNR import *



class Training_Model():
    """
    1. the center running method of this class is last function: run(self, ...)
    2. the core function, i.e matrix completion, is defined as matrix_completion(self, matrix, mask, method, param_a, param_b),
       you can check the function to find out the usage
    3. to use the matrix_completion function, you must provide, the matrix you need to be completed, which contains matrix a, b, and relation matrix,
       the mask matrix where corresponding element is 1 if the matrix needed to be completed is known or the value is not 0, else the value is 0;
    4. init_hyperparameter is very important, if you give improper value to hyper_param["Candès and Recht's method"]["mu"], it will cause infinit time comsume.
    """

    def __init__(self, cv_fold=2, test_size=0.2, method_list=["BNNR"],
                 seq_simlarity_list=[], on_metric="mean_roc_auc",
                 fig_dir="", csv_dir="",
                 colors=["deeppink", "orange", "purple"],
                 cmap=None, result_cmp=None):

        self.sss = StratifiedShuffleSplit(n_splits=cv_fold, test_size=test_size, random_state=6)  # 1/cv_fold

        self.method_list = method_list
        self.seq_simlarity_list = seq_simlarity_list

        self.on_metric = on_metric
        self.fig_dir = fig_dir
        self.csv_dir = csv_dir
        self.colors = colors
        self.cmap = cmap
        self.result_cmp = result_cmp

        self.lw = 2

        self.a_sim = None
        self.a_name_list = None
        self.a_num = 0

        self.b_sim = None
        self.b_name_list = None
        self.b_num = 0

        self.relationship_list_true = None
        self.relationship_list_pred = None

        self.relation_matrix_true = None
        self.relation_matrix_pred = None

        self.cv_results = []

        self.hyper_param = None
        self.init_hyperparameter()

    def init_hyperparameter(self):

        hyper_param = {k: {} for k in self.method_list}
        if "BNNR" in self.method_list:
            hyper_param["BNNR"]["alpha"] = [0.05, 0.1, 1, 10]
            hyper_param["BNNR"]["beta"] = [5, 10, 20, 30]

        # tau : singular value thresholding amount;, default to 5 * (m + n) / 2
        # delta : step size per iteration; default to 1.2 times the undersampling ratio
        if "SVT" in self.method_list:
            hyper_param["SVT"]["tau"] = [20, 200, 750, 1000]
            hyper_param["SVT"]["delta"] = [0.4, 0.9, 1.4, 1.8]

        # mu :hyperparameter controlling tradeoff between nuclear norm and square loss
        if "Candès and Recht's method" in self.method_list:
            hyper_param["Candès and Recht's method"]["mu"] = [1, 5, 10, 15]
            hyper_param["Candès and Recht's method"]["null"] = np.logspace(start=-5, stop=8, num=20, base=2)

        self.hyper_param = hyper_param

    def matrix_to_list_final(self, matrix, true_y):

        row_number = self.a_num
        row_name = self.a_name_list

        col_number = self.b_num
        col_name = self.b_name_list

        result_list = pd.DataFrame(columns=['Disease', 'RNA', 'Relation', "True_y"],
                                   index=range(row_number * col_number))

        for row_index in range(row_number):
            for col_index in range(col_number):
                list_index = row_index * col_number + col_index
                result_list.iloc[list_index, 0] = row_name[row_index]
                result_list.iloc[list_index, 1] = col_name[col_index]
                result_list.iloc[list_index, 2] = matrix.iloc[row_index, col_index]
        result_list.iloc[:, 3] = true_y

        return result_list

    def matrix_to_list(self, matrix):

        result_list_np = matrix.values.reshape(-1, )

        return result_list_np

    def list_to_matrix(self, list_array_np):

        row_list = self.a_name_list
        col_list = self.b_name_list

        col_number = self.b_num

        result_matrix = pd.DataFrame(list_array_np.reshape(self.a_num, col_number), columns=col_list, index=row_list)

        return result_matrix

    def matrix_compose(self, relation_matrix):

        temp = pd.concat((self.a_sim, relation_matrix), axis=1)
        temp1 = pd.concat((relation_matrix.transpose(), self.b_sim), axis=1)
        need_complete_matrix = pd.concat((temp, temp1), axis=0).astype(np.float)

        return need_complete_matrix

    def decompose_matrix(self, matrix):

        relation_matrix_1 = matrix.iloc[0: self.a_num, self.a_num:]
        relation_matrix_2 = matrix.iloc[self.a_num:, 0: self.a_num].transpose()
        relation_matrix_pred = (relation_matrix_1 + relation_matrix_2) / 2

        return relation_matrix_pred

    def generate_mask(self, relation_matrix):

        a_mask = pd.DataFrame(np.ones((self.a_num, self.a_num)),
                              index=self.a_name_list,
                              columns=self.a_name_list)

        b_mask = pd.DataFrame(np.ones((self.b_num, self.b_num)),
                              index=self.b_name_list,
                              columns=self.b_name_list)

        temp_mask = pd.concat((a_mask, relation_matrix), axis=1)
        temp1_mask = pd.concat((relation_matrix.transpose(), b_mask), axis=1)
        matrix_mask = pd.concat((temp_mask, temp1_mask), axis=0).astype(np.float)

        return matrix_mask

    def find_optimal_cutoff(self, TPR, FPR, threshold):
        y = TPR - FPR

        Youden_index = np.argmax(y)  # Only the first occurrence is returned.
        optimal_threshold = threshold[Youden_index]
        point = [FPR[Youden_index], TPR[Youden_index]]

        return optimal_threshold, point

    def get_fpr_tpr_youdan(self, y_true, y_score):

        fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)
        optimal_threshold, _ = self.find_optimal_cutoff(tpr, fpr, threshold)
        re_fpr = np.linspace(0, 1, 100)
        tpr = interp(re_fpr, fpr, tpr)
        fpr = re_fpr
        roc_auc = roc_auc_score(y_true, y_score)

        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
        re_recall = np.linspace(0, 1, 100)
        precision = interp(re_recall, recall[::-1], precision[::-1])
        recall = re_recall
        prc_average_precision_score = average_precision_score(y_true, y_score)

        y_pred = (np.array(y_score) >= optimal_threshold).astype(int)
        report_result = classification_report(y_true, y_pred, output_dict=True)

        return fpr, tpr, optimal_threshold, roc_auc, report_result, precision, recall, prc_average_precision_score

    def performance_calculation(self, cv_result):

        fpr_list = []
        tpr_list = []
        threshold_list = []
        roc_auc_list = []

        precision_list = []
        recall_list = []
        prc_average_precision_score_list = []

        precision_0 = []
        precision_1 = []
        recall_0 = []
        recall_1 = []
        f1_score_0 = []
        f1_score_1 = []
        accuracy = []
        macro_precision = []
        macro_recall = []
        macro_f1_score = []
        weighted_precision = []
        weighted_recall = []
        weighted_f1_score = []

        for y_true, y_score in zip(cv_result["true_label"], cv_result["pred_proba"]):
            fpr, tpr, threshold, roc_auc, report_result, precision, recall, prc_average_precision_score = self.get_fpr_tpr_youdan(
                y_true, y_score)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            threshold_list.append(threshold)
            roc_auc_list.append(roc_auc)

            precision_list.append(precision)
            recall_list.append(recall)
            prc_average_precision_score_list.append(prc_average_precision_score)

            precision_0.append(report_result["0"]["precision"])
            precision_1.append(report_result["1"]["precision"])
            recall_0.append(report_result["0"]["recall"])
            recall_1.append(report_result["1"]["recall"])
            f1_score_0.append(report_result["0"]["f1-score"])
            f1_score_1.append(report_result["1"]["f1-score"])
            accuracy.append(report_result["accuracy"])
            macro_precision.append(report_result["macro avg"]["precision"])
            macro_recall.append(report_result["macro avg"]["recall"])
            macro_f1_score.append(report_result["macro avg"]["f1-score"])
            weighted_precision.append(report_result["weighted avg"]["precision"])
            weighted_recall.append(report_result["weighted avg"]["recall"])
            weighted_f1_score.append(report_result["weighted avg"]["f1-score"])

        mean_fpr = np.mean([fpr for fpr in fpr_list], axis=0)
        mean_tpr = np.mean([tpr for tpr in tpr_list], axis=0)
        std_tpr = np.std([tpr for tpr in tpr_list], axis=0)

        mean_precision = np.mean([precision for precision in precision_list], axis=0)
        mean_recall = np.mean([recall for recall in recall_list], axis=0)
        mean_prc_average_precision_score = auc(mean_recall, mean_precision)
        std_precision = np.std([precision for precision in precision_list], axis=0)
        std_average_precision_score = np.std(
            [average_precision_score for average_precision_score in prc_average_precision_score_list])
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)

        metrics_dic = dict(
            mean_fpr=mean_fpr,
            mean_tpr=mean_tpr,
            mean_roc_auc=auc(mean_fpr, mean_tpr.reshape(mean_tpr.shape[0], )),
            std_tpr=std_tpr,
            std_auc=np.std([auc for auc in roc_auc_list]),
            tprs_upper=np.minimum(mean_tpr + std_tpr, 1),
            tprs_lower=np.maximum(mean_tpr - std_tpr, 0),
            threshold_list=threshold_list,

            mean_precision=mean_precision,
            mean_recall=mean_recall,
            mean_prc_average_precision_score=mean_prc_average_precision_score,
            std_precision=std_precision,
            std_average_precision_score=std_average_precision_score,
            precisions_upper=precisions_upper,
            precisions_lower=precisions_lower,

            precision_0_mean=np.mean(precision_0),
            precision_1_mean=np.mean(precision_1),
            recall_0_mean=np.mean(recall_0),
            recall_1_mean=np.mean(recall_1),
            f1_score_0_mean=np.mean(f1_score_0),
            f1_score_1_mean=np.mean(f1_score_1),
            accuracy_mean=np.mean(accuracy),
            macro_precision_mean=np.mean(macro_precision),
            macro_recall_mean=np.mean(macro_recall),
            macro_f1_score_mean=np.mean(macro_f1_score),
            weighted_precision_mean=np.mean(weighted_precision),
            weighted_recall_mean=np.mean(weighted_recall),
            weighted_f1_score_mean=np.mean(weighted_f1_score),
        )

        return metrics_dic

    def matrix_completion(self, matrix, mask, method, param_a, param_b):

        if method == "BNNR":
            completed_matrix, iterations = bnnr(matrix.to_numpy(), mask.to_numpy(), alpha=param_a, beta=param_b)
        elif method == "SVT":
            completed_matrix = svt_solve(matrix.to_numpy(), mask.to_numpy(), algorithm='randomized', tau=param_a,
                                         delta=param_b)
        elif method == "Candès and Recht's method":
            completed_matrix = nuclear_norm_solve(matrix.to_numpy(), mask.to_numpy(), mu=param_a)

        completed_matrix = pd.DataFrame(completed_matrix, index=matrix.index, columns=matrix.columns)

        return completed_matrix

    def train_on_method_hyperparameters(self, a_sim, b_sim, relation_matrix, method, seq_sim_str, param_a=1, param_b=1):

        self.a_sim = a_sim
        self.a_name_list = self.a_sim.index
        self.a_num = len(self.a_name_list)

        self.b_sim = b_sim
        self.b_name_list = self.b_sim.index
        self.b_num = len(self.b_name_list)

        self.relation_matrix_true = relation_matrix
        self.relationship_list_true_np = self.matrix_to_list(relation_matrix)

        cv_result = {}
        cv_result["true_label"] = []
        cv_result["pred_proba"] = []

        print("param_a, param_b:", param_a, param_b)

        for train_index, test_index in self.sss.split(self.relationship_list_true_np, self.relationship_list_true_np):
            self.relationship_list_pred = self.relationship_list_true_np.copy()
            self.relationship_list_pred[test_index] = 0
            self.relation_matrix_pred = self.list_to_matrix(self.relationship_list_pred)

            composed_matrix = self.matrix_compose(self.relation_matrix_pred)
            matrix_mask = self.generate_mask(self.relation_matrix_pred)

            t0 = time.time()
            completed_matrix = self.matrix_completion(composed_matrix, matrix_mask, method, param_a, param_b)
            t1 = time.time()
            print("mentod: " + method, "cost time:", t1 - t0)

            self.relation_matrix_pred = self.decompose_matrix(completed_matrix)
            self.relationship_list_pred_np = self.matrix_to_list(self.relation_matrix_pred)
            true_y = self.relationship_list_true_np.astype(np.int32)[test_index]
            pred_prob = self.relationship_list_pred_np[test_index]

            cv_result["true_label"].append(true_y)
            cv_result["pred_proba"].append(pred_prob)

        plt.figure(figsize=(12, 12))
        plt.imshow(composed_matrix, cmap=self.result_cmp)
        plt.savefig(self.fig_dir + seq_sim_str + method + "composed_matrix.svg")
        composed_matrix.to_csv(self.csv_dir + seq_sim_str + method + r"composed_matrix.csv")
        plt.close()

        plt.figure(figsize=(12, 12))
        plt.imshow(completed_matrix, cmap=self.result_cmp)
        plt.savefig(self.fig_dir + seq_sim_str + method + "completed_matrix.svg")
        completed_matrix.to_csv(self.csv_dir + seq_sim_str + method + r"completed_matrix.csv")
        plt.close()

        return self.performance_calculation(cv_result)

    def grid_parameters(self, a_sim, b_sim, relation_matrix, method, seq_sim_str):

        param_set = list(self.hyper_param[method].values())

        one_grid_result = dict(param_a=[], param_b=[], metric_value={})

        if method == "Candès and Recht's method":
            for param_a in param_set[0]:
                one_grid_result["param_a"].append(param_a)
                one_grid_result["param_b"].append("null")
                metric_dic = self.train_on_method_hyperparameters(a_sim, b_sim, relation_matrix, method, seq_sim_str,
                                                                  param_a)
                for key in metric_dic.keys():
                    if not one_grid_result["metric_value"].get(key, None):
                        one_grid_result["metric_value"][key] = []
                    one_grid_result["metric_value"][key].append(metric_dic[key])

        else:
            for param_a in param_set[0]:
                for param_b in param_set[1]:
                    one_grid_result["param_a"].append(param_a)
                    one_grid_result["param_b"].append(param_b)

                    metric_dic = self.train_on_method_hyperparameters(a_sim, b_sim, relation_matrix, method,
                                                                      seq_sim_str, param_a, param_b)
                    for key in metric_dic.keys():
                        if not one_grid_result["metric_value"].get(key, None):
                            one_grid_result["metric_value"][key] = []
                        one_grid_result["metric_value"][key].append(metric_dic[key])

        max_index = np.argmax(one_grid_result["metric_value"][self.on_metric])
        print("param_a", one_grid_result["param_a"][max_index])
        print("param_b", one_grid_result["param_b"][max_index])
        print("accuracy_mean", one_grid_result["metric_value"]["accuracy_mean"][max_index], max_index)
        print("threshold_list", one_grid_result["metric_value"]["threshold_list"][max_index], max_index)

        return one_grid_result, max_index

    def plot_roc_on_best_paramter(self, mean_roc_auc, std_auc, mean_tpr,
                                  mean_fpr, tprs_lower, tprs_upper,
                                  color='deeppink', label=""):

        plt.plot(mean_fpr, mean_tpr,
                 color=color, lw=self.lw,
                 label=label + ' (auc = {0:0.2f} $\pm$ {1:0.2f})'''.format(mean_roc_auc, std_auc)
                 )

        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)

    def plot_method_on_same_chart(self, i):

        plt.figure(num=10 + i, figsize=(6, 6))

        seq_sim = self.seq_simlarity_list[i]
        seq_sim_str = seq_sim.replace(".csv", "").replace("_similarity", "").replace("snoRNA_", "")

        for j, method in enumerate(self.method_list):
            color = self.colors[j]
            on_grid_result, max_index = self.cv_results[i][j]["on_param_cv"]
            self.plot_roc_on_best_paramter(on_grid_result["metric_value"]["mean_roc_auc"][max_index],
                                           on_grid_result["metric_value"]["std_auc"][max_index],
                                           on_grid_result["metric_value"]["mean_tpr"][max_index],
                                           on_grid_result["metric_value"]["mean_fpr"][max_index],
                                           on_grid_result["metric_value"]["tprs_lower"][max_index],
                                           on_grid_result["metric_value"]["tprs_upper"][max_index],
                                           color,
                                           method + " " + seq_sim_str,
                                           )

        plt.plot([0, 1], [0, 1], 'k--', lw=self.lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("")
        plt.legend(loc="lower right", prop={"size": 9})
        plt.savefig(self.fig_dir + seq_sim_str + "averaged_cv_roc.svg", bbox_inches="tight")

    def plot_method_on_same_chart(self, i):

        plt.figure(num=10 + i, figsize=(6, 6))

        seq_sim = self.seq_simlarity_list[i]
        seq_sim_str = seq_sim.replace(".csv", "").replace("_similarity", "").replace("snoRNA_", "")

        for j, method in enumerate(self.method_list):
            color = self.colors[j]
            on_grid_result, max_index = self.cv_results[i][j]["on_param_cv"]
            self.plot_roc_on_best_paramter(on_grid_result["metric_value"]["mean_roc_auc"][max_index],
                                           on_grid_result["metric_value"]["std_auc"][max_index],
                                           on_grid_result["metric_value"]["mean_tpr"][max_index],
                                           on_grid_result["metric_value"]["mean_fpr"][max_index],
                                           on_grid_result["metric_value"]["tprs_lower"][max_index],
                                           on_grid_result["metric_value"]["tprs_upper"][max_index],
                                           color,
                                           method + " ",
                                           )

        plt.plot([0, 1], [0, 1], 'k--', lw=self.lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("")
        plt.legend(loc="lower right", prop={"size": 9})
        plt.savefig(self.fig_dir + seq_sim_str + "averaged_cv_roc.svg", bbox_inches="tight")

    def plot_pr_on_best_paramter(self, mean_recall, mean_precision, precisions_lower, precisions_upper,
                                 mean_prc_average_precision_score, std_average_precision_score,
                                 color='deeppink', label=""):

        plt.plot(mean_recall, mean_precision,
                 color=color, lw=self.lw,
                 label=label + '(aps = {0:0.2f} $\pm$ {1:0.2f})'''.format(mean_prc_average_precision_score,
                                                                          std_average_precision_score))
        plt.fill_between(mean_recall, precisions_lower, precisions_upper, color=color, alpha=.1)

    def plot_method_on_same_chart_pr(self, i):

        plt.figure(num=20 + i, figsize=(6, 6))

        seq_sim_str = self.seq_simlarity_list[i].replace(".csv", "").replace("_similarity", "").replace("snoRNA_", "")

        for j, method in enumerate(self.method_list):
            color = self.colors[j]
            on_grid_result, max_index = self.cv_results[i][j]["on_param_cv"]
            self.plot_pr_on_best_paramter(on_grid_result["metric_value"]["mean_recall"][max_index],
                                          on_grid_result["metric_value"]["mean_precision"][max_index],
                                          on_grid_result["metric_value"]["precisions_lower"][max_index],
                                          on_grid_result["metric_value"]["precisions_upper"][max_index],
                                          on_grid_result["metric_value"]["mean_prc_average_precision_score"][max_index],
                                          on_grid_result["metric_value"]["std_average_precision_score"][max_index],
                                          color,
                                          method + " ",
                                          )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("")
        plt.legend(loc="lower right", prop={"size": 9})
        plt.savefig(self.fig_dir + seq_sim_str + "averaged_cv_prc.svg", bbox_inches="tight")

        plt.show()

    def save_to_metrics(self, ):

        result_array = []
        for i, seq_sim in enumerate(self.seq_simlarity_list):
            for j, method in enumerate(self.method_list):
                on_grid_result, max_index = self.cv_results[i][j]["on_param_cv"]
                valid_metrics = ["seq_sim", "method"]
                valid_metrics_value = [seq_sim, method]
                for metric in on_grid_result["metric_value"].keys():
                    try:
                        if len(on_grid_result["metric_value"][metric][max_index]) > 1:
                            pass
                    except:
                        valid_metrics.append(metric)
                        valid_metrics_value.append(on_grid_result["metric_value"][metric][max_index])
                if i == 0 and j == 0:
                    result_array.append(valid_metrics)
                    result_array.append(valid_metrics_value)
                else:
                    result_array.append(valid_metrics_value)

        pd.DataFrame(np.array(result_array)).to_csv(self.csv_dir + r"final_matrics.csv")

    def refit(self, a_sim, b_sim, relation_matrix, method, seq_sim_str, param_a=1, param_b=1):
        self.a_sim = a_sim
        self.a_name_list = self.a_sim.index
        self.a_num = len(self.a_name_list)
        self.b_sim = b_sim
        self.b_name_list = self.b_sim.index
        self.b_num = len(self.b_name_list)

        relationship_list_true_np = self.matrix_to_list(relation_matrix)
        composed_matrix = self.matrix_compose(relation_matrix)
        matrix_mask = self.generate_mask(relation_matrix)

        print(composed_matrix.shape, matrix_mask.shape)

        t0 = time.time()
        completed_matrix = self.matrix_completion(composed_matrix, matrix_mask, method, param_a, param_b)
        t1 = time.time()
        print("refit mentod: " + method, "cost time:", t1 - t0)

        relation_matrix_pred = self.decompose_matrix(completed_matrix)
        relationship_list_pred = self.matrix_to_list_final(relation_matrix_pred, relationship_list_true_np)

        plt.figure(figsize=(12, 12))
        plt.imshow(completed_matrix, cmap=self.result_cmp)
        plt.savefig(self.fig_dir + seq_sim_str + method + "completed_matrix_final.svg")

        plt.figure(figsize=(12, 12))
        plt.imshow(composed_matrix, cmap=self.result_cmp)
        plt.savefig(self.fig_dir + seq_sim_str + method + "composed_matrix_final.svg")

        completed_matrix.to_csv(self.csv_dir + seq_sim_str + method + r"_compeleted_matrix_final.csv")
        relationship_list_pred.to_csv(self.csv_dir + seq_sim_str + method + r"_compeleted_list_final.csv")

    def visual_hyperparamters(self, ):

        for cv_results in self.cv_results:
            for dic_result in cv_results:
                print(dic_result['seq_sim'], dic_result['method'])
                seq_sim_str, method_str = dic_result['seq_sim'], dic_result['method']

                if dic_result['method'] == "Candès and Recht's method":
                    auc = dic_result['on_param_cv'][0]['metric_value']["mean_roc_auc"]
                    specificity = dic_result['on_param_cv'][0]['metric_value']["recall_0_mean"]
                    sensitivity = dic_result['on_param_cv'][0]['metric_value']["recall_1_mean"]
                    accuracy = dic_result['on_param_cv'][0]['metric_value']["accuracy_mean"]
                    df_tem = pd.DataFrame(np.array([auc, specificity, sensitivity, accuracy]),
                                          index=["auc", "specificity", "sensitivity", "accuracy"],
                                          columns=list(
                                              map(lambda x: round(x, 2), dic_result['on_param_cv'][0]['param_a'])))
                    fig, ax = plt.subplots(1, 1)

                    sn.heatmap(df_tem, annot=True, cmap=self.cmap, vmin=0.8, vmax=1, ax=ax, cbar=False)
                    plt.savefig(self.fig_dir + seq_sim_str + "_" + method_str + "_metrics_visual.svg",
                                bbox_inches="tight")

                    continue

                print(dic_result['on_param_cv'][0].keys())
                max_index = dic_result['on_param_cv'][1]
                auc = dic_result['on_param_cv'][0]['metric_value']["mean_roc_auc"]
                specificity = dic_result['on_param_cv'][0]['metric_value']["recall_0_mean"]
                sensitivity = dic_result['on_param_cv'][0]['metric_value']["recall_1_mean"]
                accuracy = dic_result['on_param_cv'][0]['metric_value']["accuracy_mean"]

                plot_metrics = [auc, specificity, sensitivity, accuracy]
                metric_str = ["AUC", "Specificity", "Sensitivity", "Accuracy"]

                fig, axs = plt.subplots(2, 2)
                axs = axs.reshape(-1, )
                len_a = len(self.hyper_param["BNNR"]["alpha"])
                len_b = len(self.hyper_param["BNNR"]["beta"])

                for i, plot_metric in enumerate(plot_metrics):
                    df_tem = pd.DataFrame(np.array(plot_metric).reshape(len_a, -1),
                                          index=list(map(lambda x: round(x, 2),
                                                         dic_result['on_param_cv'][0]['param_a'][::len_b])),
                                          columns=list(map(lambda x: round(x, 2),
                                                           dic_result['on_param_cv'][0]['param_b'][:len_b])))

                    sn.heatmap(df_tem, annot=True, cmap=self.cmap, vmin=0.8, vmax=1, ax=axs[i], cbar=False)

                    axs[i].set_title(metric_str[i])
                    if i == 0:
                        axs[i].get_xaxis().set_visible(False)
                    elif i == 1:
                        axs[i].set_axis_off()
                    elif i == 3:
                        axs[i].get_yaxis().set_visible(False)

                plt.savefig(self.fig_dir + seq_sim_str + "_" + method_str + "_metrics_visual.svg", bbox_inches="tight")

    def run(self, relationship_matrix_path, disease_sim_path, seq_sim_dir_path):

        relationship_matrix_df = pd.read_csv(relationship_matrix_path, header=0, index_col=0)

        relationship_matrix_df.columns = relationship_matrix_df.columns.map(int)

        disease_sim_graph_df = pd.read_csv(disease_sim_path, header=0, index_col=0)

        for i, seq_sim in enumerate(self.seq_simlarity_list):
            seq_sim_str = seq_sim.replace(".csv", "").replace("_similarity", "").replace("snoRNA_", "")
            print(seq_sim_str)
            seq_sim_path = os.path.join(seq_sim_dir_path, seq_sim)
            seq_simlarity_df = pd.read_csv(seq_sim_path, header=0, index_col=0)

            seq_simlarity_df.index = seq_simlarity_df.index.map(int)
            seq_simlarity_df.columns = seq_simlarity_df.columns.map(int)

            a_sim = disease_sim_graph_df
            b_sim = seq_simlarity_df
            relation_matrix = relationship_matrix_df

            self.cv_results.append([])

            for j, method in enumerate(self.method_list):
                self.cv_results[i].append([])
                self.cv_results[i][j] = dict(seq_sim=seq_sim_str, method=method, on_param_cv=None)
                print("method!!!" + method)
                self.cv_results[i][j]["on_param_cv"] = self.grid_parameters(a_sim, b_sim, relation_matrix, method,
                                                                            seq_sim_str)

            self.plot_method_on_same_chart(i)
            self.plot_method_on_same_chart_pr(i)

        self.save_to_metrics()
        self.visual_hyperparamters()
        plt.show()

        # refit process
        for i, seq_sim in enumerate(self.seq_simlarity_list):
            seq_sim_str = seq_sim.replace(".csv", "").replace("_similarity", "").replace("snoRNA_", "")
            print(seq_sim_str)
            seq_sim_path = os.path.join(seq_sim_dir_path, seq_sim)
            seq_simlarity_df = pd.read_csv(seq_sim_path, header=0, index_col=0)

            seq_simlarity_df.index = seq_simlarity_df.index.map(int)
            seq_simlarity_df.columns = seq_simlarity_df.columns.map(int)

            a_sim = disease_sim_graph_df
            b_sim = seq_simlarity_df
            relation_matrix = relationship_matrix_df

            for j, method in enumerate(self.method_list):
                # if method == "svt":
                # continue
                on_grid_result, max_index = self.cv_results[i][j]["on_param_cv"]
                param_a = on_grid_result["param_a"][max_index]
                param_b = on_grid_result["param_b"][max_index]
                print("refit phase")
                self.refit(a_sim, b_sim, relation_matrix, method, seq_sim_str, param_a, param_b)

