# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This script loads the results of baselines and generates latex tables
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

# this script loads selected baseline results (from their .json files) and generates graphs and tables
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, default='./log/surface_reconstruction/results/results_for_paper/ablations/',
                    help='path to top level directory containing different baseline results json files')
parser.add_argument('--baselines', nargs='+', default=['DGP', 'IGR', 'SIREN', 'DiGS_w_n', 'SIREN_wo_n', 'IGR_wo_n',
                                                       'DiGS', ], help='list of baseline names for parsing')
parser.add_argument('--modes', nargs='+', default=['GT', 'scans'], help='list of mode names for parsing')
parser.add_argument('--shapes', nargs='+', default=['anchor', 'daratech', 'dc', 'gargoyle', 'lord_quas'], help='list of models to compare')
parser.add_argument('--output_path', type=str, default='./log//surface_reconstruction/results/summary/',
                    help='path to where to output the results summary files including tables and figures')
parser.add_argument('--output_filename', type=str,  default='surface_reconstruction_ablations_results.txt',
                    help='name of output file')
args = parser.parse_args()

percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95] # This must match the precentiles used in testing

os.makedirs(args.output_path, exist_ok=True)
output_table_file_name = os.path.join(args.output_path, args.output_filename)
output_avg_table_file_name = os.path.join(args.output_path, args.output_filename.split('.')[0] + '_avg.txt')
result_file_list = [f for f in os.listdir(args.results_path) if os.path.isfile(os.path.join(args.results_path, f)) and f.endswith('.json')]

# load all of the results data into a single dictionary
alphas = []
agg_results_dict = {mode: {shape: {baseline: [] for baseline in args.baselines} for shape in args.shapes} for mode in args.modes}
for baseline in args.baselines:
    for mode in args.modes:
        results_filename = ''
        for filename in result_file_list:
            if (mode in filename) and (baseline + "_Berger" in filename):
                results_filename = os.path.join(args.results_path, filename)
        if results_filename == '':
                raise Warning("no results file for baseline {} for data {}".format(baseline, mode))


        with open(results_filename) as json_file:
            results_dict = json.load(json_file)


        for shape in agg_results_dict[mode].keys():
            agg_results_dict[mode][shape][baseline] = results_dict[shape]
            if mode == 'GT':
                alphas.append(results_dict['pod_alphas'])

# Generate latex table
with open(output_table_file_name, 'w') as file:
    for shape in args.shapes:
        flag = 0
        for baseline in args.baselines:
            result_str = ''
            if flag == 0:
                result_str = "\multirow{" + str(len(args.baselines))+ "}{*}{" + shape.replace("_", " ") + "} "
                flag = 1
            dc = str(round(float(agg_results_dict['GT'][shape][baseline]['dc']), 2))
            dh = str(round(float(agg_results_dict['GT'][shape][baseline]['dh']), 2))
            dc1 = str(round(float(agg_results_dict['scans'][shape][baseline]['one_sided_dc']), 2))
            dh1 = str(round(float(agg_results_dict['scans'][shape][baseline]['one_sided_dh']), 2))
            malcv_10 = str(round(float(100*agg_results_dict['GT'][shape][baseline]['malcv'][0]), 2))
            malcv_50 = str(round(float(100*agg_results_dict['GT'][shape][baseline]['malcv'][-1]), 2))
            result_str = result_str + " & {} & {} & {} & {} & {} & {} & {} " \
                                      "\\\\ \n".format(baseline.replace("_", " "), dc, dh, dc1, dh1, malcv_10, malcv_50)
            file.write(result_str)
        file.write("\hline\n")

# Generate latex average results table
with open(output_avg_table_file_name, 'w') as avg_file:
    for baseline in args.baselines:
        flag = 0
        dc, dh, dc1, dh1, mlcv_10, mlcv_50 = [], [], [], [], [], []
        for shape in args.shapes:
            result_str = ''
            if flag == 0:
                result_str = "\multirow{" + str(len(args.baselines))+ "}{*}{" + shape.replace("_", " ") + "} "
                flag = 1
            dc.append(agg_results_dict['GT'][shape][baseline]['dc'])
            dh.append(agg_results_dict['GT'][shape][baseline]['dh'])
            dc1.append(agg_results_dict['scans'][shape][baseline]['one_sided_dc'])
            dh1.append(agg_results_dict['scans'][shape][baseline]['one_sided_dh'])
            mlcv_10.append(agg_results_dict['GT'][shape][baseline]['malcv'][0])
            mlcv_50.append(agg_results_dict['GT'][shape][baseline]['malcv'][-1])

        dc = str(round(float(np.array(dc).mean()), 2))
        dh = str(round(float(np.array(dh).mean()), 2))
        dc1 = str(round(float(np.array(dc1).mean()), 2))
        dh1 = str(round(float(np.array(dh1).mean()), 2))

        malcv_10 = str(round(float(100*np.array(mlcv_10).mean()), 2))
        malcv_50 = str(round(float(100*np.array(mlcv_50).mean()), 2))
        result_str = result_str + "{} & {} & {} & {} & {} & {} & {} " \
                                  "\\\\ \n".format(baseline.replace("_", " "), dc, dh, dc1, dh1, malcv_10, malcv_50)
        avg_file.write(result_str)
    avg_file.write("\hline\n")



