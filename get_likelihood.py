from likelihood import *
from data.utils.verif_utils import *
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy
from fairlearn.metrics import *
import json

# Iterate through all the datasets and models to calculate likelihood
for dataset in ['adult', 'bank', 'german']:
    if dataset == 'adult': 
        df, X_train, y_train, X_test, y_test = load_adult_ac1()
        sens_att = 'sex'
    elif dataset == 'bank':
        df, X_train, y_train, X_test, y_test = load_bank()
        sens_att = 'age'
    else:
        df, X_train, y_train, X_test, y_test = load_german()
        sens_att = 'age'
    sens_idx = list(df.columns).index(sens_att)
    sens_list = X_test[:, sens_idx]
    sens_choices = np.unique(sens_list).astype(int)
    model_dir = './models/%s/'%dataset
    model_files = os.listdir(model_dir)

    # Results
    results = {}    
    for model_file in model_files:
        print('==================  STARTING MODEL ' + model_file)
        model = load_model(model_dir + model_file, compile=False)
        softmax_pos = model.predict(X_test.reshape(len(X_test), 1, -1)).flatten()
        predictions = [1 if s>=0.5 else 0 for s in softmax_pos]
        behavioral_diff = 0
        model_name = model_file.split('.')[0]
        fig, axs = plt.subplots(int((len(sens_choices)+1)/3), 3, figsize=(5*3, 5*(int(len(sens_choices)+1)/3)))
        if len(axs.shape) == 1:
            axs = axs.reshape(1, -1)
        # The entropy of different subgroups
        pos_entropy = {item:None for item in ['hist_full']+['hist_s=%d'%j for j in sens_choices]}
        neg_entropy = {item:None for item in ['hist_full']+['hist_s=%d'%j for j in sens_choices]}

        # Strutcture- {histogram:subgroup:CPD_list}
        sens_pos, sens_neg = {item:{} for item in ['hist_full']+['hist_s=%d'%j for j in sens_choices]}, {item:{} for item in ['hist_full']+['hist_s=%d'%j for j in sens_choices]}

        # Strutcture- {histogram:subgroup:population}
        population_pos, population_neg = {item:{} for item in ['hist_full']+['hist_s=%d'%j for j in sens_choices]}, {item:{} for item in ['hist_full']+['hist_s=%d'%j for j in sens_choices]}
        decisions = []
        for i, subgroup in enumerate(['full']+['s=%d'%j for j in sens_choices]):
            row = i//3
            col = i%3
            hist_subgroup = 'hist_'+subgroup
            for pred in ['neg', 'pos']:
                hist_path = './hist_subgroups/%s/%s/%s/%s_%s_hist.csv'%(dataset, sens_att, subgroup, model_name, pred)
                contribs_path = './contribs/test/%s/%s_contribs.csv'%(dataset, model_name)
                output_path = './likelihood_subgroups/%s/%s/%s/%s_%s_likelihood.csv'%(dataset, sens_att, subgroup, model_name, pred)
                
                if not os.path.exists(output_path):
                    compute_single_class_likelihood(contribs_path, output_path, hist_path)
                
            # Get the data
            pos_cpd = pd.read_csv('./likelihood_subgroups/%s/%s/%s/%s_pos_likelihood.csv'%(dataset, sens_att, subgroup, model_name))['Score']
            neg_cpd = pd.read_csv('./likelihood_subgroups/%s/%s/%s/%s_neg_likelihood.csv'%(dataset, sens_att, subgroup, model_name))['Score']

            for idx in range(len(pos_cpd)):
                if idx == len(decisions):
                    if pos_cpd[idx] > neg_cpd[idx]: decisions.append([0])
                    else: decisions.append([1])
                else:
                    if pos_cpd[idx] > neg_cpd[idx]: decisions[idx].append(0)
                    else: decisions[idx].append(1)

            # Split using sensitive attributes
            sens_pos[hist_subgroup] = {'s=%d'%j: [pos_cpd[i] for i in range(len(pos_cpd)) if sens_list[i] == j] for j in sens_choices}
            sens_neg[hist_subgroup] = {'s=%d'%j: [neg_cpd[i] for i in range(len(neg_cpd)) if sens_list[i] == j] for j in sens_choices}

            population_pos[hist_subgroup] = {'s=%d'%j: len([i for i in range(len(sens_pos[hist_subgroup]['s=%d'%j])) if sens_pos[hist_subgroup]['s=%d'%j][i]<sens_neg[hist_subgroup]['s=%d'%j][i]]) for j in sens_choices}
            population_neg[hist_subgroup] = {'s=%d'%j: len([i for i in range(len(sens_pos[hist_subgroup]['s=%d'%j])) if sens_pos[hist_subgroup]['s=%d'%j][i]>sens_neg[hist_subgroup]['s=%d'%j][i]]) for j in sens_choices}
            max_value = max(max(pos_cpd), max(neg_cpd))

            # Plot the data
            axs[row, col].set_title('Histogram Constructed with Subgroup: %s'%subgroup)
            pos_entropy['hist_full'] = entropy([population_pos['hist_full']['s=%d'%k]/(population_pos['hist_full']['s=%d'%k]+population_neg['hist_full']['s=%d'%k]) for k in sens_choices])
            pos_entropy[hist_subgroup] = entropy([population_pos[hist_subgroup]['s=%d'%k]/(population_pos[hist_subgroup]['s=%d'%k]+population_neg[hist_subgroup]['s=%d'%k]) for k in sens_choices])

            neg_entropy['hist_full'] = entropy([population_neg['hist_full']['s=%d'%k]/(population_pos['hist_full']['s=%d'%k]+population_neg['hist_full']['s=%d'%k]) for k in sens_choices])
            neg_entropy[hist_subgroup] = entropy([population_neg[hist_subgroup]['s=%d'%k]/(population_pos[hist_subgroup]['s=%d'%k]+population_neg[hist_subgroup]['s=%d'%k]) for k in sens_choices])
            
            for j in sens_choices:
                behavioral_diff += abs(pos_entropy[hist_subgroup]-pos_entropy['hist_full'])
                behavioral_diff += abs(neg_entropy[hist_subgroup]-neg_entropy['hist_full'])
                axs[row, col].scatter(sens_pos[hist_subgroup]['s=%d'%j], sens_neg[hist_subgroup]['s=%d'%j], alpha=0.1, label='s=%d'%j, s=8)
            axs[row, col].plot(range(int(np.ceil(max_value))+1), range(int(np.ceil(max_value))+1), color='black', alpha=0.1)
            axs[row, col].set_xlim(0, int(np.ceil(max_value))+1)
            axs[row, col].set_ylim(0, int(np.ceil(max_value))+1)

            # Texts
            pos_text = '\n'.join(['Closer to Positive']+['s=%d: %d'%(j, population_pos[hist_subgroup]['s=%d'%j]) for j in sens_choices])
            neg_text = '\n'.join(['Closer to Negative']+['s=%d: %d'%(j, population_neg[hist_subgroup]['s=%d'%j]) for j in sens_choices])
            axs[row, col].text(0.02, 0.98, pos_text, transform=axs[row, col].transAxes, 
                verticalalignment='top', horizontalalignment='left', fontsize=8)
            axs[row, col].text(0.98, 0.02, neg_text, transform=axs[row, col].transAxes, 
                verticalalignment='bottom', horizontalalignment='right', fontsize=8)
            axs[row, col].set_xlabel('Distance to Positive (Y=1)')
            axs[row, col].set_ylabel('Distance to Negative (Y=0)')
            axs[row, col].legend()
        decisions = [1 if (sum(item) != len(sens_choices)+1 and sum(item) != 0) else 0 for item in decisions]
        plt.tight_layout()
        plt.savefig('./visualizations/model_fairness/%s/%s/%s_plot.png'%(dataset, sens_att, model_name))
        if len(sens_choices) == 2:
            results[model_name] = {'Accuracy': round(accuracy_score(y_test, predictions), 3),
                                'Demographic Parity': round(demographic_parity_difference(y_test, predictions, sensitive_features=sens_list), 3),
                                'Equalized Odds': round(equalized_odds_difference(y_test, predictions, sensitive_features=sens_list), 3),
                                "Group level discrepancy": round(behavioral_diff, 3), 
                                "Individual level discrepancy": round(sum(decisions)/len(decisions), 3)}
        else:
            results[model_name] = {'Accuracy': round(accuracy_score(y_test, predictions), 3),
                                'Demographic Parity': None,
                                'Equalized Odds': None,
                                "Group level discrepancy": round(behavioral_diff, 3), 
                                "Individual level discrepancy": round(sum(decisions)/len(decisions), 3)}
        
        print('Visualization for model %s on sensitive attribute %s saved!'%(model_name, sens_att))
    # Store the prediction
    with open('./results/%s_%s_results.json'%(dataset, sens_att), 'w') as f:
        json.dump(results, f, indent=4)
    print('Fairness estimation result for dataset %s on sensitive attribute %s saved!'%(dataset, sens_att))
    exit()
