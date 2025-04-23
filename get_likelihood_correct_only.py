from likelihood import *
from data.utils.verif_utils import *
import pandas as pd
import matplotlib.pyplot as plt
import os
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
    # Target positive rate
    target_pr = sum(y_train)/len(y_train)

    model_dir = './models/%s/'%dataset
    model_files = os.listdir(model_dir)
    results = {item.split('.')[0]: {} for item in model_files}
    for model_file in model_files:
        print('==================  STARTING MODEL ' + model_file)
        behavioral_diff = 0
        model_name = model_file.split('.')[0]
        fig, axs = plt.subplots(int(np.ceil(len(sens_choices)/2)), 2, figsize=(5*2, 5*(np.ceil(len(sens_choices))/2)))
        if int(np.ceil(len(sens_choices)/2)) == 1:  
            axs = axs.reshape(1, -1)

        # Prediction on the model (positive softmax, negative softmax, predictions)
        model = load_model(model_dir + model_file, compile=False)
        softmax_pos = model.predict(X_test.reshape(len(X_test), 1, -1)).flatten()
        softmax_neg = 1-softmax_pos
        predictions = np.array([1 if s>=0.5 else 0 for s in softmax_pos])
        agree_index_whole = []
        for sens in sens_choices:
            subgroup = 's=%d'%sens
            row = sens//2
            col = sens%2
            for pred in ['neg', 'pos']:
                hist_path = './hist_subgroups_correct_only/%s/%s/%s/%s_%s_hist.csv'%(dataset, sens_att, subgroup, model_name, pred)
                contribs_path = './contribs/test/%s/%s_contribs.csv'%(dataset, model_name)
                output_path = './likelihood_subgroups_correct_only/%s/%s/%s/%s_%s_likelihood.csv'%(dataset, sens_att, subgroup, model_name, pred)
                
                if not os.path.exists(output_path):
                    compute_single_class_likelihood(contribs_path, output_path, hist_path)
                
            # Get the data
            pos_cpd = pd.read_csv('./likelihood_subgroups_correct_only/%s/%s/%s/%s_pos_likelihood.csv'%(dataset, sens_att, subgroup, model_name))['Score']
            neg_cpd = pd.read_csv('./likelihood_subgroups_correct_only/%s/%s/%s/%s_neg_likelihood.csv'%(dataset, sens_att, subgroup, model_name))['Score']


            # Agreement in the sensitive set
            agree_index = [i for i in range(len(predictions)) if (sens_list[i] == sens and ((predictions[i] == 1 and neg_cpd[i]>=pos_cpd[i]) or (predictions[i] == 0 and neg_cpd[i]<pos_cpd[i])))]
            agree_index_whole += agree_index 

            # Filter the predictions and scores according to sensitive subgroup
            softmax_pos_sub = [s for i, s in enumerate(softmax_pos) if i in agree_index]
            softmax_neg_sub = [s for i, s in enumerate(softmax_neg) if i in agree_index]
            pred_sub = [p for i, p in enumerate(predictions) if i in agree_index]

            # Filter the relative cpds according to sensitive subgroup
            rel_pos_sub = [np.arctan(n/p) for i, (p, n) in enumerate(zip(pos_cpd, neg_cpd)) if i in agree_index]
            rel_neg_sub = [np.arctan(p/n) for i, (p, n) in enumerate(zip(pos_cpd, neg_cpd)) if i in agree_index]

            # Whether we should increase or decrease the positive rate
            pr = sum(pred_sub)/len(pred_sub)
            if pr > target_pr:
                target = 'Positive'
                direction = 'Pos to Neg'
                pop_delta = int((pr - target_pr) * len(pred_sub))
                
                # Select individuals from positive class
                filter_softmax= [s for s in softmax_pos_sub if s >= 0.5]
                filter_rel = [s for i, s in enumerate(rel_pos_sub) if pred_sub[i] == 1]

                # Select individuals using softmax
                softmax_corr_idx = sorted(range(len(filter_softmax)), key=lambda i: filter_softmax[i])[:pop_delta]
                
                # Suggest individuals using CPD
                cpd_corr_idx = sorted(range(len(filter_rel)), key=lambda i: filter_rel[i])[:pop_delta]

            else:
                target = 'Negative'
                direction = 'Neg to Pos'
                pop_delta = int((target_pr - pr) * len(pred_sub))
                
                # Select individuals from negative class
                filter_softmax = [s for s in softmax_neg_sub if s >= 0.5]
                filter_rel = [s for i, s in enumerate(rel_neg_sub) if pred_sub[i] == 0]

                # Select individuals using softmax
                softmax_corr_idx = sorted(range(len(filter_softmax)), key=lambda i: filter_softmax[i])[:pop_delta]

                # Suggest individuals using CPD
                cpd_corr_idx = sorted(range(len(filter_rel)), key=lambda i: filter_rel[i])[:pop_delta]


            agree_idx = [i for i in softmax_corr_idx if i in cpd_corr_idx]
            not_selected_idx = [i for i in range(len(filter_softmax)) if (i not in softmax_corr_idx and i not in cpd_corr_idx)]
            softmax_disagree_idx = [i for i in softmax_corr_idx if i not in agree_idx]
            cpd_disagree_idx = [i for i in cpd_corr_idx if i not in agree_idx]
            
            if pop_delta != 0:
                results[model_name]['subgroup=%d'%sens] = {'Direction': direction,
                                            'Number of Correction': pop_delta,
                                            'Correction Agreement Rate': round(len(agree_idx)/pop_delta, 3)}
            else: 
                results[model_name]['subgroup=%d'%sens] = {'Direction': direction,
                                            'Number of Correction': pop_delta,
                                            'Correction Agreement Rate': None}
            

            # Get the points
            cpd_selection = [tuple([filter_rel[i], filter_softmax[i]]) for i in cpd_corr_idx]
            softmax_selection = [tuple([filter_rel[i], filter_softmax[i]]) for i in softmax_corr_idx]
            agree = [tuple([filter_rel[i], filter_softmax[i]]) for i in agree_idx]
            softmax_disagree = [tuple([filter_rel[i], filter_softmax[i]]) for i in softmax_disagree_idx]
            cpd_disagree = [tuple([filter_rel[i], filter_softmax[i]]) for i in cpd_disagree_idx]
            not_selected = [tuple([filter_rel[i], filter_softmax[i]]) for i in not_selected_idx]
            
            assert(len(softmax_disagree) == len(cpd_disagree))
            assert(len(agree) + len(softmax_disagree) == len(softmax_corr_idx))

            # Visualization
            if len(agree) != 0:
                axs[row, col].scatter(np.array(agree)[:, 0], np.array(agree)[:, 1], alpha=0.3, label='Agreed Correction', s=5)
                # Mark the boundary
                axs[row, col].axhline(y=max(np.array(softmax_selection)[:, 1]), color='r', linestyle='--', alpha=0.3)
                axs[row, col].axvline(x=max(np.array(cpd_selection)[:, 0]), color='r', linestyle='--', alpha=0.3)
            if len(softmax_disagree) != 0:
                axs[row, col].scatter(np.array(softmax_disagree)[:, 0], np.array(softmax_disagree)[:, 1], alpha=0.3, label='Diff Softmax Correction', s=5)
            if len(cpd_disagree) != 0:
                axs[row, col].scatter(np.array(cpd_disagree)[:, 0], np.array(cpd_disagree)[:, 1], alpha=0.3, label='Diff CPD Suggestion', s=5)
            if len(not_selected) != 0:
                axs[row, col].scatter(np.array(not_selected)[:, 0], np.array(not_selected)[:, 1], alpha=0.3, label='Other Samples', s=5)

            # Adding text annotations at the specified corners
            axs[row, col].text(0.02, 0.02, len(agree), fontsize=10, color='black', ha='left', va='bottom', transform=axs[row, col].transAxes)
            axs[row, col].text(0.98, 0.02, len(softmax_disagree), fontsize=10, color='black', ha='right', va='bottom', transform=axs[row, col].transAxes)
            axs[row, col].text(0.02, 0.98, len(cpd_disagree), fontsize=10, color='black', ha='left', va='top', transform=axs[row, col].transAxes)
            axs[row, col].text(0.98, 0.98, len(not_selected), fontsize=10, color='black', ha='right', va='top', transform=axs[row, col].transAxes)
            axs[row, col].set_xlabel('Relative CPD in %s Class'%target)
            axs[row, col].set_ylabel('Softmax in %s Class'%target)
            axs[row, col].set_ylim(0.495, 1.005)
            axs[row, col].set_xlim(-0.005, np.pi/2+0.005)
            axs[row, col].set_title('Bias Correction for s=%d (Originally %s)'%(sens, target))
            axs[row, col].legend()
        if len(sens_choices) % 2 == 1:
            axs[-1, -1].axis('off')

        
        results[model_name]['Agreed Prediction Rate'] = round(len(agree_index_whole)/len(predictions), 3)
        results[model_name]['Original Prediction Accuracy'] = round(accuracy_score(y_test, predictions), 3)
        results[model_name]['Agreed Prediction Accuracy'] = round(accuracy_score(y_test[agree_index_whole], predictions[agree_index_whole]), 3)
        plt.tight_layout()
        plt.savefig('./visualizations/bias_correction/%s/%s/%s_plot.png'%(dataset, sens_att, model_name))
        print('Visualization for model %s on sensitive attribute %s saved!'%(model_name, sens_att))
    with open('./results/%s_%s_correction_results.json'%(dataset, sens_att), 'w') as f:
        json.dump(results, f, indent=4)
    print('Correction evaluation result for dataset %s on sensitive attribute %s saved!'%(dataset, sens_att))
