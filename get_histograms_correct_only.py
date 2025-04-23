from data.utils.verif_utils import *
import os
import numpy as np
from fairlearn.metrics import *
from tensorflow import keras
from scripts.config.parameters import EPSILON_PREC
import pandas as pd
import scripts.config.parameters as p
from statistics import pstdev
from config.logger import create_logger

logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)

def construct_hist(layer_id: int,
                   node_id: int,
                   act_levels: list,):
    """
    :param layer_id: Which layer of the model to consider (usually the one before the last)
    :param node_id: Which neuron of the layer to consider
    :param act_levels: Activation levels for this specific neuron
    :return: The histogram of a given neuron of a model
    """
    nbr_act_levels = len(act_levels)

    if nbr_act_levels <= 0 or '' in act_levels:
        raise ValueError(
            f'ERROR construct_hist: Activation levels not found for node {node_id} at layer {layer_id}')

    # Counting in the string the number of decimal digits. If higher than 10, raise error, if equal to -1, '.' is not found
    contrib_prec = [0 <= str(elem)[::-1].find('.') <= p.EPSILON_PREC for elem in act_levels]
    if not all(contrib_prec):
        false_indexes = [i for i, elem in enumerate(contrib_prec) if elem is False]
        raise ValueError(
            f'ERROR construct_hist: contribs file contains act levels with a precision higher than {p.EPSILON}, or contribs that are not float (dot character not found). At line {false_indexes}')

    try:
        act_levels = [float(elem) for elem in act_levels]
    except ValueError:
        raise ValueError(f'ERROR construct_hist: Impossible to convert act levels to float, please check contribs file')

    sigma = pstdev(act_levels)
    sigma = round(sigma, p.EPSILON_PREC) if sigma >= p.EPSILON else 0.0
    max_ = max(act_levels)
    min_ = min(act_levels)
    logger.debug(f'Standard deviation is {sigma:.10f}')

    # Standard dev is neg (impossible)
    if sigma < -p.EPSILON:
        raise ValueError(
            f'ERROR construct_hist: Invalid distribution for node {node_id} of layer {layer_id}. The standard deviation is negative : {sigma}'
        )
    # We append file since the header is already written
    hist_dict = {header:[] for header in ['layerId', 'nodeId', 'binId', 'sigmaInterval_lb', 'sigmaInterval_ub', 'sigmaFreq']}
    # Standard dev is null (no variance)
    if -p.EPSILON <= sigma < p.EPSILON:
        str_single_value = f"{min_:.10f}"
        logger.warning(f'Distribution for node {node_id} of layer {layer_id} with null variance')
        logger.debug(
            f'Unique binId 0 of bounds ({str_single_value}, {str_single_value}), and frequency {nbr_act_levels}\n')
        hist_dict['layerId'].append(layer_id)
        hist_dict['nodeId'].append(node_id)
        hist_dict['binId'].append(0)
        hist_dict['sigmaInterval_lb'].append(str_single_value)
        hist_dict['sigmaInterval_ub'].append(str_single_value)
        hist_dict['sigmaFreq'].append(nbr_act_levels)

    # Standard dev is pos (most of the case)
    else:
        # Step is sigma - LSP, to avoid that contributions fall on bin edges
        hist, bins = np.histogram(act_levels,
                                    bins=np.arange(start=min_, stop=max_ + 2 * sigma,
                                                    step=sigma - p.LOW_SMOOTHED_PROB))

        logger.debug(f'Min is {min_:.10f} et max is {max_:.10f}, step {sigma:.10f}')

        if not np.isclose(bins, np.round(bins, p.EPSILON_PREC), atol=p.EPSILON_PREC).all():
            raise ValueError(f'ERROR construct_hist: bins cannot be rounded to precision')

        bins = np.round(bins, p.EPSILON_PREC)

        step = bins[1] - bins[0]
        logger.debug(f'The step is {step:.10f}')

        if len(bins) < 2:
            raise ValueError(f'ERROR construct_hist: too few bins for node {node_id} of layer {layer_id}')
        prev_bin = bins[0]
        if step < p.EPSILON:
            raise ValueError(f'ERROR construct_hist: step is null for node {node_id} of layer {layer_id}')

        nbr_bins = len(bins[1:])
        logger.debug(f'There are {nbr_bins} bins.')
        if nbr_bins > p.POS_UNDEF_INT:
            raise ValueError(
                f'ERROR construct_hist: There are {nbr_bins}, which is too high for the current precision. '
                f'Hist file would be incorrect')
        # For verification purpose
        count = 0
        for i, b in enumerate(bins[1:]):
            logger.debug(
                f'BinId {i} ({prev_bin:.10f}, {b:.10f}) has a frequency of {hist[i]}' + (
                    '\n' if (i == nbr_bins - 1) else ''))

            if hist[i] != 0:
                count += hist[i]
                hist_dict['layerId'].append(layer_id)
                hist_dict['nodeId'].append(node_id)
                hist_dict['binId'].append(i)
                hist_dict['sigmaInterval_lb'].append(f'{prev_bin:.10f}')
                hist_dict['sigmaInterval_ub'].append(f'{b:.10f}')
                hist_dict['sigmaFreq'].append(hist[i])
            prev_bin = b

        if count != nbr_act_levels:
            raise ValueError(f'ERROR construct_hist: There are {nbr_act_levels} inputs, but hist contains {count}')
    
    return hist_dict

for dataset in ['adult']:
    if dataset == 'adult': 
        df, X_train, y_train, X_test, y_test = load_adult_ac1()
        sens_att = 'race'
    elif dataset == 'bank':
        df, X_train, y_train, X_test, y_test = load_bank()
        sens_att = 'age'
    else:
        df, X_train, y_train, X_test, y_test = load_german()
        sens_att = 'age'

    sens_idx = list(df.columns).index(sens_att)
    model_dir = './models/%s/'%dataset
    model_files = os.listdir(model_dir)
    sens_list = df[sens_att]
    sens_choices = np.unique(sens_list).astype(int)

    for model_file in model_files:
        contrib_dict_test = {'inputId': [], 'layerId': [], 'nodeId': [], 'nodeContrib': []}
        contrib_dict_train = {'inputId': [], 'layerId': [], 'nodeId': [], 'nodeContrib': []}
        subgroup_idx = {'s=%d'%i:[] for i in sens_choices}
        if not model_file.endswith('.h5'):
            continue;
        print('==================  STARTING MODEL ' + model_file)
        model_name = model_file.split('.')[0]
        if model_name == '':
            continue
        model = load_model(model_dir + model_file, compile=False)
        # Manually define an input tensor to fix the missing input issue, and get the last layer output
        outputs = model(model.inputs)
        layer_outputs = [layer.output for layer in model.layers]

        # Create a new model that outputs intermediate activations
        model_with_outputs = keras.Model(inputs=model.input, outputs=layer_outputs)

        # 1. Get the contributions for the testing set
        activation_levels = model_with_outputs.predict(X_test.reshape(len(X_test), 1, -1))
        predictions = activation_levels[-1].squeeze()
        activation_levels = activation_levels[-2].squeeze()
        activation_levels = np.round(activation_levels, decimals=EPSILON_PREC)

        # Store the activation levels
        for i, activation_level in enumerate(activation_levels): 
            for j, nodeContrib in enumerate(activation_level):
                contrib_dict_test['inputId'].append(i)
                contrib_dict_test['layerId'].append(-1)
                contrib_dict_test['nodeId'].append(j)
                contrib_dict_test['nodeContrib'].append(nodeContrib)
        
        # Output the contribution file
        pd.DataFrame(contrib_dict_test).to_csv('./contribs/test/%s/%s_contribs.csv'%(dataset, model_name), index=None)

        # 2. Get the contributions for the training set
        activation_levels = model_with_outputs.predict(X_train.reshape(len(X_train), 1, -1))
        # Correctly predicted index
        pos_idx, neg_idx = [], []

        predictions = activation_levels[-1].squeeze()
        # Get the negative and positive class prediction indices, and the sensitive attribute indices
        for i, pred in enumerate(predictions):
            if pred >= 0.5 and y_train[i] == 1: 
                pos_idx.append(i)
            elif pred < 0.5 and y_train[i] == 0: 
                neg_idx.append(i)
            subgroup_idx['s=%d'%X_train[i, sens_idx]].append(i)

        activation_levels = activation_levels[-2].squeeze()
        activation_levels = np.round(activation_levels, decimals=EPSILON_PREC)

        # Store the activation levels
        for i, activation_level in enumerate(activation_levels): 
            for j, nodeContrib in enumerate(activation_level):
                contrib_dict_train['inputId'].append(i)
                contrib_dict_train['layerId'].append(-1)
                contrib_dict_train['nodeId'].append(j)
                contrib_dict_train['nodeContrib'].append(nodeContrib)
        
        # Output the contribution file
        pd.DataFrame(contrib_dict_train).to_csv('./contribs/train/%s/%s_contribs.csv'%(dataset, model_name), index=None)

        # 3. Get the histogram for pos and neg class using the training set data
        for label in ['pos', 'neg']:
            # Slice the dataset into different subgroups
            for subgroup in ['s=%d'%j for j in sens_choices]:
                hist_dict = {header:[] for header in ['layerId', 'nodeId', 'binId', 'sigmaInterval_lb', 'sigmaInterval_ub', 'sigmaFreq']}
                if not os.path.exists('./hist_subgroups_correct_only/%s/%s/%s/%s_%s_hist.csv'%(dataset, sens_att, subgroup, model_name, label)):
                    for node_id in np.unique(contrib_dict_train['nodeId']):
                        # Filter contrib of given node
                        if label == 'neg':
                            contribs_node = [contrib for i, contrib in enumerate(contrib_dict_train['nodeContrib']) if 
                                             (contrib_dict_train['nodeId'][i] == node_id and contrib_dict_train['inputId'][i] in neg_idx and contrib_dict_train['inputId'][i] in subgroup_idx[subgroup])]
                        else:
                            contribs_node = [contrib for i, contrib in enumerate(contrib_dict_train['nodeContrib']) if 
                                             (contrib_dict_train['nodeId'][i] == node_id and contrib_dict_train['inputId'][i] in pos_idx and contrib_dict_train['inputId'][i] in subgroup_idx[subgroup])]
                        node_hist_dict = construct_hist(layer_id=-1,
                                    node_id=node_id,
                                    act_levels=contribs_node)
                        # Append the hist
                        for key in hist_dict.keys():
                            hist_dict[key].extend(node_hist_dict[key])
                    print('Histogram constructed on subgroup: %s for model: %s'%(subgroup, model_name))
                    pd.DataFrame(hist_dict).to_csv('./hist_subgroups_correct_only/%s/%s/%s/%s_%s_hist.csv'%(dataset, sens_att, subgroup, model_name, label), index=None)

                else:
                    print('Histogram on subgroup: %s for model: %s exists!'%(subgroup, model_name))
