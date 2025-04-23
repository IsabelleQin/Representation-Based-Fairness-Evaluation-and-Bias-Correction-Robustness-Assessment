from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append('../sc')
from config import parameters as p
from config.logger import create_logger

# ATTR_INFO = tuple[str, pd.Series] | str
logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)


def filter_indexes(indexes_array: list[list[str]],
                   set_name: str | None = None,
                   correct: str | bool | None = None,
                   output_class: str | list[str] | None = None,
                   sensitive_class: str | list[str] | None = None) -> list[int]:
    """
    :param indexes_array: Unfiltered indexes to read
    :param set_name: None, '', 'train', 'test', 'valid'. The name of the set to filter
    :param correct: None, '', 'true',True, 'false', False. Filter input where model prediction is correct/incorrect
    :param output_class: Output class to filter. If output_class is a list, filter all input that have oc in this list
    :param sensitive_class: Sensitive class to filter.
    If sensitive_class is a list, filter all input that have sc in this list
    :return: A list of ID of input filtered
    """
    filtered = indexes_array.copy()
    if set_name != '' and set_name is not None:
        filtered = [line for line in filtered if line[p.set_name_pos] == set_name]
    if correct in [True, 'true']:
        filtered = [line for line in filtered if line[p.true_class_pos] == line[p.pred_class_pos]]
    elif correct in [False, 'false']:
        filtered = [line for line in filtered if line[p.true_class_pos] != line[p.pred_class_pos]]

    if isinstance(output_class, str):
        output_class = [output_class]
    if isinstance(output_class, list):
        filtered = [line for line in filtered if line[p.true_class_pos] in output_class]
    if isinstance(sensitive_class, str):
        sensitive_class = [sensitive_class]
    if isinstance(sensitive_class, list):
        filtered = [line for line in filtered if line[p.sens_attr_pos] in sensitive_class]

    if len(filtered) <= 0:
        logger.warning(
            f'No indexes after filtering{" for " + set_name + " set" if set_name else ""},{" output class : " + str(output_class) if output_class else ""}{" sensitive class : " + str(sensitive_class) if sensitive_class else ""}')
        return []

    return [int(line[p.input_id_pos]) for line in filtered]


def dataframe_to_loader(df: DataFrame, target: str) -> DataLoader:
    """Convert pandas dataframe to dataloader. Remove inputId column before processing"""
    features = torch.tensor(df.drop([target], axis=1).values)
    target_values = torch.tensor(df[target].values)
    dataset = TensorDataset(features, target_values)
    data_loader = DataLoader(dataset, batch_size=df.shape[0], shuffle=False)
    return data_loader


def file_reader(path: Path, sep: str = ',', header: Optional[list[str]] = None) -> list[list[str]]:
    """
    :param path: Path of the file to read
    :param sep: separator used, default is ','.
    :param header: Optional. If passed verify that the first line is the same as header.
    If header is [''], file reader skip the first line
    :return: List of list of string representing the file.
    """
    if not path.exists():
        raise ValueError(f'ERROR file_reader: Path {path} does not exist')
    with open(path, 'r') as file:
        content = file.readlines()

    if not content:
        logger.warning(f'Empty file at {path}')
        return []
    else:
        first_line = content[0]
        first_line = first_line.strip('\n').strip(' ').split(sep)
        result = []
        if header:
            if header != [''] and first_line != header:
                raise ValueError(
                    f'ERROR file_reader: Header of file is not the one expected. Got {first_line} instead of {header}')
        else:
            result.append(first_line)
        expected_len = len(first_line)

        if len(content) == 1:
            logger.warning(f'No content in file {path}')
            return result

        for line_nbr, line in enumerate(content[1:]):
            # remove \n and ' ' at the end and at the beginning of line, and make a list
            line = line.strip('\n').strip(' ').split(sep)
            # Ignore empty lines
            if line == [] or line == ['']:
                continue
            if len(line) != expected_len:
                raise ValueError(
                    f'ERROR file_reader: line {line_nbr} has incorrect format. Expected length {expected_len}, got {len(line)}')

            result.append(line)

        return result
