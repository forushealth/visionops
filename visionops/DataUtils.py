"""DataFrame/data-adapter utilities for classification data preparation."""

import os
import copy
from typing import Any, Optional, Union
import mimetypes
import io
import warnings

from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd
import numpy as np

from . import ImgUtils as imgu
from . import SysUtils as sysu

def _df_get_splits(df: pd.DataFrame, counts_or_fracs: Union[list,int], random_seed: int =None,
                    reset_index: bool =True):
    """NOTE: This function was extracted from df_get_splits() for refactoring purposes.
    Recommended to use that function instead as it adds stratified splitting support.

    Splits and returns smaller DataFrame samples based on list of elements / fractions.
    Optionally performs and shufflling.

    :param df: Input DataFrame
    :type df: pandas.core.frame.DataFrame
    :param counts_or_fracs: Total number of splits (int) or sample counts (list of ints) or sample fractions (list of floats).
                            If this is an integer, that many splits are created of equal length.
                            If last element is -1, length of the last split is auto-calculated.
                            Otherwise, all elements should either be integers (indicating respective sample counts), or fractions.
    :type counts_or_fracs: list or int
    :param random_seed: Any integer value will shuffle the df first.
                        Defaults to None (no shuffling).
    :type random_seed: int, optional
    :param reset_index: Whether to reset the original index. If False, all splits maintain the original index as read from the CSV.
                        If True, all splits start with index 0.
                        Defaults to True.
    :type reset_index: bool, optional
    :raises ValueError: If elements within 'counts_or_fracs' are of mixed datatypes (except when last element negetive)
    :raises ValueError: If elements of 'counts_or_fracs' adds to more than 1 or len(df)
    :return: List of DataFrames sampled from input df. As many DataFrames will be returned as the length of counts_or_fracs list.
    :rtype: list
    """
    df = df.copy()
    if type(counts_or_fracs) == int:    # Create array of equal fractions
        counts_or_fracs = np.ones(counts_or_fracs) / counts_or_fracs

    if np.isclose(sum(counts_or_fracs),1.0): # Requested all elements in total
        # But due to rounding errors in frac-to-count conversion, last few elements may be missed
        # So forcing last split to auto-detect remaining length
        counts_or_fracs[-1] = -1

    if counts_or_fracs[-1] < 0: # Auto-detect last split's length
        counts_or_fracs = counts_or_fracs[:-1]
        flag = True
    else:
        flag = False

    df_copy = df if not type(random_seed)==int else df.sample(frac=1, random_state=random_seed, ignore_index=reset_index)

    counts_or_fracs = np.array(counts_or_fracs)
    # if np.all(counts_or_fracs>=1):
    if all([x>=1 or x==0 for x in counts_or_fracs]):    # Integers
        count = counts_or_fracs       # e.g. [60,20,20]
    # elif np.all(counts_or_fracs<1) and np.all(counts_or_fracs>=0):
    elif all([x>=0 and x<1 for x in counts_or_fracs]):  # Fractions
        # Rounding errors can happen here!
        count = np.round(counts_or_fracs * len(df_copy)).astype(int) # e.g. [0.6, 0.2, 0.2]
    else:
        raise ValueError("Invalid input, elements must be all integers or fractions, only last element can be negative")

    if count.sum() > len(df_copy):
        raise ValueError(f"Invalid input, sum of split elements exceeding length of 'df' = {len(df_copy)}")

    split_indices = count.cumsum()

    if flag:
        df_list = np.array_split(df_copy, split_indices)
    else:
        df_list = np.array_split(df_copy, split_indices)[:-1]

    if reset_index:
        for df_elem in df_list:
            df_elem.reset_index(drop=True, inplace=True)

    return df_list


def _df_make_folds(df, counts_or_fracs, random_seed: int =None, fold_col_name="fold",
                   reset_index:bool = True):
    """NOTE: This function was extracted from df_make_folds() for refactoring purposes.
    Recommended to use that function instead as it adds stratified splitting support.

    Same as _df_get_splits() but instead of returning a list of different DataFrames,
    this returns a single DataFrame with a new column 'fold_col_name' added.
    The value of this coulmn indicates which split number the row belongs to.
    The fold values can be used to filter and create different training / val dataframes.

    :param df: Input DataFrame
    :type df: pandas.core.frame.DataFrame
    :param counts_or_fracs: Sampling counts or sampling fractions.
                            If this is an integer, that many splits are created of equal length.
                            If last element is -1, length of the last split is auto-calculated.
                            Otherwise, all elements should either be integers (indicating counts), or fractions.

    :type counts_or_fracs: list or int
    :param random_seed: Any integer value will shuffle the df first.
                        Defaults to None (no shuffling).
    :type random_seed: int, optional
    :param fold_col_name: Name of the new, fold number column to be created. Defaults to "fold".
    :type fold_col_name: str, optional
    :param reset_index: Has no effect if `random_seed` in None. Otherwise if False, all shuffled rows maintain the original index as read from the CSV.
                        If True, index is reset to start from 0.
                        Defaults to True.
    :type reset_index: bool, optional
    :return: A single DataFrame with a new column named 'fold_col_name' added.
            Value count of fold column will approximately be the same as indicated by corresponding elements in 'counts_or_fracs'
    :rtype: pandas.core.frame.DataFrame
    """
    if type(fold_col_name) is not str:
        raise TypeError("ERROR: 'fold_col_name' must be a string")
    df = df.copy()
    splits = _df_get_splits(df, counts_or_fracs, random_seed=random_seed, reset_index=False)
    for fold_num, temp_df in enumerate(splits):
        temp_df[fold_col_name] = fold_num

    df = pd.concat(splits, ignore_index=False)

    return df if not reset_index else df.sort_index()


def df_get_splits(df: pd.DataFrame, counts_or_fracs: Union[list,int], stratify_col_name=None,
                  random_seed: int =None, reset_index: bool = True):
    """Splits and returns smaller DataFrame samples based on list of elements / fractions provided.
    Optionally performs stratified splits and shufflling.

    :param df: Input DataFrame
    :type df: pandas.core.frame.DataFrame
    :param counts_or_fracs: Total number of splits (int) or sample counts (list of ints) or sample fractions (list of floats).
                            If this is an integer, that many splits are created of equal length.
                            If last element is -1, length of the last split is auto-calculated.
                            Otherwise, all elements should either be integers (indicating respective sample counts), or fractions.

    :type counts_or_fracs: list or int
    :param stratify_col_name: Name of the column in 'df' based on which stratified split is to be performed.
                                Leave as None to split without stratification. Defaults to None
    :type stratify_col_name: None or str, optional
    :param random_seed: Any integer value will shuffle the df first then perform the splitting.
                        An additional round of shuffling is performed on each split.
                        Defaults to None (no shuffling).
    :type random_seed: int, optional
    :param reset_index: Whether to reset the original index. If False, all splits maintain the original index as read from the CSV.
                        If True, all splits start with index 0.
                        Defaults to True.
    :type reset_index: bool, optional
    :return: List of DataFrames sampled from input df. As many DataFrames will be returned as the length of counts_or_fracs list.
    :rtype: list
    """
    df = df.copy()
    if stratify_col_name == None:
        return _df_get_splits(df, counts_or_fracs, random_seed, reset_index=reset_index)

    stratify_error = f"ERROR: Can't find column named '{stratify_col_name}' in df"
    if type(stratify_col_name) is not str:
        raise TypeError(stratify_error)
    if stratify_col_name not in df.columns:
        raise ValueError(stratify_error)

    vals = df[stratify_col_name].unique()

    # When performing stratified splits with integer counts, each fold's counts must be adjusted
    # This is not required if fractions are used since length of intermediate dataframes automatically becomes smaller
    if type(counts_or_fracs) == list and all([type(e)==int for e in counts_or_fracs]):
            counts_or_fracs = [e//len(vals) for e in counts_or_fracs]

    groups = df.groupby(stratify_col_name)
    splits = []
    for val in vals:
        df_temp = groups.get_group(val)
        temp_splits = _df_get_splits(df_temp, counts_or_fracs, random_seed, reset_index=reset_index)

        if splits == []:
            splits = [elem.copy() for elem in temp_splits]
        else:
            for i in range(len(splits)):
                splits[i] = pd.concat([splits[i], temp_splits[i].copy()], ignore_index=reset_index)

    if type(random_seed)==int:
        for i in range(len(splits)):
            splits[i] = splits[i].sample(frac=1, random_state=random_seed, ignore_index=reset_index)
    return splits


def df_make_folds(df, counts_or_fracs, stratify_col_name=None, random_seed: int =None, fold_col_name="fold_num",
                  reset_index:bool = True):
    """
    Same as df_get_splits() but instead of returning a list of different DataFrames,
    this returns a single DataFrame with a new column 'fold_col_name' added.
    The value of this coulmn indicates which split number the row belongs to.
    The fold values can be used to filter and create different training / val dataframes.
    Optionally performs stratified splits and shufflling.

    :param df: Input DataFrame
    :type df: pandas.core.frame.DataFrame
    :param counts_or_fracs: Total number of splits (int) or sample counts (list of ints) or sample fractions (list of floats).
                            If this is an integer, that many splits are created of equal length.
                            If last element is -1, length of the last split is auto-calculated.
                            Otherwise, all elements should either be integers (indicating respective sample counts), or fractions.

    :type counts_or_fracs: list or int
    :param stratify_col_name: Name of the column in 'df' based on which stratified split is to be performed.
                                Leave as None to split without stratification. Defaults to None
    :type stratify_col_name: None or str, optional
    :param random_seed: Any integer value will shuffle the df first then perform the splitting.
                        Defaults to None (no shuffling).
    :type random_seed: int, optional
    :param fold_col_name: Name of the new, fold number column to be created. Defaults to "fold".
    :type fold_col_name: str, optional
    :param reset_index: Has no effect if `random_seed` in None. Otherwise if False, all shuffled rows maintain the original index as read from the CSV.
                        If True, index is reset to start from 0.
                        Defaults to True.
    :type reset_index: bool, optional
    :return: A single DataFrame with a new column named 'fold_col_name' added.
            Value count of fold column will approximately be the same as indicated by corresponding elements in 'counts_or_fracs'
    :rtype: pandas.core.frame.DataFrame
    """
    df = df.copy()
    if stratify_col_name == None:
        return _df_make_folds(df, counts_or_fracs, random_seed, fold_col_name)

    stratify_error = f"ERROR: Can't find column named '{stratify_col_name}' in df"
    if type(stratify_col_name) is not str:
        raise TypeError(stratify_error)
    if stratify_col_name not in df.columns:
        raise ValueError(stratify_error)

    splits = df_get_splits(df, counts_or_fracs, stratify_col_name, random_seed, reset_index=False)
    for fold_num, df_split in enumerate(splits):
        df_split[fold_col_name] = fold_num

    df = pd.concat(splits, ignore_index=False)

    if type(random_seed)==int:
        df = df.sample(frac=1, random_state=random_seed, ignore_index=reset_index)

    return df if not reset_index else df.sort_index()


def df_get_class_weights(df, col_name_target, class_weights = "balanced", as_dict = False):
    """Calls Scikit-learn's compute_class_weight function on df[col_name_target]

    :param df: Pandas dataframe
    :type df: pd.DataFrame
    :param col_name_target: Name of the target / label / class column in the DataFrame
    :type col_name_target: str
    :param class_weights: dict, 'balanced' or None.
                            If 'balanced', class weights will be given by
                            ``n_samples / (n_classes * np.bincount(y))``.
                            If a dictionary is given, keys are classes and values
                            are corresponding class weights.
                            If None is given, the class weights will be uniform.
                            Defaults to "balanced".
    :type class_weights: dict, str, None
    :param as_dict: If true, returns a dictionary with class name and weight as key-value pairs.
                    If False, only returns a numpy array with weights.
                    Defaults to False.
    :type as_dict: bool, optional
    :return: Class weights
    :rtype: dict or numpy.ndarray
    """
    y = df[col_name_target]
    y_classes = y.unique()
    weights = compute_class_weight(class_weights, classes= y_classes, y= y)
    if as_dict:
        weights = dict(zip(y_classes, weights, strict=True))
    return weights


def df_oversample(df, col_name_target, random_seed=None):
    """Returns a dataframe that is oversampled based on df[col_name_target].
    Oversampling is done in a round-robin scheme such that every next item is from a different class,
    until all samples of every class has been included at least once.
    In effect, this will lead to samples from smaller classes to be included multiple times while
    samples from the largest class will be included only once.

    :param df: Dataframe
    :type df: pd.DataFrame
    :param col_name_target:  Name of the target / label / class column in the DataFrame
    :type col_name_target: str
    :param random_seed: Any integer value will shuffle the df first then perform the oversampling.
                        Defaults to None (no shuffling).
    :type random_seed: int, optional
    :return: Oversampled dataframe
    :rtype: pd.DataFrame
    """
    if random_seed is not None: df = df.sample(frac=1, random_state=random_seed)

    distrib = dict(df[col_name_target].value_counts())
    n_classes = len(distrib)
    max_len = max(distrib.values()) * n_classes

    rows = list()
    for idx in range(max_len):
        label_idx = idx % n_classes
        label = list(distrib.keys())[label_idx]
        row_id = (idx // n_classes) % distrib[label]
        row = df[df[col_name_target]==label].iloc[row_id]
        rows.append(row)

    df_temp = pd.DataFrame(rows)
    return df_temp


def df_plot_class_dist(df, col_name_class, col_name_group=None, bar_split_vertical=False,
                            x_axis_label = "Classes", legend_label = "Groups"):
    """
    Plots the distribution of classes in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col_name_class (str): The name of the column containing the target labels.
        col_name_group (str, optional): The name of the column indicating groups for the classes, e.g. train/val/test sets.
                                         If None, plots the distribution for the entire dataset.
        bar_split_vertical (bool, optional): If True, bars are stacked vertically; otherwise, horizontally. Defaults to False.
        x_axis_label (str, optional): Label to show for the x-axis. Defaults to "Classes".
        legend_label (str, optional): Label to show for the legend. Defaults to "Groups".
    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    classes = sorted(df[col_name_class].unique())

    if col_name_group is None:
        counts = df[col_name_class].value_counts().reindex(classes, fill_value=0)
        bars = ax.bar(classes, counts)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')
        ax.set(xlabel=x_axis_label, ylabel='Count', title='Class Distribution (Entire Dataset)')
    else:
        sets = df[col_name_group].unique()
        counts = {s: df[df[col_name_group] == s][col_name_class].value_counts().reindex(classes, fill_value=0) for s in sets}
        x = np.arange(len(classes))
        bar_width = 0.25

        if bar_split_vertical:
            bottom = np.zeros(len(classes))
            for s in sets:
                class_counts = [counts[s][c] for c in classes]
                bars = ax.bar(x, class_counts, bar_width, label=str(s), bottom=bottom)
                bottom += class_counts
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{int(height)}', ha='center', va='center')
        else:
            for i, s in enumerate(sets):
                offset = (i - len(sets) / 2 + 0.5) * bar_width
                bars = ax.bar(x + offset, counts[s], bar_width, label=s)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

        ax.set(xticks=x, xticklabels=classes, xlabel=x_axis_label, ylabel='Count', title='Class Distribution by grouping')
        ax.legend(title=legend_label)

    fig.tight_layout()
    return fig

# One-hot encoding, class mapping, etc.
def standardize_target_vector(y, mlc_sep = ','):
    """Classification tasks (binary, multiclass, multilabel) can have different representations of their target vectors.
    This adapter function can be used to convert most of such common representations into a standard output of list of lists.
    This can be useful for finding class-maps or performing one-hot encoding etc.
    E.g.
    >>> standardize_target_vector(["dog", "cat", "dog", "cat", "cat", "dog"]) # Binary classification
    [['dog'], ['cat'], ['dog'], ['cat'], ['cat'], ['dog']]

    >>> standardize_target_vector([1, 2, 1, 0, 3, 5, 2]) # Multi-class classification
    [[1], [2], [1], [0], [3], [5], [2]]

    >>> standardize_target_vector(np.array([1, 2, 1, 0, 3, 5, 2]))
    [[1], [2], [1], [0], [3], [5], [2]]

    >>> standardize_target_vector(np.array([[1], [2], [1], [0], [3], [5], [2]]))
    [[1], [2], [1], [0], [3], [5], [2]]

    >>> standardize_target_vector(["sky,grass", "river", "road,sky"]) # Multi-label classification
    [['sky', 'grass'], ['river'], ['road', 'sky']]

    >>> standardize_target_vector(["sky+grass", "river", "road+sky"], mlc_sep="+")
    [['sky', 'grass'], ['river'], ['road', 'sky']]

    >>> standardize_target_vector([["sky", "grass"],["river"],["road", "sky"]])
    [['sky', 'grass'], ['river'], ['road', 'sky']]

    :param y: Iterable of classes
    :type y: list, np.array etc.
    :param mlc_sep: A string seperator used for multi-label classification labeling.
                    Has no effect for binary / multi-class classification labels.
                    Defaults to ",".
    :type mlc_sep: str, optional
    """
    d_type = list(set(type(elem) for elem in y))
    if len(d_type) != 1:
        raise ValueError("ERROR: Input must be an iterable of same datatype")

    temp_list = []
    # If input is an array of strings, e.g. ["sky,grass", "river", "road,sky"]
    if d_type[0] == str or d_type[0] == np.str_:
        for i in range(len(y)):
            # Remove leading and trailing brackets,
            # we want labels appearing like "abc, def" not "['abc', 'def']"
            y[i] = y[i][1:] if y[i][0] in {"(", "{", "["} else y[i]
            y[i] = y[i][:-1] if y[i][-1] in {"(", "{", "["} else y[i]

            if mlc_sep in y[i]:
                temp_list.append(y[i].split(mlc_sep))
            else:
                temp_list.append([y[i]])
        y_2D = temp_list # e.g. y = [["sky", "grass"], ["river"], ["road", "sky"]]

    # If input is an array of numbers, e.g. [1, 2, 1, 0, 3, 5, 2]
    elif not all(hasattr(elem, "__iter__") for elem in y):
        y_2D = [[elem] for elem in y] # e.g. y = [[1], [2], [1], [0], [3], [5], [2]]

    # If input is an array of iterables, e.g. [["sky", "grass"],["river"],["road", "sky"]]
    else:
        y_2D = [list(elem) for elem in y] # Converts np.array into list, otherwise y_2D = y would have sufficed!

    return y_2D

def onehot_encode(y, map_label_to_idx = None, mlc_sep = ","):
    """Get the one-hot encoded vector for a given input and class map
    E.g:
    >>> onehot_encode([["sky"], ["river"], ["sky"], ["grass"]])
    array([[0, 0, 1],
       [0, 1, 0],
       [0, 0, 1],
       [1, 0, 0]])

    >>> c_map = {'sky': 0, 'grass': 1, 'river': 2, 'road': 3}
    >>> onehot_encode("sky", c_map)
    array([[1, 0, 0, 0]])

    >>> onehot_encode(["sky", "river", "sky", "grass"], c_map)
    array([[1, 0, 0, 0],
           [0, 0, 1, 0],
           [1, 0, 0, 0],
           [0, 1, 0, 0]])

    >>> onehot_encode([["sky"], ["river"], ["sky"], ["grass"]], c_map)
    array([[1, 0, 0, 0],
           [0, 0, 1, 0],
           [1, 0, 0, 0],
           [0, 1, 0, 0]])

    >>> onehot_encode([["sky"], ["river", "road"], ["sky","road"]], c_map)
    array([[1, 0, 0, 0],
           [0, 0, 1, 1],
           [1, 0, 0, 1]])

    >>> onehot_encode(["sky", "river,road", "sky,road"], c_map)
    array([[1, 0, 0, 0],
           [0, 0, 1, 1],
           [1, 0, 0, 1]])

    >>> onehot_encode(["sky", "river+road", "sky+road"], c_map)
    array([[1, 0, 0, 0],
           [0, 0, 1, 1],
           [1, 0, 0, 1]])

    >>> onehot_encode([["sky", "river", "sky", "grass"]], c_map) # AVOID INPUT FORMATS LIKE THIS!
    array([[1, 1, 1, 0]])

    :param y: Iterable of classes
    :type y: list, np.array etc.
    :param map_label_to_idx: Dictionary with keys that are unique classes in input 'y', and values are corresponding indexes.
                            If None, the class map is attempeted to be derived from the input sequence,
                            first label in the lexicographically sorted list of lables is assigned id 0 etc.
                            Defaults to None.
    :type map_label_to_idx: dict, Optional
    :param mlc_sep: A string seperator used for multi-label classification labeling.
                    Has no effect for binary / multi-class classification labels.
                    Defaults to ",".
    :type mlc_sep: str, optional
    :return: Numpy array with dim N x M, where N = len(y) and M = len(map_label_to_idx).
             All elements in this array are 0 except for those that correspond to y's labels
    :rtype: np.array
    """
    y = [y] if type(y) == str or not hasattr(y, "__iter__") else y
    if map_label_to_idx == None:
        map_label_to_idx = get_class_maps(y, mlc_sep=mlc_sep)[0]

    class_names = list(map_label_to_idx.keys()) if type(map_label_to_idx)==dict else map_label_to_idx
    y_2D = standardize_target_vector(y, mlc_sep)

    mlb = MultiLabelBinarizer(classes = class_names)
    return mlb.fit_transform(y_2D)

def onehot_decode(ohe_arr, map_idx_to_label = None, mlc_sep = None):
    """Get the classes from a one-hot encoded vector and class map.
    Class maps can be obtained by calling the function `get_class_maps`
    >>> map_i2l = {0: 'sky', 1: 'grass', 2: 'river', 3: 'road'}
    >>> onehot_decode(np.array([[1, 0, 0, 0]]), map_i2l)
    [('sky',)]

    >>> onehot_decode(np.array([[1, 0, 0, 0], [0, 0, 1, 0]]), map_i2l)
    [('sky',), ('river',)]

    >>> onehot_decode(np.array([[1, 0, 0, 0], [0, 0, 1, 0]]), map_i2l, mlc_sep=",")
    ['sky', 'river']

    >>> ohe_arr = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 1],
                            [1, 0, 0, 1]])
    >>> onehot_decode(ohe_arr, map_i2l, mlc_sep="+")
    ['sky', 'river+road', 'sky+road']

    >>> onehot_decode(np.array([[1, 0, 0, 1], [0, 0, 1, 0]]))
    [(0, 3), (2,)]

    :param ohe_arr: One-hot encoded array
    :type ohe_arr: np.array
    :param map_idx_to_label: Dictionary with keys that are indexes, values are corresponding unique labels in input.
                            If None, class IDs are assigned as per the column index.
                            Defaults to None.
    :type map_idx_to_label: dict, optional
    :param mlc_sep: A string seperator used to join multi-label classification labels.
                    Has effect only when the labels are strings.
                    Has no effect for binary / multi-class classification labels.
                    Defaults to None, i.e. no joining performed.
    :type mlc_sep: str, optional
    :return: List of decoded labels
    :rtype: list
    """
    if map_idx_to_label == None:
        map_idx_to_label = dict((i,i) for i in range(ohe_arr.shape[-1]))

    if ohe_arr.shape[-1] != len(map_idx_to_label):
        raise ValueError(
            f"ERROR: No. of columns ({ohe_arr.shape[-1]}) in input array must be same as number of classes ({len(map_idx_to_label)}) in 'map_idx_to_label'"
        )
    mlb = MultiLabelBinarizer(classes = list(map_idx_to_label.values()))

    class_0 = map_idx_to_label[0]
    class_type = type(class_0)  # Assume all elements of the dictionary have same type
    mlb.fit([[class_0]]) # Dummy fitting to first class arbitrarily to initialize the object, then use inverse_transform

    decoded_classes = mlb.inverse_transform(ohe_arr)
    if class_type == str and type(mlc_sep) == str:
        y = []
        for elem in decoded_classes:
            temp_str = mlc_sep.join(elem)
            y.append(temp_str)
    else:
        y = decoded_classes
    return y

def get_class_maps(y, sort="asc", mlc_sep = ","):
    """Given an array of outputs 'y', this returns two dictionaries and a string code infering the classification type.
    First dictionary is a mapping from all unique classes in 'y' to corresponding index,
    second dictionary is the reverse mapping.
    The string code indicates classification type: 'bc' (binary), 'mcc' (multi-class) or 'mlc' (multi-label)
    The index values will always be in the range(0, len(<num unique classes in y>))
    E.g.
    >>> get_class_maps(["dog", "cat", "dog", "cat", "cat", "dog"]) # Binary classification
    ({'cat': 0, 'dog': 1}, {0: 'cat', 1: 'dog'}, 'bc')

    >>> get_class_maps(np.array([[1], [2], [1], [0], [3], [5], [2]])) # Multi-class classification
    ({0: 0, 1: 1, 2: 2, 3: 3, 5: 4}, {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}, 'mcc')

    >>> get_class_maps([1, 2, 1, 0, 3, 5, 2])
    ({0: 0, 1: 1, 2: 2, 3: 3, 5: 4}, {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}, 'mcc')

    >>> get_class_maps([1, 2, 1, 0, 3, 5, 2], sort="asc")
    ({0: 0, 1: 1, 2: 2, 3: 3, 5: 4}, {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}, 'mcc')

    >>> get_class_maps([1, 2, 1, 0, 3, 5, 2], sort="des")
    ({5: 0, 3: 1, 2: 2, 1: 3, 0: 4}, {0: 5, 1: 3, 2: 2, 3: 1, 4: 0}, 'mcc')

    >>> get_class_maps([1, 2, 1, 0, 3, 5, 2], sort="fifo")
    ({1: 0, 2: 1, 0: 2, 3: 3, 5: 4}, {0: 1, 1: 2, 2: 0, 3: 3, 4: 5}, 'mcc')

    >>> get_class_maps([1, 2, 1, 0, 3, 5, 2], sort="lifo")
    ({5: 0, 3: 1, 0: 2, 2: 3, 1: 4}, {0: 5, 1: 3, 2: 0, 3: 2, 4: 1}, 'mcc')

    >>> get_class_maps(["sky,grass", "river", "road,sky"]) # Multi-label classification
    ({'grass': 0, 'river': 1, 'road': 2, 'sky': 3},
    {0: 'grass', 1: 'river', 2: 'road', 3: 'sky'}, 'mlc')

    >>> get_class_maps(["sky+grass", "river", "road+sky"], mlc_sep="+")
    ({'grass': 0, 'river': 1, 'road': 2, 'sky': 3},
    {0: 'grass', 1: 'river', 2: 'road', 3: 'sky'}, 'mlc')

    >>> get_class_maps([["sky", "grass"],["river"],["road", "sky"]])
    ({'grass': 0, 'river': 1, 'road': 2, 'sky': 3},
    {0: 'grass', 1: 'river', 2: 'road', 3: 'sky'}, 'mlc')

    :param y: Any iterable containing observed classes.
              All elements of this iterable MUST be of same datatype.
              NOTE: Tuple of single element is not considered a tuple
    :type y: iterable (list, np.array, etc.)
    :param sort: A string literal in {"fifo", "lifo", "asc", "desc"}.
                    * "asc" : Index 0 is assigned to the first label in the lexicographically sorted list of labels
                    * "des": Index 0 is assigned to the last label in the lexicographically sorted list of labels
                    * "fifo": Index 0 is assignet to the first label appearing in 'y'
                    * "lifo": Index 0 is assignet to the last label appearing in 'y'
                Defaults to "asc".
    :type sort: str, optional
    :param mlc_sep: A string seperator used for multi-label classification labeling.
                    Has no effect for binary / multi-class classification labels.
                    Defaults to ",".
    :type mlc_sep: str, optional
    :return: dict1: map_label_to_idx, keys are unique classes in input 'y', values are corresponding indexes
             dict2: map_idx_to_label, keys are indexes, values are corresponding unique classes in input 'y'
             clf_type: 'bc' (binary), 'mcc' (multi-class) or 'mlc' (multi-label)
    :rtype: (dict, dict, str)
    """
    if sort not in {"fifo", "lifo", "asc", "des"}:
        raise ValueError("sort must be one of {'fifo', 'lifo', 'asc', 'des'}")
    y = list(copy.deepcopy(y))  # Convert to list, needed if input is an immutable iterable like tuple

    clf_type = None # Classification type: 'bc' (binary), 'mcc' (multi-class) or 'mlc' (multi-label)
    d_type = list(set(type(elem) for elem in y))
    if len(d_type) != 1:
        raise ValueError("ERROR: Input must be an iterable of same datatype")

    # STEP 1: Clean up the target list
    y_2D = standardize_target_vector(y, mlc_sep)
    if any([len(elem)>1 for elem in y_2D]):
        clf_type = "mlc"

    # STEP 2: Flatten the list of list
    y_flattened = [item for sublist in y_2D for item in sublist]

    # STEP 3: Get unique elements and sort
    classes = list(dict.fromkeys(y_flattened))    # This preserves the order, unlike set. So this is "fifo" by default
    if not clf_type:
        clf_type = "mcc" if len(classes) >2 else "bc"

    if sort == "lifo":
        classes.reverse()
    elif sort == "asc":
        classes = sorted(classes)
    elif sort == "des":
        classes = sorted(classes, reverse=True)

    # STEP 3: Generate maps
    map_label_to_idx    = dict((elem, i) for i, elem in enumerate(classes))
    map_idx_to_label    = dict((i, elem) for i, elem in enumerate(classes))

    return map_label_to_idx, map_idx_to_label, clf_type

# Image Classification data class
class DataAdapterImgClf:
    """This class parses labelling information from a CSV file and prepares standardized dataframes for various types of tasks.
       It also provides convenience methods to easily split the dataframe into train, validation, test etc.
       These dataframes can then be easily used with PyTorch / TensorFlow dataset / dataloader classes.
    """
    def __init__(self, root_dir_path, col_name_path = "path", col_name_y = "label_name", mlc_sep = ",",
                 root_img_path = None, col_name_target="target", framework = None, onehot = False,
                 class_mapping = "asc", exclude_file_names = None):
        """Initialize the object. Read argument descriptions below for details.

        :param root_dir_path: This path is recursively searched for exactly 1 CSV file which is supposed
                              to contain all the image path and label information.
                              If images are in a different directory, use the `root_img_path` argument.
        :type root_dir_path: str
        :param col_name_path: In the CSV file, which column name contains the image paths.
                                Defaults to "path".
        :type col_name_path: str, optional
        :param col_name_y: In the CSV file, which column name contains the, ground truth label.
                            Defaults to "label_name".
        :type col_name_y: str, optional
        :param mlc_sep: If the dataset is of multi-label type, which seperator is used for entries in `col_name_y` column.
                        Has no effect for binary / multi-class classification labels.
                        Defaults to ",".
        :type mlc_sep: str, optional
        :param root_img_path: Add this prefix to the paths in `col_name_path`, no validation is performed.
                              If None, then the file name of `col_name_path` is recursively
                              searched in `root_dir_path` and it's absolute path is used.
                              Defaults to None.
        :type root_img_path: str | None, optional
        :param col_name_target: In the internal dataframe read from the CSV file, a new column
                                is created to hold the final ground-truth labels.
                                Use this argument to set the name of this column.
                                Values of this column maybe different from `col_name_y` based on
                                `framework` or `onehot` encoding arguments.
                                Defaults to "target".
        :type col_name_target: str, optional
        :param framework: One of {None, "pytorch", "tensorflow"}.
                          Used to format the target column `col_name_target` of the internal dataframe.
                          If None, original label column `col_name_y` is copied into `col_name_target`.
                          Defaults to None
        :type framework: str | None, optional
        :param onehot: One-hot encodeding preference for target column `col_name_target`.
                       Ignored if `framework` is None.
                       Will be overridden if the combination of `framework` and task type ("bc" / "mcc" / "mlc")
                       requires a different onehot logic.
                       Defaults to False.
        :type onehot: bool, optional
        :param exclude_file_names: List of file names to be excluded from the dataset.
                                   List elements should be string file names,
                                   if paths are provided then the file names are extracted.
                                   Defaults to None.
        :type exclude_file_names: list | None, optional
        """

        if not isinstance(root_dir_path, (str, os.PathLike)):
            raise TypeError("'root_dir_path' must be a path")
        if not os.path.exists(root_dir_path):
            raise FileNotFoundError(f"ERROR: Can't find any directory named f{root_dir_path}")

        self.col_name_y     = col_name_y
        self.col_name_path  = col_name_path
        self.col_name_target= col_name_target
        self.info_mlc_sep   = mlc_sep
        self.info_framework = framework
        self.info_is_onehot = onehot
        self.info_dataload_summary = dict()

        csv_search_dir = os.path.abspath(os.path.expanduser(root_dir_path))
        if root_img_path is not None and not isinstance(root_img_path, (str, os.PathLike)):
            raise TypeError("'root_img_path' must be a path or None")
        expanded_root_img_path = (
            os.path.expanduser(root_img_path) if root_img_path is not None else None
        )
        if expanded_root_img_path is not None and os.path.isdir(expanded_root_img_path):
            all_paths = sysu.search(os.path.abspath(expanded_root_img_path))
            csv_paths = sysu.search(csv_search_dir, ["*.csv", "*.CSV"])
        else:
            root_img_path = csv_search_dir # For logging in info_dataload_summary
            all_paths = sysu.search(csv_search_dir)
            csv_paths = [p for p in all_paths if p.lower().endswith(".csv")]

        # `all_paths` has all the absolute file paths found recursively inside image root dir
        # `csv_paths` has all the absolute CSV file paths found recursively inside CSV root dir
        # `file_paths` has all the absolute non-CSV file paths found recursively inside the image root dir
        file_paths = [p for p in all_paths if not p.lower().endswith(".csv")]

        if len(csv_paths) == 0:
            raise FileNotFoundError("ERROR: No CSV metadata file found!")

        df = pd.read_csv(csv_paths[0]) # Assume only 1 .csv file exists
        self.df_raw = df.copy()

        # Select only image files from file paths
        img_paths = list()
        for elem in file_paths:
            file_type_guess = mimetypes.guess_type(elem)[0]
            if type(file_type_guess)==str and file_type_guess.startswith('image'):
                img_paths.append(elem)

        # For each path in CSV, find corresponding absolute path
        # This is done since files can be at arbitrary depth from root_dir_path
        map_img_path = dict()
        list_img_not_in_dir = list() # list of paths that are in the CSV but not in the directory

        for path in df[col_name_path]:
            for abs_path in img_paths:
                if abs_path.endswith(path):
                    map_img_path.update({path:abs_path})
                    break
            if path not in map_img_path.keys():  # CSV entry is there but file not found
                map_img_path.update({path:None})
                list_img_not_in_dir.append(path)

        y_col = df[col_name_y].to_list()
        if type(class_mapping) == str:
            map_label_to_idx, map_idx_to_label, info_task = get_class_maps(y_col, sort= class_mapping, mlc_sep=",")
        elif type(class_mapping) == dict:
            map_label_to_idx, map_idx_to_label, info_task = get_class_maps(y_col, mlc_sep=",")
            if set(class_mapping.keys()) != set(map_label_to_idx.keys()):
                raise ValueError("Labels in `class_map` dictionary does not match unique class labels found in data. Please check your `class_mapping` dictionary or keep default string value 'asc'")
            if set(class_mapping.values()) != set(map_label_to_idx.values()):
                raise ValueError("Values in `class_map` dictionary must be a set of contiguous, non-negative integers. Please check your `class_mapping` dictionary or keep default string value 'asc'")

            map_label_to_idx = class_mapping
            # flip the key-value pairs in map_label_to_idx into a new dictionary map_idx_to_label
            map_idx_to_label = dict([(v,k) for k,v in map_label_to_idx.items()])

        unique_calsses = map_label_to_idx.keys()

        df["file_name"] = df[col_name_path].apply(os.path.basename)
        df[col_name_path] = df[col_name_path].map(map_img_path)
        df.dropna(inplace=True)

        # list of paths that are in the directory but not in the CSV
        list_img_not_in_csv = list(set(img_paths) - set(df[col_name_path]))

        if exclude_file_names:
            exclude_file_names = [os.path.basename(x) for x in exclude_file_names]
            df = df[df["file_name"].apply(lambda x:x not in exclude_file_names)]

        self.df = df    # Update after changes
        self.df_train   = None
        self.df_val     = None
        self.df_test    = None

        self.info_map_label_to_idx  = map_label_to_idx
        self.info_map_idx_to_label  = map_idx_to_label
        self.info_unique_classes    = unique_calsses
        self.info_task              = info_task

        self.to_framework(framework, onehot)

        # Create a backup of this dataframe for later resetting, if required
        self._df_bak    = self.df.copy()

        summary_txt = ""
        if len(csv_paths) >1:
            summary_txt+="WARNING: Multiple CSV files found! Using the first one."
        if len(list_img_not_in_csv) > 0:
            summary_txt+="\nWARNING: There are some image files in the data directory that don't have entries in the CSV (no labels). Excluding them."
        if len(list_img_not_in_dir) > 0:
            summary_txt+="\nWARNING: There are some file paths in the CSV that could not be found in the data directory (no images). Excluding them."
        if len(file_paths) - len(img_paths) > 0:
            summary_txt+="\nWARNING: Data directory not clean, i.e. there are some files that are neither images nor CSVs. Excluding them."
        if summary_txt == "":
            summary_txt = "All set!"
        print(summary_txt)

        self.info_dataload_summary["status"] = summary_txt
        self.info_dataload_summary["root_dir_path"] = root_dir_path
        self.info_dataload_summary["root_img_path"] = root_img_path
        self.info_dataload_summary["csv_file_processed"] = csv_paths[0]
        self.info_dataload_summary["count_csv"] = len(csv_paths)
        self.info_dataload_summary["count_non-img-csv_in_dir"] = len(file_paths) - len(img_paths)
        self.info_dataload_summary["count_img_in_dir"] = len(img_paths)
        self.info_dataload_summary["count_paths_in_csv"] = len(self.df_raw)
        self.info_dataload_summary["count_img_not_in_csv"] = len(list_img_not_in_csv)
        self.info_dataload_summary["count_img_not_in_dir"] = len(list_img_not_in_dir)
        self.info_dataload_summary["count_img_processed"] = len(df)
        self.info_dataload_summary["list_csv"] = csv_paths
        self.info_dataload_summary["list_non-img-csv_in_dir"] = list(set(file_paths) - set(img_paths))
        self.info_dataload_summary["list_img_not_in_csv"] = list_img_not_in_csv
        self.info_dataload_summary["list_img_not_in_dir"] = list_img_not_in_dir

        return

    def summary(self):
        """Return a human-readable summary of the current dataframe state."""
        summary_txt = "Head: \n"
        summary_txt += self.df.head().to_string()
        summary_txt += "\n\n"

        summary_txt += "Info:\n"
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        summary_txt += buffer.getvalue()
        summary_txt += "\n\n"

        summary_txt += "Description:\n"
        summary_txt += self.df.describe().to_string()
        summary_txt += "\n\n"

        summary_txt += "Value Counts:\n"
        summary_txt += self.df[self.col_name_y].value_counts().to_string()
        summary_txt += "\n\n"

        return summary_txt

    def get_class_dist_plots(self, bar_split_vertical=True):
        """
        Plots the distribution of classes in a DataFrame.

        Args:
        Returns:
            matplotlib.figure.Figure: The matplotlib figure object.
        """

        tvt_split_exists = "set" in self.df.columns
        plot1 = df_plot_class_dist(self.df, self.col_name_y, None, bar_split_vertical, "Classes", "Sets")
        if tvt_split_exists:
            plot2 = df_plot_class_dist(self.df, "set", None, bar_split_vertical, "Sets", "Classes")
            plot3 = df_plot_class_dist(self.df, self.col_name_y, "set", bar_split_vertical, "Classes", "Sets")
            plot4 = df_plot_class_dist(self.df, "set", self.col_name_y, bar_split_vertical, "Sets", "Classes")
        else:
            plot2 = None
            plot3 = None
            plot4 = None

        return plot1, plot2, plot3, plot4


    def to_framework(self, framework, onehot = False):
        """Process internal dataframe to be compatible with a given framework.

        :param framework: One of {None, "pytorch", "tensorflow"}.
                          Used to format the target column `col_name_target` of the internal dataframe.
                          If None, original label column `col_name_y` is copied into `col_name_target`.
        :type framework: str or None
        :param onehot: One-hot encodeding preference for target column `col_name_target`.
                       Ignored if `framework` is None.
                       Will be overridden if the combination of `framework` and task type ("bc" / "mcc" / "mlc")
                       requires a different onehot logic.
                       Defaults to False.
        :type onehot: bool, optional
        """

        if framework not in {None, "pytorch", "tensorflow"}:
            raise ValueError("ERROR: 'framework' must be in {None, 'pytorch', 'tensorflow'}")
        y_col = self.df[self.col_name_y].to_list()
        if framework == "pytorch":
            # Only numeric indexes are used for Binary and Multi-class data
            if self.info_task in {"bc", "mcc"}:
                if onehot:
                    target_col = onehot_encode(y_col, self.info_map_label_to_idx, self.info_mlc_sep).tolist()
                else:
                    target_col = [self.info_map_label_to_idx[elem] for elem in y_col]
            elif self.info_task in {"mlc"}:
                target_col = onehot_encode(y_col, self.info_map_label_to_idx, self.info_mlc_sep).tolist()
                onehot = True
            self.df[self.col_name_target] = target_col
            self.info_is_onehot = onehot

        elif framework == "tensorflow":
            if onehot:
                target_col = onehot_encode(y_col, self.info_map_label_to_idx, self.info_mlc_sep).tolist()
            else:
                target_col = y_col
            self.df[self.col_name_target] = target_col
            self.info_is_onehot = onehot

        elif framework == None:
            self.df[self.col_name_target] = self.df_raw[self.col_name_y]
            self.info_is_onehot = False

        # self.info_mlc_sep = mlc_sep
        self.info_framework = framework # Update internal variable when its done

        # Refresh train / val / test dataframes if they already existed
        if "set" in self.df.columns:
            _ = self.get_tvt_dfs()

        return


    def split(self, counts_or_fracs: Union[list,int], stratify=True, random_seed: int =None):
        """Splits and returns smaller DataFrame samples based on list of elements / fractions provided.
        Optionally performs stratified splits and shufflling.

        :param df: Input DataFrame
        :type df: pandas.core.frame.DataFrame
        :param counts_or_fracs: Total number of splits (int) or sample counts (list of ints) or sample fractions (list of floats).
                                If this is an integer, that many splits are created of equal length.
                                If last element is -1, length of the last split is auto-calculated.
                                Otherwise, all elements should either be integers (indicating respective sample counts), or fractions.

        :type counts_or_fracs: Union[list,int]
        :param stratify: If True, performs a stratified split based on self.y_col_name, otherwise performs splitting without stratification.
                         Defaults to True.
        :type stratify: bool, optional
        :param random_seed: Any integer value will shuffle the df first then perform the splitting.
                            Defaults to None (no shuffling).
        :type random_seed: int, optional
        :return: List of DataFrames sampled from input df. As many DataFrames will be returned as the length of counts_or_fracs list.
        :rtype: list
        """
        stratify_col = self.col_name_y if stratify == True else None
        return df_get_splits(copy.deepcopy(self.df), counts_or_fracs, stratify_col, random_seed)

    def create_tvt_split(self, counts_or_fracs: Optional[list] = None, stratify=True, random_seed: int =None):
        """ Create train-val-test splits by internally adding a column named 'set' to self.df
        The value of this coulmn indicates which phase (train/val/test) number the row belongs to.
        The dataframes for train-val-test can be easily obtained by calling .get_tvt_dfs() method.

        :param counts_or_fracs: Must be a list of 3 floats or ints where 1st, 2nd, 3rd values correspond to train, val, test sets respectively.
                                All elements should either be integers (indicating respective sample counts), or fractions.
                                Indicates sample counts (list of ints) or sample fractions (list of floats).
                                Defaults to [0.8, 0.2, 0.0]
        :type counts_or_fracs: list, optional
        :param stratify: If True, performs a stratified split based on self.y_col_name, otherwise performs splitting without stratification.
                         Defaults to True.
        :type stratify: bool, optional
        :param random_seed: Any integer value will shuffle the df first then perform the splitting.
                            Defaults to None (no shuffling).
        :type random_seed: int, optional
        """
        if counts_or_fracs is None:
            counts_or_fracs = [0.8, 0.2, 0.0]
        if type(counts_or_fracs) is not list:
            raise TypeError("'counts_or_fracs' must be a list")
        if len(counts_or_fracs) != 3:
            raise ValueError("'counts_or_fracs' must contain exactly three elements")

        stratify_col = self.col_name_y if stratify == True else None
        df_temp = df_make_folds(self.df, counts_or_fracs, stratify_col, random_seed, fold_col_name="set")
        df_temp["set"] = df_temp["set"].map({0:"train", 1:"val", 2:"test"})

        self.df = df_temp.copy()
        _ = self.get_tvt_dfs()
        return



    def create_folds(self, counts_or_fracs: Union[list,int], stratify=True, random_seed: int =None):
        """Internally modifies self.df to add a column named 'fold_num' which is alloted different fold numbers.
        The value of this coulmn indicates which fold number the row belongs to.
        The fold values can be used to filter and create different training / val dataframes, perform K-fold validation, etc.
        Optionally performs stratified splits and shufflling.

        :param counts_or_fracs: Total number of splits (int) or sample counts (list of ints) or sample fractions (list of floats).
                            If this is an integer, that many splits are created of equal length.
                            If last element is -1, length of the last split is auto-calculated.
                            Otherwise, all elements should either be integers (indicating respective sample counts), or fractions.

        :type counts_or_fracs: Union[list,int]
        :param stratify: If True, performs a stratified split based on self.y_col_name, otherwise performs splitting without stratification.
                         Defaults to True.
        :type stratify: bool, optional
        :param random_seed: Any integer value will shuffle the df first then perform the splitting.
                            Defaults to None (no shuffling).
        :type random_seed: int, optional
        """
        stratify_col = self.col_name_y if stratify == True else None
        self.df = df_make_folds(self.df, counts_or_fracs, stratify_col, random_seed)
        return

    def create_tvt_split_from_folds(self, train_folds: list, val_folds: list, test_folds: list):
        """Create train-val-test split using folds. Can't be used unless "create_folds" method is already called.
        This is useful for implementing K-fold style training.
        The dataframes for train-val-test can be easily obtained by calling .get_tvt_dfs() method.

        E.g. create 5 folds first then create dfferent combinations of test-val-train sets like:
        * train -> fold_num [0,1,2] combined; val -> fold_num [3], test -> fold_num [4]
        * train -> fold_num [0,1,3] combined; val -> fold_num [2], test -> fold_num [4]
        * train -> fold_num [1,4,0] combined; val -> fold_num [2,3] combined, test -> []
        etc.

        :param train_folds: List of fold numbers (int) to be included for training set
        :type train_folds: list of ints
        :param val_folds: List of fold numbers (int) to be included for validation set
        :type val_folds: list of ints
        :param test_folds: List of fold numbers (int) to be included for test set
        :type test_folds: list of ints
        """
        if "fold_num" not in self.df.columns:
            raise ValueError("No folds found, must call 'create_folds' method first")
        combined_inputs = train_folds + val_folds + test_folds
        max_fold_num = self.df["fold_num"].max()
        if any(type(e) is not int for e in combined_inputs):
            raise TypeError("List elements must be integers between 0 to max no. of folds")
        if any(e not in range(max_fold_num + 1) for e in combined_inputs):
            raise ValueError("List elements must be integers between 0 to max no. of folds")
        if len(combined_inputs) != len(set(combined_inputs)):
            raise ValueError("Duplicate fold numbers detected between train / val / test inputs")

        mapping_dict = dict()
        for fold_num  in combined_inputs:
            if fold_num in train_folds:
                mapping_dict.update({fold_num:"train"})
            elif fold_num in val_folds:
                mapping_dict.update({fold_num:"val"})
            else:
                mapping_dict.update({fold_num:"test"})

        self.df["set"] = self.df["fold_num"].map(mapping_dict)
        _ = self.get_tvt_dfs()
        return

    def get_tvt_dfs(self):
        """Convenience method to fetch train-val-test dataframes together

        :return: List of train, val and test dataframes in that respective order
        :rtype: list of Pandas DataFrames
        """
        if "set" not in self.df.columns:
            print("Train-Val-Test split not performed yet.\nFirst call create_tvt_split() or create_tvt_split_from_folds()")
            return [None, None, None]

        v_counts = self.df["set"].value_counts().to_dict()

        if v_counts.get("train", 0) == 0:
            self.df_train = None
        else:
            self.df_train   = self.df[self.df["set"]=="train"][[self.col_name_path, self.col_name_target]]
            self.df_train.reset_index(drop=True, inplace=True)

        if v_counts.get("val", 0) == 0:
            self.df_val = None
        else:
            self.df_val   = self.df[self.df["set"]=="val"][[self.col_name_path, self.col_name_target]]
            self.df_val.reset_index(drop=True, inplace=True)

        if v_counts.get("test", 0) == 0:
            self.df_test = None
        else:
            self.df_test   = self.df[self.df["set"]=="test"][[self.col_name_path, self.col_name_target]]
            self.df_test.reset_index(drop=True, inplace=True)


        return [self.df_train, self.df_val, self.df_test]

    def reset_dfs(self):
        """ Resets all dfs to their original state, i.e. after constructor call"""
        self.df = self._df_bak.copy()
        self.df_train = None
        self.df_val   = None
        self.df_test   = None
        return

    def to_csv(self, *args, **kwargs):
        """Simple wrapper for saving self.df
        Accepts all valid arguments for pd.DataFrame.to_csv method
        """
        self.df.to_csv(*args, **kwargs)

    def get_class_weights(self, class_weights = "balanced", as_dict = False):
        """Calls Scikit-learn's compute_class_weight function on the training set.
        If dataset is not yet split into train-val-test sets, the weights are computed on the full dataset.

        :param class_weights: dict, 'balanced' or None.
                                If 'balanced', class weights will be given by
                                ``n_samples / (n_classes * np.bincount(y))``.
                                If a dictionary is given, keys are classes and values
                                are corresponding class weights.
                                If None is given, the class weights will be uniform.
                                Defaults to "balanced".
        :type class_weights: dict, str, None
        :param as_dict: If true, returns a dictionary with class name and weight as key-value pairs.
                        If False, returns a numpy array with weights.
                        Defaults to False.
        :type as_dict: bool, optional
        :return: Class weights
        :rtype: dict or numpy.ndarray
        """
        if "set" not in self.df.columns:
            print("WARNING: Calculating weights for the full dataset instead of training set since train-val-test split is not yet performed.")
            print("First call create_tvt_split() or create_tvt_split_from_folds() then call this method again to get the class weights for only the training set.")
            weights = df_get_class_weights(self.df, self.col_name_y, class_weights, as_dict)
        else:
            weights = df_get_class_weights(self.df[self.df["set"]=="train"], self.col_name_y, class_weights, as_dict)

        return weights

    def image_grid(self, dataset_type = None, labels = None, count = 25, random_seed = None, figsize = (10,10),
                   display_filename = False, display_label_idx = False, display_in_notebook = True,
                   title = None):
        """Render an image grid from the current dataframe split/filter settings."""
        if dataset_type not in {None, "train", "val", "test"}:
            raise ValueError("'dataset_type' must be in {None, 'train', 'val', 'test'}")

        if labels == None:
            labels = set(self.info_unique_classes)
        if type(labels) == str:
            labels = {labels} # create a set of single label
        elif type(labels) == list:
            labels = set(labels)
        elif type(labels) != set:
            raise TypeError("`labels` should be a single string or a set or list of strings")
        if not labels.issubset(self.info_unique_classes):
            raise ValueError(f"One or more provided labels doesn't exist in the dataset. Available options are \n{self.info_unique_classes}")

        if dataset_type and "set" not in self.df.columns:
            print("WARNING: Ignoring split and considering full dataset since train-val-test split is not yet performed.")
            dataset_type = None

        df_temp = self.df[self.df[self.col_name_y].isin(labels)]

        if dataset_type == None:
            file_paths = df_temp[self.col_name_path].to_list()
            labels = df_temp[self.col_name_y].to_list()
            names = df_temp["file_name"].to_list()
        else:
            file_paths = df_temp[df_temp["set"]==dataset_type][self.col_name_path].to_list()
            labels = df_temp[df_temp["set"]==dataset_type][self.col_name_y].to_list()
            names = df_temp[df_temp["set"]==dataset_type]["file_name"].to_list()

        if display_label_idx:
            labels = [f"{self.info_map_label_to_idx[labels[i]]}:{labels[i]}" for i in range(len(labels))]
        if display_filename:
            labels = [f"{labels[i]}:{names[i]}" for i in range(len(labels))]

        return imgu.image_grid_from_pathlist(file_paths, labels, count, random_seed, figsize, display_in_notebook, title)

# ----------------------------------------------------------------------
# Compatibility helpers for the original configuration-dictionary API.
# ----------------------------------------------------------------------

class ConfigDict(dict):
    """Dictionary that auto-saves to YAML on every update."""

    def __init__(self, *args, yaml_path: str = "config.yaml", **kwargs):
        """Initialize dictionary and persist initial content to YAML."""
        self.yaml_path = yaml_path
        super().__init__(*args, **kwargs)
        self._save_yaml()

    def _save_yaml(self):
        import yaml

        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(self), f, sort_keys=False)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._save_yaml()

    def update(self, *args, **kwargs):
        """Update dictionary and persist changes to YAML."""
        super().update(*args, **kwargs)
        self._save_yaml()


def config_dictionary(experiment_name: str, version: str, yaml_path: str = "config.yaml") -> ConfigDict:
    """Create a configuration dictionary and persist it to YAML."""
    from datetime import date

    config = ConfigDict(
        {
            "yaml_path": yaml_path,
            "experiment_name": experiment_name,
            "experiment_version": str(version),
            "date": str(date.today()),
            "dataset_name": "",
            "data_directory_path": "",
            "image_size": (224, 224, 3),
            "train_data_csv": "",
            "val_data_csv": "",
            "image_path_column": "path",
            "image_label_column": "label",
            "image_categorical_label_column": "categorical_label",
            "batch_size": 32,
            "model_save_path": "",
            "model_checkpoints_path": "",
            "tag": [experiment_name, "BC"],
            "logging": {
                "use_mlflow": True,
                "log_hyperparams": True,
                "log_metrics": True,
                "log_artifacts": True,
            },
            "train_csv_path": f"../data/train/train_{experiment_name}_v{version}.csv",
            "test_csv_path": f"../data/test_val/test_{experiment_name}_v{version}.csv",
            "val_csv_path": f"../data/test_val/val_{experiment_name}_v{version}.csv",
            "model_path": f"../models/model_{experiment_name}_v{version}.h5",
            "logs_path": f"../logs/logs_{experiment_name}_v{version}.txt",
            "artifacts_path": f"../artifacts/{experiment_name}_v{version}",
            "train_prepros_pickle_path": f"../outputs/train_preprocessing_{experiment_name}_v{version}.pkl",
            "test_prepros_pickle_path": f"../outputs/val_preprocessing_{experiment_name}_v{version}.pkl",
        },
        yaml_path=yaml_path,
    )
    return config


def train_val_split(
    dataset_dataframe: pd.DataFrame,
    config_dict: dict,
    val_split: float = 0.2,
    test_split: float = None,
    image_path_column: str = "path",
    image_label_column: str = "labels",
    stratify_column_name: str = None,
):
    """Split dataframe into train/val (and optional test), write CSVs, and update config."""
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    df = dataset_dataframe.copy()
    config_dict["image_path_column"] = image_path_column
    config_dict["image_label_column"] = image_label_column

    if image_label_column in df.columns and isinstance(df[image_label_column].iloc[0], str):
        le = preprocessing.LabelEncoder()
        df["categorical_label"] = le.fit_transform(df[image_label_column])
        config_dict["image_categorical_label_column"] = "categorical_label"

    dataset_name = str(config_dict.get("dataset_name", "dataset")) or "dataset"
    version = str(config_dict.get("experiment_version", "1.0"))
    train_data_csv_path = f"{dataset_name}_{version}_train_data.csv"
    val_data_csv_path = f"{dataset_name}_{version}_val_data.csv"

    stratify_series = None
    if stratify_column_name and stratify_column_name in df.columns:
        stratify_series = df[stratify_column_name]

    if test_split:
        val_test_ratio = val_split / (1 - test_split)
        df_train, df_test = train_test_split(
            df,
            test_size=test_split,
            stratify=stratify_series,
            random_state=100,
        )
        stratify_val_test = df_test[stratify_column_name] if stratify_column_name and stratify_column_name in df_test.columns else None
        df_val, df_test = train_test_split(
            df_test,
            test_size=val_test_ratio,
            stratify=stratify_val_test,
            random_state=100,
        )
        test_data_csv_path = f"{dataset_name}_{version}_test_data.csv"
        df_test.to_csv(test_data_csv_path, index=False)
        config_dict["test_data_csv"] = test_data_csv_path
    else:
        df_train, df_val = train_test_split(
            df,
            test_size=val_split,
            stratify=stratify_series,
            random_state=100,
        )

    df_train.to_csv(train_data_csv_path, index=False)
    df_val.to_csv(val_data_csv_path, index=False)

    config_dict["train_data_csv"] = train_data_csv_path
    config_dict["val_data_csv"] = val_data_csv_path

    if test_split:
        return df_train, df_val, df_test
    return df_train, df_val


def load_data(
    dataset_dataframe: pd.DataFrame,
    config_dict: dict,
    image_size: tuple = (224, 224, 3),
    full_dataset: bool = False,
    number_images: int = 8,
) -> tuple:
    """Load image arrays and labels from dataframe using config keys."""
    from PIL import Image

    img_array = []
    label_array = []
    size_image_x, size_image_y = image_size[:2]
    config_dict["image_size"] = image_size

    data_directory = config_dict.get("data_directory_path", "")
    image_path_column = config_dict.get("image_path_column", "path")
    image_label_column = config_dict.get("image_label_column", "labels")

    if full_dataset:
        df_img_path = dataset_dataframe[image_path_column]
        df_img_label = dataset_dataframe[image_label_column]
    else:
        df_img_path = dataset_dataframe[image_path_column].iloc[:number_images]
        df_img_label = dataset_dataframe[image_label_column].iloc[:number_images]

    for i, name in enumerate(df_img_path):
        try:
            img = Image.open(f"{data_directory}/{name}").convert("RGB")
            img = img.resize((size_image_x, size_image_y))
            img_array.append(np.array(img))
            label_array.append(df_img_label.iloc[i])
        except Exception as e:
            print(f"Error loading image {name}: {e}")

    return np.array(img_array), np.array(label_array)


class VisualiseDataset:
    """Visualize original and preprocessed image grids."""

    def __init__(
        self,
        dataset_array: np.ndarray,
        image_number: int = 8,
        labels: Optional[np.ndarray] = None,
        class_names: Optional[Union[dict, list, tuple, np.ndarray]] = None,
    ):
        """Validate dataset tensor and initialize visualization state."""
        if not isinstance(dataset_array, np.ndarray):
            raise TypeError("dataset_array must be a numpy.ndarray")
        if dataset_array.ndim != 4:
            raise ValueError("dataset_array must be shape (n, h, w, c)")
        if image_number <= 0:
            raise ValueError("image_number must be positive")

        self.dataset_array = dataset_array
        self.image_number = min(image_number, len(dataset_array))
        self.class_names = class_names
        self.labels = None
        if labels is not None:
            label_array = np.asarray(labels)
            if label_array.ndim == 0:
                label_array = label_array.reshape(1)
            if label_array.shape[0] != len(dataset_array):
                raise ValueError(
                    "labels must have the same length as dataset_array (first dimension). "
                    f"Got labels={label_array.shape[0]} and dataset={len(dataset_array)}."
                )
            self.labels = label_array
        self.augmented_array = None
        self.preprocessed_array = None

    def _resolve_class_name(self, class_idx: int) -> str:
        """Map class index to readable class name when mapping is available."""
        mapping = self.class_names
        if mapping is None:
            return str(class_idx)
        if isinstance(mapping, dict):
            if class_idx in mapping:
                return str(mapping[class_idx])
            if str(class_idx) in mapping:
                return str(mapping[str(class_idx)])
            return str(class_idx)
        if isinstance(mapping, (list, tuple, np.ndarray)):
            if 0 <= class_idx < len(mapping):
                return str(mapping[class_idx])
        return str(class_idx)

    def _format_label_text(self, label_value: Any) -> str:
        """Convert scalar/one-hot/multi-hot labels into compact human-readable text."""
        arr = np.asarray(label_value)
        if arr.ndim == 0:
            value = arr.item()
            if isinstance(value, (np.integer, int)):
                return self._resolve_class_name(int(value))
            if isinstance(value, (np.floating, float)) and np.isfinite(value):
                rounded = int(round(float(value)))
                if abs(float(value) - rounded) < 1e-6:
                    return self._resolve_class_name(rounded)
            return str(value)

        flat = arr.reshape(-1)
        if flat.size == 0:
            return "N/A"

        if np.issubdtype(flat.dtype, np.number):
            vals = flat.astype(np.float32)
            if vals.size == 1:
                return self._format_label_text(vals[0])

            if np.all((vals >= 0.0) & (vals <= 1.0)):
                active = np.where(vals > 0.5)[0].tolist()
                if active:
                    return ", ".join(self._resolve_class_name(int(idx)) for idx in active)
                return "none"

            if np.isclose(float(np.sum(vals)), 1.0, atol=1e-3):
                return self._resolve_class_name(int(np.argmax(vals)))

            return np.array2string(np.round(vals, 3), separator=",")

        return ", ".join(str(x) for x in flat.tolist())

    def augmentation(self, augmentation_fn=None) -> np.ndarray:
        """Apply augmentation function (or default Albumentations pipeline)."""
        if augmentation_fn is not None:
            self.augmented_array = np.array([augmentation_fn(img) for img in self.dataset_array])
            return self.augmented_array

        try:
            import albumentations as A

            def _to_uint8(image: np.ndarray) -> np.ndarray:
                arr = np.asarray(image)
                if arr.dtype == np.uint8:
                    return arr
                arr = arr.astype(np.float32)
                if arr.size == 0:
                    return arr.astype(np.uint8)
                mn = float(np.nanmin(arr))
                mx = float(np.nanmax(arr))
                if mn >= 0.0 and mx <= 1.0:
                    arr = arr * 255.0
                elif mn < 0.0 or mx > 255.0:
                    denom = mx - mn if abs(mx - mn) > 1e-6 else 1.0
                    arr = (arr - mn) / denom * 255.0
                return np.clip(arr, 0.0, 255.0).astype(np.uint8)

            aug = A.Compose(
                [
                    A.CLAHE(clip_limit=10, tile_grid_size=(8, 8), p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Rotate(limit=55, p=0.5),
                ]
            )
            self.augmented_array = np.array([aug(image=_to_uint8(img))["image"] for img in self.dataset_array])
        except Exception:
            # Fallback when albumentations is unavailable.
            self.augmented_array = self.dataset_array.copy()
        return self.augmented_array

    def preprocessing(self, preprocessing_fn=None) -> np.ndarray:
        """Apply preprocessing function or default normalization."""
        images = self.augmented_array if self.augmented_array is not None else self.augmentation()
        if preprocessing_fn is not None:
            self.preprocessed_array = np.array([preprocessing_fn(img) for img in images])
        else:
            arr = images.astype("float32")
            if arr.size > 0 and (float(np.nanmin(arr)) < 0.0 or float(np.nanmax(arr)) > 1.0):
                mins = arr.min(axis=(1, 2, 3), keepdims=True)
                maxs = arr.max(axis=(1, 2, 3), keepdims=True)
                denom = np.where((maxs - mins) < 1e-6, 1.0, maxs - mins)
                arr = (arr - mins) / denom
            self.preprocessed_array = np.clip(arr, 0.0, 1.0)
        return self.preprocessed_array

    def display(self):
        """Display original vs preprocessed image rows and return figure."""
        preprocessed = self.preprocessing() if self.preprocessed_array is None else self.preprocessed_array

        fig, axes = plt.subplots(
            2,
            self.image_number,
            figsize=(self.image_number * 2.8, 2 * 2.8),
            dpi=100,
            squeeze=False,
        )

        for i in range(self.image_number):
            label_text = self._format_label_text(self.labels[i]) if self.labels is not None else "N/A"
            top_title = f"Original\nLabel: {label_text}" if self.labels is not None else "Original"
            bottom_title = (
                f"Preprocessed & Augmented\nLabel: {label_text}"
                if self.labels is not None
                else "Preprocessed & Augmented"
            )
            axes[0, i].imshow(self.dataset_array[i])
            axes[0, i].set_title(top_title, fontsize=9)
            if self.labels is not None:
                axes[0, i].text(
                    0.02,
                    0.02,
                    f"Label: {label_text}",
                    transform=axes[0, i].transAxes,
                    fontsize=8,
                    color="white",
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
                )
            axes[0, i].axis("off")

            axes[1, i].imshow(preprocessed[i])
            axes[1, i].set_title(bottom_title, fontsize=9)
            if self.labels is not None:
                axes[1, i].text(
                    0.02,
                    0.02,
                    f"Label: {label_text}",
                    transform=axes[1, i].transAxes,
                    fontsize=8,
                    color="white",
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
                )
            axes[1, i].axis("off")

        plt.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.04, hspace=0.35, wspace=0.08)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*FigureCanvasAgg is non-interactive.*", category=UserWarning)
            plt.show()
        return fig


def visualise_dataset(
    dataset_array: np.ndarray,
    image_number: int = 8,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[Union[dict, list, tuple, np.ndarray]] = None,
):
    """Legacy wrapper to visualize dataset."""
    viz = VisualiseDataset(
        dataset_array,
        image_number=image_number,
        labels=labels,
        class_names=class_names,
    )
    return viz.display()


def plot_histogram_per_class(df: pd.DataFrame, class_mapping: dict = None):
    """Plot number of images per class and return figure."""
    import seaborn as sns

    if class_mapping is None:
        unique_classes = sorted(df["labels"].unique())
        class_mapping = {cls: i for i, cls in enumerate(unique_classes)}

    class_counts = df["labels"].value_counts().reindex(class_mapping.keys(), fill_value=0)
    classes = sorted(class_mapping.keys(), key=lambda x: class_mapping[x])
    x = np.arange(len(classes))
    colors = sns.color_palette("tab20", len(classes))

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, [class_counts[cls] for cls in classes], width=0.8, color=colors, edgecolor="black")

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height / 2, str(int(height)), ha="center", va="center")

    ax.set_xlabel("Class Labels")
    ax.set_ylabel("Number of Images")
    ax.set_title("Histogram of Number of Images per Class (Dataset)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    return fig


def analyze_image_classification_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
) -> tuple:
    """Run chi-square balance checks and return (distribution_figure, summary_text)."""
    import seaborn as sns
    from scipy.stats import chi2_contingency

    datasets = {"Train": train_df, "Validation": val_df}
    if test_df is not None:
        datasets["Test"] = test_df

    alpha = 0.05
    summary_paragraphs = []

    for name, df in datasets.items():
        k = max(1, df["labels"].nunique())
        expected = [len(df) / k] * k
        observed = df["labels"].value_counts().sort_index().values
        try:
            chi2, p, _, _ = chi2_contingency([observed, expected])
            if p > alpha:
                summary_paragraphs.append(
                    f"{name} set: chi2={chi2:.2f}, p={p:.4f} (p > {alpha}): "
                    "no significant imbalance. The dataset is well-balanced."
                )
            else:
                summary_paragraphs.append(
                    f"{name} set: chi2={chi2:.2f}, p={p:.4f} (p <= {alpha}): "
                    "significant imbalance detected! Consider data augmentation or weighted loss."
                )
        except Exception as e:
            summary_paragraphs.append(f"{name} set: chi-square could not be computed ({e}).")

    summary = "\n\n".join(summary_paragraphs)

    combined = pd.concat([df.assign(split=name) for name, df in datasets.items()], ignore_index=True)
    dist_fig, dist_ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x="labels", hue="split", data=combined, ax=dist_ax)
    dist_ax.set_title("Class Distribution in Train, Validation, and Test Sets")
    dist_ax.tick_params(axis="x", rotation=45, labelright=False, labelleft=True)
    plt.tight_layout()
    plt.show()
    return dist_fig, summary
