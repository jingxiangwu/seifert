#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:40:22 2024

@author: Jingxiang Wu
"""

from itertools import product
import numpy as np
import torch
from torch import optim
import math
from prettytable import PrettyTable
import pandas as pd
import time
from sympy import symbols, Matrix, simplify

from google.colab import drive
import os
# Dynamically create a filename based on data attributes
def create_filename(data_info):
    Nc = data_info['Nc']
    p_values = '_'.join(map(str, data_info['p_list']))
    q_values = '_'.join(map(str, data_info['q_list']))
    return f'/content/drive/MyDrive/Nc{Nc}_p{p_values}_q{q_values}.pkl'
# Check if Google Drive is mounted; if not, mount it
if not os.path.isdir('/content/drive/My Drive'):
    drive.mount('/content/drive')
else:
    print("Google Drive is already mounted.")
    

def DynkinPartition(data_single_fibre, anyonic=True):
    Nc = data_single_fibre['Nc']
    p = data_single_fibre['p']
    k = p - Nc
    if anyonic == False:
      k = p

    def helper(Nc, k, path, result):
        if Nc == 1:
            result.append(path + [k])
            return
        for i in range(k+1):
            helper(Nc-1, k-i, path + [i], result)

    result = []
    helper(Nc, k, [], result)
    output = np.array(result, dtype=np.int32)
    if anyonic == False:
      output -= 1

    return output[:, 1:]


def count_equalities_within_tolerance(elements, epsilon=1e-6):
  """
  elements: 1d np.array of complex numbers
  """
  assert len(elements.shape) == 1
  count = 0
  N = len(elements)

  for i in range(N):
      next_i = (i + 1) % N
      if np.isclose(elements[i], elements[next_i], atol=epsilon).all():
          count += 1

  return count


def find_ell(dyn, data_single_fibre):
    q = data_single_fibre['q']
    Nc = len(dyn) + 1
    num_boxes = np.sum(np.arange(1, Nc, dtype=np.int32) * dyn)
    temp = (Nc * (Nc - 1) / 2 - num_boxes) / q
    if temp % 1 == 0:
        ell = int(temp) % Nc
    else:
        print("error in find_ell", dyn, q)
    return ell

def dyn_to_mt(dyn):
    Nc = len(dyn) + 1
    mim1 = np.cumsum(dyn + 1)  # m^i - m^1
    min1_conc = np.concatenate(([0], mim1))
    output = min1_conc - np.sum(mim1) / Nc
    if np.isclose(np.sum(output), 0):
        return output
    else:
        print("error in dyn_to_mt", dyn)
def spin_to_eigen(spins):
    """Converts spins to eigenvalues. """
    return np.exp(2 * math.pi * 1j * spins)

def mt_to_dyn(mt):
    Nc = len(mt)
    dyn_output = mt[1:] - mt[:-1] - 1
    if np.all(dyn_output % 1 == 0):
        return dyn_output.astype(np.int32)
    else:
        print("error: non-integer dyns found")
        return None

def print_all_dyn(dyn_list, data_single_fibre):
    q = data_single_fibre['q']
    sorted_dyn_list = sorted(dyn_list, key=lambda dyn: (find_ell(dyn, data_single_fibre), tuple(dyn)))
    table = PrettyTable()
    table.field_names = ["Dynkin label", "mtilde", "ell"]
    for dyn in sorted_dyn_list:
        dyn_to_mt_val = np.round(dyn_to_mt(dyn), decimals=4)
        find_ell_val = find_ell(dyn, data_single_fibre)
        table.add_row([dyn, dyn_to_mt_val.tolist(), find_ell_val])
    print(table)

def create_dyn_dataframe(dyn_list, data_single_fibre):
    q = data_single_fibre['q']
    p = data_single_fibre['p']
    # Sorting the list, first by ell then by dynkin label
    sorted_dyn_list = sorted(dyn_list, key=lambda dyn: (find_ell(dyn, data_single_fibre), tuple(dyn)))

    # Data container for DataFrame
    data = {
        "Dynkin label": [],
        "mtilde": [],
        "ell": [],
        "eigenvalues": [],
        "eigen_degree":[]
    }

    for dyn in sorted_dyn_list:
        dyn_to_mt_val = dyn_to_mt(dyn)
        eigens = spin_to_eigen(dyn_to_mt_val/p)
        find_ell_val = find_ell(dyn, data_single_fibre)

        eigen_degree_list = count_equalities_within_tolerance(eigens)

        data["Dynkin label"].append(dyn)
        data["mtilde"].append(dyn_to_mt_val.tolist())
        data["ell"].append(find_ell_val)
        data["eigenvalues"].append(eigens)
        data["eigen_degree"].append(eigen_degree_list)

    df = pd.DataFrame(data)

    return df





def kronecker_product_dfs(dataframes):
    combined_rows = []

    # Assuming 'ell' is common and present in all DataFrames
    unique_ells = set(dataframes[0]['ell'].unique())
    for df in dataframes[1:]:
        unique_ells.intersection_update(set(df['ell'].unique()))

    for ell in unique_ells:
        # Filter rows for each DataFrame based on 'ell'
        filtered_dfs = [df[df['ell'] == ell].to_dict('records') for df in dataframes]

        # Produce Cartesian product of filtered rows
        row_products = list(product(*filtered_dfs))

        # Accumulate combined rows
        for row_product in row_products:
            combined_row = {'ell': ell}
            for col in dataframes[0].columns:
                if col != 'ell':
                    # Accumulate values from each DataFrame's row into a list
                    combined_row[col] = [row[col] for row in row_product]
            combined_rows.append(combined_row)

    # Convert the accumulated rows into a DataFrame
    df_big = pd.DataFrame(combined_rows)

    return df_big




def create_dyn_dataframe_all_fibres(data_all_fibres, anyonic=True):
  Nc = data_all_fibres['Nc']
  p_list = data_all_fibres['p_list']
  q_list = data_all_fibres['q_list']
  n = len(p_list)
  df_list = []
  for i in range(n):
    data_single_fibre = {'Nc':Nc, 'p':p_list[i], 'q':q_list[i]}
    dyn_list = DynkinPartition(data_single_fibre,anyonic)
    df = create_dyn_dataframe(dyn_list, data_single_fibre)
    df_list.append(df)
  return kronecker_product_dfs(df_list)

from IPython.display import display

def display_with_rounding(df, decimals=3):
    """
    Display a pandas DataFrame with numbers rounded to a given precision.

    Parameters:
    - df: pandas DataFrame to be displayed.
    - decimals: The number of decimal places to round each column to.
    """
    # Create a formatting string for the given number of decimals
    format_str = "{0:." + str(decimals) + "f}"

    # Define a function that rounds a number and works with both complex and float types
    def complex_round(x):
        if isinstance(x, complex):
            return complex(format_str.format(x.real), format_str.format(x.imag))
        elif isinstance(x, float):
            return format_str.format(x)
        return x

    # Apply the function to the DataFrame for display
    df_display = df.applymap(complex_round)

    # Use IPython's display function to show the DataFrame
    display(df_display)
    

data_all_fibres = {'Nc':3, 'p_list':[4,5,5], 'q_list':[1,1,1]}
df_big = create_dyn_dataframe_all_fibres(data_all_fibres, anyonic=False)

# Save the DataFrame with the dynamically created filename
filename = create_filename(data_all_fibres)
df_big.to_pickle(filename)

print(f"DataFrame saved to {filename}")