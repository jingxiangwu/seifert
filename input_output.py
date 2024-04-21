# dual_output.py
import sys
from contextlib import contextmanager
from conn_optimizer_batch import FlatConn


@contextmanager
def print_and_save_to_file(filename, mode='w', output_option="both"):
    class DualOutput:
        def __init__(self, filename, mode, output_option):
            self.terminal = sys.stdout
            self.log = open(filename, mode)
            self.output_option = output_option

        def write(self, message):
            if self.output_option in ("screen", "both"):  # Option 1: Only screen, Option 3: Both
                self.terminal.write(message)
            if self.output_option in ("file", "both"):  # Option 2: Only file, Option 3: Both
                self.log.write(message)

        def flush(self):
            if self.output_option in ("screen", "both"):
                self.terminal.flush()
            if self.output_option in ("file", "both"):
                self.log.flush()

    dual_output = DualOutput(filename, mode, output_option)
    original_stdout = sys.stdout
    sys.stdout = dual_output
    try:
        yield
    finally:
        sys.stdout = original_stdout
        dual_output.log.close()



import pandas as pd
import torch  # Ensure torch is imported if using torch.Tensor

def update_if_irreducible(df, output_dict, output_loss, output_X_matrices, option="ask"):
    """
    Updates the 'if_irreducible' column in the DataFrame based on a dictionary,
    with interactive options for handling existing non-None values and also updates
    'final_loss' and 'final_X_matrices' if 'if_irreducible' is updated.
    Args:
    df (pd.DataFrame): The DataFrame containing 'Dynkin_label', 'if_irreducible', etc.
    output_dict (dict): Dictionary mapping 'Dynkin_label' to new 'if_irreducible' statuses.
    output_loss (dict): Dictionary mapping 'Dynkin_label' to new 'final_loss' values.
    output_X_matrices (dict): Dictionary mapping 'Dynkin_label' to new 'final_X_matrices' values.
    option (int): Determines the overwrite behavior:
        1 - Ask interactively for each mismatch whether to overwrite.
        2 - Overwrite all.
        3 - Do not overwrite any existing non-None values.
    Returns:
    pd.DataFrame: The updated DataFrame.
    """
    def update_row(row_old):
        row = row_old.copy() # Create a copy of the row to avoid modifying the original in-place
        current_irreducible = row['if_irreducible'] # existing value at "if_irreducible" column
        new_irreducible = output_dict.get(row['Dynkin_label']) # new value at "if_irreducible" column
        corrections = 0 # count the number of non-irreducible -> irreducible
        if new_irreducible is not None:
            if current_irreducible is None or option == "overwrite":
                # if current is empty fill in the new value
                # overwrite unless current is irreducible
                if current_irreducible != "irreducible":
                    row['if_irreducible'] = new_irreducible
                    row['final_loss'] = output_loss.get(row['Dynkin_label'], row['final_loss'])
                    row['final_X_matrices'] = output_X_matrices.get(row['Dynkin_label'], row['final_X_matrices'])
                    if new_irreducible == "irreducible" and current_irreducible != "irreducible":
                        corrections += 1
            elif current_irreducible != new_irreducible:
                if option == "ask":
                    print(f"Mismatch found: {row['Dynkin_label']}\n - Current: {current_irreducible}, current_loss: {row['final_loss']},\n - New: {new_irreducible}, new_loss: {output_loss.get(row['Dynkin_label'])}")
                    response = input("Do you want to overwrite? (y/n): ")
                    if response.lower() == 'y':
                        row['if_irreducible'] = new_irreducible
                        row['final_loss'] = output_loss.get(row['Dynkin_label'], row['final_loss'])
                        row['final_X_matrices'] = output_X_matrices.get(row['Dynkin_label'], row['final_X_matrices'])
                elif option == "keep":
                    pass  # Do nothing, keep existing values

        # Handle cases where new values are torch.Tensors
        if isinstance(row['final_loss'], torch.Tensor):
            row['final_loss'] = row['final_loss'].item()

        row["corrections"] = corrections
        return row

    if 'if_irreducible' not in df.columns:
        df['if_irreducible'] = None
    if 'final_loss' not in df.columns:
        df['final_loss'] = None
    if 'final_X_matrices' not in df.columns:
        df['final_X_matrices'] = None

    # Apply the process_row function and create a new DataFrame
    results = df.apply(update_row, axis=1)
    df = results.drop('corrections', axis=1)  # Drop the 'corrections' column
    count = results['corrections'].sum()  # Sum the 'Modified' column for the total count of modifications 


    return df, count

# # Example usage
# df_temp = pd.DataFrame({
#     'Dynkin_label': ['A1', 'A2', 'B1'],
#     'if_irreducible': [None, 'irreducible', 'reducible'],
#     'final_loss': [None, 2.5, 3.0],
#     'final_X_matrices': [None, "MatrixA2", "MatrixB1"]
# })

# output_1 = {'A1': 'irreducible', 'A2': 'reducible', 'B1': 'irreducible'}
# output_2 = {'A1': torch.tensor(1.2), 'A2': 2.1, 'B1': torch.tensor(0.8)}
# output_3 = {'A1': "MatrixA1", 'A2': "MatrixA2_new", 'B1': "MatrixB1_new"}

# # Option can be 1, 2, or 3 based on your choice
# updated_df = update_if_irreducible(df_temp, output_1, output_2, output_3, option="ask")
# print(updated_df)

import pandas as pd

def split_dataframe(df, n):
    # Calculate the size of each chunk
    chunk_size = len(df) // n
    
    # Create a list of DataFrames, where each chunk is of size chunk_size
    sub_dfs = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    

    return sub_dfs

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import time

def single_run(data_all_fibres, conn_candidates, parameters):
    filename, lr, steps, batch_size, num_threads, mode, output_option = parameters
    with print_and_save_to_file(filename, mode=mode, output_option=output_option):
        model = FlatConn(data_all_fibres, conn_candidates, lr=lr, steps=steps, batch_size=batch_size, num_threads=num_threads)
        model.gpu()  # Run optimizations
        output_if_irreducible, output_loss, output_X_matrices = model.cpu() # check irreducible

    # Detach tensors and convert to numpy arrays if within a dictionary
    if isinstance(output_if_irreducible, torch.Tensor):
        output_if_irreducible = output_if_irreducible.detach().item()  # Convert to a basic type if it's a single value tensor
    if isinstance(output_loss, torch.Tensor):
        output_loss = output_loss.detach().item()  # Convert to a basic type if it's a single value tensor
    if isinstance(output_X_matrices, dict):
        # Iterate through each key in the dictionary where each value is a list of tensors
        for key, tensor_list in output_X_matrices.items():
            output_X_matrices[key] = [tensor.detach().cpu().numpy() for tensor in tensor_list]

    return (output_if_irreducible, output_loss, output_X_matrices)

def run_model(data_all_fibres, conn_candidates_list, output_file, 
              max_workers=48, lr=0.01, steps=15001, batch_size=20, num_threads=48, output_option="both"):
    aggregated_irreducible = defaultdict(list)
    aggregated_loss = defaultdict(list)
    aggregated_matrices = defaultdict(dict)

    tic_full = time.time()
    count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(single_run, data_all_fibres, conn, 
                                   [f"{output_file}_{idx}", lr, steps, batch_size, num_threads, 'w', output_option]
                                   )
                                   : conn for idx, conn in enumerate(conn_candidates_list)}

        # Retrieve results as they complete
        for future in as_completed(futures):
            conn_candidates = futures[future]
            try:
                output_if_irreducible, output_loss, output_X_matrices = future.result()
                # Aggregate results
                for key, value in output_if_irreducible.items():
                    aggregated_irreducible[key] = (value)
                for key, value in output_loss.items():
                    aggregated_loss[key] = (value)
                for key, matrices in output_X_matrices.items():
                    if key not in aggregated_matrices:
                        aggregated_matrices[key] = []
                    aggregated_matrices[key] = (matrices)  # Assuming matrices is a list of numpy arrays
                count += 1
                print(f"Computation {count} completed")
            except Exception as e:
                print(f"An error occurred with {conn_candidates}: {str(e)}")
    toc_full = time.time()
    print(f"total time = {toc_full-tic_full}")
    return aggregated_irreducible, aggregated_loss, aggregated_matrices