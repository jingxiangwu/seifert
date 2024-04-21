from itertools import product
from prettytable import PrettyTable
from IPython.display import display
import numpy as np
import pandas as pd


class ConnCandidates:
    def __init__(self, data_all_fibres, anyonic=True):
        self.data_all_fibres = data_all_fibres
        self.anyonic = anyonic
    """
    # single fibre analysis
    """

    # Function to partition Dynkin labels based on the provided data and criteria (anyonic or not).
    def _dynkin_partition(self, data_single_fibre):
        # Extract necessary parameters from the data.
        Nc = data_single_fibre['Nc']
        p = data_single_fibre['p']
        k = p - Nc if self.anyonic else p  # Adjust 'k' based on whether anyonic.

        # Helper function for recursive partition generation.
        def helper(Nc, k, path, result):
            # If at the last step, append the path to the result.
            if Nc == 1:
                result.append(path + [k])
                return
            # Iterate and recursively build paths.
            for i in range(k+1):
                helper(Nc-1, k-i, path + [i], result)

        result = []
        helper(Nc, k, [], result)

        # Convert the result to a numpy array and adjust if not anyonic.
        output = np.array(result, dtype=np.int32)[:, 1:]
        if not self.anyonic:
            output -= 1

        return output

    def _count_equalities_within_tolerance(self, elements, epsilon=1e-6):
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

    def _find_ell(self, dyn, data_single_fibre):
        q = data_single_fibre['q']
        Nc = len(dyn) + 1
        num_boxes = np.sum(np.arange(1, Nc, dtype=np.int32) * dyn)
        temp = (Nc * (Nc - 1) / 2 - num_boxes) / q
        if temp % 1 == 0:
            ell = int(temp) % Nc
        else:
            print("error in find_ell", dyn, q)
        return ell

    def _dyn_to_mtilde(self, dyn):
        Nc = len(dyn) + 1
        mim1 = np.cumsum(dyn + 1)  # m^i - m^1
        min1_conc = np.concatenate(([0], mim1))
        output = min1_conc - np.sum(mim1) / Nc
        if np.isclose(np.sum(output), 0):
            return output
        else:
            print("error in dyn_to_mt", dyn)

    def _mtilde_to_dyn(self, mt):
        dyn_output = mt[1:] - mt[:-1] - 1
        if np.all(dyn_output % 1 == 0):
            return dyn_output.astype(np.int32)
        else:
            print("error: non-integer dyns found")
            return None

    # Function to print all Dynkin labels in a pretty table format.
    def print_all_dyn(self, dyn_list, data_single_fibre):
        # q = data_single_fibre['q']
        sorted_dyn_list = sorted(dyn_list,
                                 key=lambda dyn:
                                     (self._find_ell(dyn, data_single_fibre), tuple(dyn)))
        table = PrettyTable()
        table.field_names = ["Dynkin_label", "mtilde", "ell"]
        for dyn in sorted_dyn_list:
            dyn_to_mt_val = np.round(self._dyn_to_mtilde(dyn), decimals=4)
            find_ell_val = self._find_ell(dyn, data_single_fibre)
            table.add_row([dyn, dyn_to_mt_val.tolist(), find_ell_val])
        print(table)

    def _spin_to_eigen(self, spins):
        """Converts spins to eigenvalues. """
        return np.exp(2 * np.pi * 1j * spins)

    def _create_dyn_dataframe(self, dyn_list, data_single_fibre):
        p = data_single_fibre['p']
        # Sorting the list, first by ell then by dynkin label
        sorted_dyn_list = sorted(dyn_list,
                                 key=lambda dyn:
                                     (self._find_ell(dyn, data_single_fibre), tuple(dyn)))

        # Data container for DataFrame
        data = {
            "Dynkin_label": [],
            "mtilde": [],
            "ell": [],
            "eigenvalues": [],
            "eigen_degree": []
        }

        for dyn in sorted_dyn_list:
            dyn_to_mt_val = self._dyn_to_mtilde(dyn)
            eigens = self._spin_to_eigen(dyn_to_mt_val / p)
            find_ell_val = self._find_ell(dyn, data_single_fibre)
            eigen_degree_list = self._count_equalities_within_tolerance(eigens)

            data["Dynkin_label"].append(dyn)
            data["mtilde"].append(dyn_to_mt_val.tolist())
            data["ell"].append(find_ell_val)
            data["eigenvalues"].append(eigens)
            data["eigen_degree"].append(eigen_degree_list)

        return pd.DataFrame(data)

    """
    Here starts the multiple fibres analysis
    """

    def _kronecker_product_dfs(self, dataframes):
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

    def create_dyn_dataframe_all_fibres(self):
        Nc = self.data_all_fibres['Nc']
        p_list = self.data_all_fibres['p_list']
        q_list = self.data_all_fibres['q_list']
        n = len(p_list)
        df_list = []
        for i in range(n):
            data_single_fibre = {'Nc': Nc, 'p': p_list[i], 'q': q_list[i]}
            dyn_list = self._dynkin_partition(data_single_fibre)
            df = self._create_dyn_dataframe(dyn_list, data_single_fibre)
            df_list.append(df)

        def convert_to_tuple_of_tuples(lst):
            return tuple(tuple(arr) for arr in lst)

        df_big = self._kronecker_product_dfs(df_list)
        df_big['Dynkin_label'] = df_big['Dynkin_label'].apply(convert_to_tuple_of_tuples)
        df_big['eigen_degree'] = df_big['eigen_degree'].apply(tuple)
        df_big['if_irreducible'] = None
        df_big['final_loss'] = None
        df_big.sort_values(by=['eigen_degree', 'ell'], inplace=True)
        df_big.reset_index(drop=True, inplace=True)
        return df_big

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
