import time
from sympy import symbols, Matrix
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity
from concurrent.futures import ThreadPoolExecutor



class FlatConn:
    def __init__(self, data_all_fibres, df_big,
                 lr=0.005, steps=20001, loss_threshold=1e-12,
                 batch_size=72,
                 torch_dtype=torch.complex128,
                 np_dtype=np.complex128,
                 num_threads=18):
        """
        data_all_fibres: {'Nc':3, 'p_list':[p1,p2,p3], 'q_list':[q1,q2,q3]}
        filter_criteria
        df_big: The DataFrame containing "Dynkin label" and "eigen_degree" information
        steps: Number of steps to run the optimization for.
        loss_threshold: The threshold for determining convergence in the optimization.
        """
        # Set seeds for reproducibility
        np.random.seed(40)
        torch.manual_seed(40)
#         torch.set_num_threads(num_threads)  # Set this to the number of your CPU cores

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(40)

        self.data_all_fibres = data_all_fibres
        self.df_big = df_big
        self.steps = steps
        self.loss_threshold = loss_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.conn_found = []
        self.conn_not_found = []
        # Extract indices and eigenvalues, convert to numpy then to tensor
        indices = np.array(self.df_big.index)  # Extract indices from DataFrame
        eigenvalues_array = np.stack(self.df_big['eigenvalues'].values)
        self.eigenvalues_tensor = torch.from_numpy(eigenvalues_array).to(self.device)
        # size(num_flat_conn, n, Nc)
        self.indices_tensor = torch.from_numpy(indices).to(self.device)
        # size (num_flat_conn, )

        self.batch_size = min(batch_size, len(df_big))  # Adjust based on your GPU memory
        self.torch_dtype = torch_dtype
        self.np_dtype = np_dtype

    def gpu(self):
        print(f"Total number of candidate connections {len(self.df_big)}")
        tic = time.time()

        # Define batch size and process in batches
        for i in range(0, len(self.eigenvalues_tensor), self.batch_size):
            batch_indices = self.indices_tensor[i:i + self.batch_size]
            batch_eigenvalues = self.eigenvalues_tensor[i:i + self.batch_size]
            # size(self.batch_size, n, Nc)

            optimized_U, final_losses = self.optimize_matrices(batch_eigenvalues)  #

            # Threshold and store results
            for j, (U, loss) in enumerate(zip(optimized_U, final_losses)):
                idx = batch_indices[j].item()  # Convert tensor to integer index
                if loss < self.loss_threshold:
                    X_matrices = self.AUV_to_X(U, batch_eigenvalues[j])
                    self.conn_found.append((idx, X_matrices, loss))
                else:
                    self.conn_not_found.append((idx, loss))

        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # Ensure all GPU operations are complete

        toc = time.time()
        print(f"GPU operations completed in {toc - tic} seconds")

    #     return U, batch_losses.detach()
    def optimize_matrices(self, batch_eigenvalues):
        batch_size, n, Nc = batch_eigenvalues.shape
        U = torch.randn(batch_size, n, Nc, Nc, dtype=self.torch_dtype,
                        requires_grad=True, device=self.device)
        optimizer = optim.Adam([U], lr=self.lr)
        # Mask to track active optimization
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        for step in range(self.steps):
            optimizer.zero_grad()
            batch_losses = self.batch_objective(U, batch_eigenvalues)

            for i in range(batch_size):
                if active_mask[i]:
                    # perform the backprop for the active elements
                    batch_losses[i].backward(retain_graph=(i < batch_size - 1))
                    # Zero out the gradient after the backward pass
                    # for both active and inactive elements
                if (U.grad is not None) and (not active_mask[i]):
                    # For inactive elements, zero out the gradient
                    U.grad.data[i].zero_()

            optimizer.step()

            # Update active mask
            active_mask &= (batch_losses >= self.loss_threshold)
            # Only continue if loss is above threshold

            if step % 1000 == 0 or not active_mask.any():
                print(f'Step {step}, Loss: {batch_losses.detach()}')
                if not active_mask.any():
                    print("Early stopping triggered.")
                    break

        return U, batch_losses.detach()

    def batch_objective(self, U, batch_a):
        batch_size, n, Nc, _ = U.shape  # Assumes U is [batch_size, n, Nc, Nc]
        eye = torch.eye(Nc, dtype=self.torch_dtype, device=self.device).expand(batch_size, -1, -1)
        # expand from [Nc, Nc] to [batch_size, Nc, Nc].
        product = eye.clone()  # Start with the identity matrix for each item in the batch

        # Vectorized computation of the product
        # batch_a size (batch_size, n, Nc)
        for i in range(n):
            Ui = U[:, i, :, :]
            ai = batch_a[:, i, :]
            ai_diag = torch.diag_embed(ai.to(self.device))
            # Convert eigenvalues to diagonal matrices
            xi = Ui @ ai_diag @ torch.linalg.inv(Ui)
            product = product @ xi

        mm = product - eye
        # mm of size [batch_size, Nc, Nc].
        return torch.mean(torch.abs(mm) ** 2,  dim=[1, 2])

    def AUV_to_X(self, U, a):
        return [ui @ torch.diag(ai.to(self.device)) @ torch.linalg.inv(ui) for ui, ai in zip(U, a)]

    def prepare_check_irreducible(self, Nc):
        n = len(self.data_all_fibres['p_list'])
        r_symbols = symbols(f'r_1:{Nc+1}_1:{Nc+1}')
        r_matrix = Matrix(Nc, Nc, r_symbols)
        X_symbols = [[symbols(f'X{j}_{i+1}{k+1}') for i in range(Nc) for k in range(Nc)]
                     for j in range(1, n+1)]
        X_matrices = [Matrix(Nc, Nc, X_symbols[j]) for j in range(n)]

        def commutator(x, rr):
            return x * rr - rr * x

        # Compute and flatten the commutators.
        commutators = [commutator(X, r_matrix) for X in X_matrices]
        commutators_flatten = [elem for comm in commutators for elem in comm]

        coefficients_matrix = Matrix(Nc**2 * n, Nc**2,
                                     lambda i, j: commutators_flatten[i].coeff(r_symbols[j]))

        def substitution_func(numpy_X):
            subs_dict = {X_matrices[j][i, k]: numpy_X[j, i, k]
                         for j in range(n) for i in range(Nc) for k in range(Nc)}
            substituted_matrix = coefficients_matrix.subs(subs_dict)
            return substituted_matrix
        return substitution_func

    def process_results(self, X_matrices, final_loss, check_irreducible):
        numpy_matrices = [matrix.cpu().detach().numpy() for matrix in X_matrices]
        numpy_X = np.stack(numpy_matrices)
        coefficients_matrix_substituted = check_irreducible(numpy_X)
        coefficients_matrix_np = np.array(coefficients_matrix_substituted.tolist(),
                                          dtype=self.np_dtype)
        coefficients_matrix_rank = np.linalg.matrix_rank(coefficients_matrix_np, tol=1e-10)
        return coefficients_matrix_rank, final_loss

    def eigen_degree(self, eigens_list):
        def count_equalities_within_tolerance(elements, epsilon=1e-6):
            assert len(elements.shape) == 1
            count = 0
            N = len(elements)
            for i in range(N):
                next_i = (i + 1) % N
                if np.isclose(elements[i], elements[next_i], atol=epsilon).all():
                    count += 1
            return count
        return [count_equalities_within_tolerance(np.array(eigens)) for eigens in eigens_list]

    def cpu(self):
        """Handles CPU-specific tasks."""
        # Assuming processing results is the primary CPU task
        print("Processing CPU-bound tasks...")
        check_irreducible = self.prepare_check_irreducible(self.data_all_fibres['Nc'])
        if_irreducible = {}
        output_loss = {}
        output_X_matrices = {}
        tic = time.time()

        for idx, X_matrices, final_loss in self.conn_found:
            rank, loss = self.process_results(X_matrices, final_loss, check_irreducible)
            print(f"candidate {idx}")
            # dyn = tuple(tuple(arr) for arr in self.df_big.loc[i, "Dynkin_label"])
            dyn = self.df_big.loc[idx, "Dynkin_label"]
            eigen = self.df_big.loc[idx, 'eigenvalues']

            print('degrees are ', self.eigen_degree(eigen))
            print(f"Dynkin label is {dyn}")
            if rank == self.data_all_fibres['Nc']**2 - 1:
                print('\033[92mIrreducible\033[0m since rank is Nc^2 - 1')
                if_irreducible[dyn] = 'irreducible'
                output_loss[dyn] = final_loss
                output_X_matrices[dyn] = X_matrices
            else:
                print('Reducible')
                if_irreducible[dyn] = 'reducible'
                output_loss[dyn] = final_loss
                output_X_matrices[dyn] = X_matrices
            print(f'Final loss is {loss}')
            print('==================')

        for idx, final_loss in self.conn_not_found:
            print(f"candidate {idx}")
            dyn = self.df_big.loc[idx, "Dynkin_label"]
            eigen = self.df_big.loc[idx, "eigenvalues"]
            print('degrees are ', self.eigen_degree(eigen))
            print(f"Dynkin label is {dyn}")
            print(f'Final loss is {final_loss}')
            print('==================')
            if_irreducible[dyn] = 'conn_not_found'
            output_loss[dyn] = final_loss

        toc = time.time()
        print(f"CPU operations completed in {toc - tic} seconds")
        return if_irreducible, output_loss, output_X_matrices
