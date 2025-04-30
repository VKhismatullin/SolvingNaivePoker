import numpy as np
import pandas as pd


def compute_phi(T, sigma_Z_init, sigma_S_init, sigma_r, sigma_a):
    """
    Compute phi_r and phi_a arrays based on given formulas.
    
    Parameters:
        T (int): Number of time steps.
        N (int): Vectorization dimension.
        sigma_Z_init (float): Initial value for sigma^Z.
        sigma_S_init (float): Initial value for sigma^S.
        sigma_r (float): Constant sigma_r value.
        sigma_a (float): Constant sigma_a value.
        
    Returns:
        phi_r (ndarray): Array of shape (T, T, N).
        phi_a (ndarray): Array of shape (T, T, N).
    """
    N = sigma_r.shape[-1]
    # Initialize phi_r and phi_a with zeros
    phi_r = np.zeros((T+1, T+1, N))
    phi_a = np.zeros((T+1, T+1, N))
    
    # Initialize sigma_Z and sigma_S arrays
    sigma_Z = np.zeros((T+1, N))
    sigma_S = np.zeros((T+1, N))
    sigma_Z[0, :] = sigma_Z_init
    sigma_S[0, :] = sigma_S_init
    
    # Boundary conditions
    phi_r[0, :] = 0  # phi_r(0, j) = 0 for all j
    phi_a[0, :] = 0  # Initialize to 0
    phi_a[0, 0] = 1  # phi_a(0, 0) = 1
    
    # Evolutionary formulas for sigma_Z and sigma_S
    for t in range(1, T+1):
        sigma_Z[t] = np.sqrt((sigma_Z[t-1]**2 * sigma_r**2) / (sigma_Z[t-1]**2 + sigma_r**2))
        sigma_S[t] = np.sqrt((sigma_S[t-1]**2 * sigma_a**2) / (sigma_S[t-1]**2 + sigma_a**2))
        
        phi_r[t, t-1] = (sigma_Z[t-1]**2) / (sigma_Z[t-1]**2 + sigma_r**2)
        phi_a[t, t] = (sigma_S[t-1]**2) / (sigma_S[t-1]**2 + sigma_a**2)
        
        for j in range(1, t):
            phi_r[t, j-1] = phi_r[t-1, j-1] * (sigma_r**2 / (sigma_Z[t-1]**2 + sigma_r**2))
        for j in range(t):    
            phi_a[t, j] = phi_a[t-1, j] * (sigma_a**2 / (sigma_S[t-1]**2 + sigma_a**2))
    
    return phi_r, phi_a, sigma_Z, sigma_S


def find_latest_optimal(T, N, phi_r, phi_a, gamma_r, gamma_a):
    """
    Compute w_r_a, w_r_r, w_a_a, w_a_r arrays based on given formulas.
    
    Parameters:
        T (int): Number of time steps.
        N (int): Vectorization dimension.
        phi_r (ndarray): Precomputed phi_r array.
        phi_a (ndarray): Precomputed phi_a array.
        gamma_r (float): Constant gamma_r value.
        n_Z_t (float): Constant n^Z_t value.
        
    Returns:
        w_r_a, w_r_r, w_a_a, w_a_r (ndarrays): Arrays of shape (T+1, T+1, T+1, 2, N).
    """
    w_r_a = np.zeros((T + 1, T + 1, T + 1, 2, N)) #+ np.nan
    w_r_r = np.zeros((T + 1, T + 1, T + 1, 2, N)) #+ np.nan
    w_a_a = np.zeros((T + 1, T + 1, T + 1, 2, N)) #+ np.nan
    w_a_r = np.zeros((T + 1, T + 1, T + 1, 2, N)) #+ np.nan
    n_Z_t = -1 / gamma_a + gamma_r * phi_a[T, T]
    
    w_r_a[T, T, :, 1] = -gamma_r * phi_a[T,:]
    w_r_r[T, T, :, 1] = 0
    w_a_a[T, T, :, 0] = -(phi_a[T, T] / (2 * n_Z_t)) * phi_a[T, :]
    w_a_r[T, T, :, 0] = ((1 + gamma_r * phi_a[T, T]) / n_Z_t) * phi_r[T, :]
        
    
    return w_r_a, w_r_r, w_a_a, w_a_r

def reduce_final_r1_to_r0(t, T, N, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r, nu_r):
    """
    Perform updates to reduce r1 to r0 as per the given formulas.

    Parameters:
        T (int): Number of time steps.
        N (int): Vectorization dimension.
        eta_r (ndarray): Array of shape (T+1, T+1, T+1, N).
        eta_a (ndarray): Array of shape (T+1, T+1, T+1, N).
        w_r_a (ndarray): Array of shape (T+1, T+1, T+1, 2, N).
        w_r_r (ndarray): Array of shape (T+1, T+1, T+1, 2, N).
        w_a_a (ndarray): Array of shape (T+1, T+1, T+1, 2, N).
        w_a_r (ndarray): Array of shape (T+1, T+1, T+1, 2, N).
        nu_r (ndarray): Array of shape (T+1, T+1, T+1, 2, N).
        delta_r (ndarray): Array of shape (T+1, T+1).

    Returns:
        Updated eta_r, w_r_a, nu_r, w_r_r.
    """
    delta_r_t = w_r_a[t, t, t, 1, :] # delta_r(t, t)

    # Update eta_r
    eta_r[t, t, t, 0] = delta_r_t * eta_a[t, t, t, 0]

    # Update w_r_a
    w_r_a[t, t, :, 0] = w_r_a[t, t, :, 1] + delta_r_t * w_a_a[t, t, :, 0]

    # Update nu_r
    nu_r[t, t, t, 0] = nu_r[t, t, t, 1]

    # Update w_r_r
    w_r_r[t, t, :, 0] = w_r_r[t, t, :, 1] + delta_r_t * w_a_r[t, t, :, 0]

    return eta_r, w_r_a, nu_r, w_r_r

def nu_hat_r(i, t, nu_r):
    return np.sum(nu_r[i, t, t: i+1, 0, :], axis=0)  
def nu_hat_a(i, t, nu_a):
    return np.sum(nu_a[i, t, t: i, 0, :], axis=0)
def eta_hat_a(i, t, eta_a):
    return np.sum(eta_a[i, t, t: i+1, 0, :], axis=0)
def eta_hat_r(i, t, eta_r):
    return np.sum(eta_r[i, t, t: i+1, 0, :], axis=0) 

def nu_over_r(i, t, nu_r):
    return np.sum(nu_r[i, t, t: i+1, 1, :], axis=0)    
def nu_over_a(i, t, nu_a):
    return np.sum(nu_a[i, t, t: i, 1, :], axis=0)
def eta_over_a(i, t, eta_a):
    return np.sum(eta_a[i, t, t+1: i+1, 1, :], axis=0)
def eta_over_r(i, t, eta_r):
    return np.sum(eta_r[i, t, t+1: i+1, 1, :], axis=0)    


# 44-47

#### 46 nu neverna t ili t+1
def find_optimal_r(t, T, N, gamma_r, gamma_a, w_r_r, w_r_a, w_a_r, w_a_a, nu_r, nu_a, eta_r, eta_a, phi_a):
    """
    Update the optimal weights based on the given formulas.

    Parameters:
        T (int): Total number of time steps.
        gamma_r (float): Constant parameter for risk weights.
        gamma_a (float): Constant parameter for auxiliary weights.
        w_r_r, w_r_a, w_a_r, w_a_a (ndarray): Weight arrays of shape (T+1, T+1, T+1, 2, N).
        nu_r, nu_a (ndarray): Nu arrays of shape (T+1, T+1, T+1, 2, N).
        eta_r, eta_a (ndarray): Eta arrays of shape (T+1, T+1, T+1, N).
        phi_a (ndarray): Array of shape (T, N).

    Returns:
        Updated weights and nu arrays.
    """
    # Calculate n^r_t
    sum_r = np.sum(
        (1 / gamma_r) * w_r_r[t+1:T+1, t+1, t, 0]**2 - (1 / gamma_a) * w_r_a[t+1:T+1, t+1, t, 0]**2,
        axis = 0
    )
    n_r_t = (1 / gamma_r) + sum_r
    
    # Update nu_r(t, t, t, 1)
    sum_w_r = np.sum(w_a_r[t+1:T+1, t+1, t, 0] + w_r_r[t+1:T+1, t+1, t, 0], axis=0)
    # 
    temp = []
    for i in range(t+1, T+1):
        temp.append(w_r_r[i, t+1, t, 0] * nu_hat_r(i, t+1, nu_r))
    sum_nu_r = (1 / gamma_r) * np.sum(temp, axis=0)
    
    temp = []
    for i in range(t+1, T+1):
        temp.append(w_r_a[i, t+1, t, 0] * nu_hat_a(i, t+1, nu_a))
    sum_nu_a = (1 / gamma_a) * np.sum(temp, axis=0)

    nu_r[t, t, t, 1] = -1 / n_r_t * (1 + sum_w_r - sum_nu_r + sum_nu_a)

    # Update w_r^a(t, t, j, 1)
    for j in range(t+1):
        # 46-2
        temp = []
        for i in range(t+1, T+1):
            temp.append(w_r_r[i, t+1, t, 0] * 
                        (w_r_a[i, t+1, j, 0] + eta_hat_r(i, t+1, eta_r) * phi_a[t, j]))

        sum_w_r_a = (1 / gamma_r) * np.sum(temp, axis=0) 
        # 46-3
        temp = []
        for i in range(t+1, T+1):
            temp.append(w_r_a[i, t+1, t, 0] * 
                        (w_a_a[i, t+1, j, 0] + eta_hat_a(i, t+1, eta_a) * phi_a[t, j]))
        
        sum_w_a_a = (1 / gamma_a) * np.sum(temp, axis=0) 
        # Finish 2
        w_r_a[t, t, j, 1] = -1 / n_r_t * (
            (1 + sum_w_r) * phi_a[t, j] - sum_w_r_a + sum_w_a_a
        )

        # Update w_r^r(t, t, j, 1)
        sum_w_rr = np.sum(
            (1 / gamma_r) * w_r_r[t+1:T+1, t+1, t, 0] * w_r_r[t+1:T+1, t+1, j, 0], axis=0
        )
        sum_w_ar = np.sum(
            (1 / gamma_a) * w_r_a[t+1:T+1, t+1, t, 0] * w_a_r[t+1:T+1, t+1, j, 0], axis=0
        )
        w_r_r[t, t, j, 1] = -1 / n_r_t * (-sum_w_rr + sum_w_ar)

    return w_r_r, w_r_a, nu_r, n_r_t


def reduce_step_1(t, k, nu_r, nu_a, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r):
    """
    Update weights and nu arrays based on the given formulas.

    Parameters:
        t (int): Current time step.
        k (int): Step index.
        nu_r, nu_a (ndarray): Nu arrays of shape (T+1, T+1, T+1, 2, N).
        eta (ndarray): Eta array of shape (T+1, T+1, T+1, N).
        w_r_a, w_r_r, w_a_a, w_a_r (ndarray): Weight arrays of shape (T+1, T+1, T+1, 2, N).

    Returns:
        Updated nu and weight arrays.
    """

    # Update nu_a and nu_r
    nu_a[t, k - 1, k - 1, 1] = w_a_r[t, k, k - 1, 0] * nu_r[k - 1, k - 1, k - 1, 1]
    nu_r[t, k - 1, k - 1, 1] = w_r_r[t, k, k - 1, 0] * nu_r[k - 1, k - 1, k - 1, 1]

    # Update nu for j > k-1
    nu_a[t, k - 1, k:, 1] = nu_a[t, k, k:,0]
    nu_r[t, k - 1, k:, 1] = nu_r[t, k, k:,0]
    
    # Update eta
    eta_a[t, k - 1, :, 1] = eta_a[t, k, :,0]
    eta_r[t, k - 1, :, 1] = eta_r[t, k, :,0]

    # Update weight arrays
    w_a_a[t, k - 1, :, 1] = w_a_a[t, k, :, 0] + w_a_r[t, k, k - 1, 0] * w_r_a[k - 1, k - 1, :, 1]
    w_a_r[t, k - 1, :, 1] = w_a_r[t, k, :, 0] + w_a_r[t, k, k - 1, 0] * w_r_r[k - 1, k - 1, :, 1]
    w_r_a[t, k - 1, :, 1] = w_r_a[t, k, :, 0] + w_r_r[t, k, k - 1, 0] * w_r_a[k - 1, k - 1, :, 1]
    w_r_r[t, k - 1, :, 1] = w_r_r[t, k, :, 0] + w_r_r[t, k, k - 1, 0] * w_r_r[k - 1, k - 1, :, 1]

    return nu_r, nu_a, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r


# 48-51
## Перепроверить бы те формулы по-хорошему
def find_optimal_a(t, T, N, gamma_r, gamma_a, w_r_r, w_r_a, w_a_r, w_a_a, nu_r, nu_a, eta_r, eta_a, phi_r):
    """
    Update the weights of optimal a based on the given formulas.

    Parameters:
        T (int): Total number of time steps.
        gamma_r (float): Constant parameter for risk weights.
        gamma_a (float): Constant parameter for auxiliary weights.
        w_r_r, w_r_a, w_a_r, w_a_a (ndarray): Weight arrays of shape (T+1, T+1, T+1, 2, N).
        nu_r, nu_a (ndarray): Nu arrays of shape (T+1, T+1, T+1, 2, N).
        eta_r, eta_a (ndarray): Eta arrays of shape (T+1, T+1, T+1, N).
        phi_a (ndarray): Array of shape (T, N).

    Returns:
        Updated weights and nu arrays.
    """
    
    
    delta_t_t = w_r_a[t, t, t, 1, :]
    #### Compute n^a_t
    n_a = (
        -(1 / gamma_a)
        + np.sum((1 / gamma_r) *  w_r_a[t+1:T,t+1, t, 0, :]**2 - (1 / gamma_a) * w_a_a[t+1:T,t+1, t, 0, :]**2, axis=0)
        + (1 / gamma_r) * delta_t_t**2
    )
    
    #### Compute eta(t, t, t, 0)
    
    sum_w_delta = delta_t_t + np.sum(w_r_a[t+1:T,t+1, t, 0, :] + w_a_a[t+1:T,t+1, t, 0, :], axis=0)
    
    # First find the eta_overlines:
    temp_r = []
    # Element for i= t is empty
    for i in range(t+1, T+1):
        temp_r.append(w_r_a[i, t+1, t, 0, :] * eta_over_r(i, t, eta_r))
    
    temp_a = []
    for i in range(t+1, T+1):
        temp_a.append(w_a_a[i, t+1, t, 0, :] * eta_over_a(i, t, eta_a))
    
    eta_a[t, t, t, 0] = -1 / n_a * (
        1
        + sum_w_delta
        - (1 / gamma_r) * (np.sum(temp_r, axis=0) + delta_t_t * eta_over_r(t, t, eta_r))
        + (1 / gamma_a) * np.sum(temp_a, axis=0)
    )

    #### Update w_a^a(t, t, j, 0)
    for j in range(t):
        w_a_a[t, t, j, 0] = -1 / n_a * (
            -(1 / gamma_r) * np.sum(w_r_a[t+1:T+1, t+1, t, 0, :] * w_r_a[t+1:T+1, t, j, 1], axis=0) 
            - (1 / gamma_r) * delta_t_t * w_r_a[t, t, j, 1]
            + (1 / gamma_a) * np.sum(w_a_a[t+1:T+1, t+1, t, 0, :] * w_a_a[t+1:T+1, t, j, 1], axis=0)
        )
        
    temp_r = []
    for i in range(t+1, T+1):
        # temp_r.append(w_r_a[i, t+1, t, 0, :] *nu_over_r(i, t, nu_r))
        temp_r.append(nu_over_r(i, t, nu_r))

    temp_a = []
    for i in range(t+1, T+1):
        # temp_a.append(w_a_a[i, t+1, t, 0, :] * nu_over_r(i, t, nu_a))
        temp_a.append(nu_over_a(i, t, nu_a))
        
    temp_r, temp_a = np.array(temp_r), np.array(temp_a)

    # First find the nu overlines        
    other = nu_over_r(t, t, nu_r)
    #### Update w_a^r(t, t, j, 0)
    for j in range(t):
        

        w_a_r[t, t, j, 0] = -1 / n_a * (
            -np.sum(w_r_a[t+1:T+1, t+1, t, 0, :] * (w_r_r[t+1:T+1, t, j, 1] + phi_r[t, j].reshape(1, -1) * temp_r),
                                   axis=0) * (1 / gamma_r) 
            - (1 / gamma_r) * delta_t_t * (w_r_r[t, t, j, 1] + phi_r[t, j].reshape(1, -1) * other)
            + 
             np.sum(w_a_a[t+1:T+1, t+1, t, 0, :] * (w_a_r[t+1:T+1, t, j, 1] + phi_r[t, j].reshape(1, -1) * temp_a),
                                    axis=0) * (1 / gamma_a)
        )
    

    return w_a_r, w_a_a, eta_a, n_a


def reduce_step_2(t, k, nu_r, nu_a, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r):
    """
    Update weights and nu arrays based on the given formulas.

    Parameters:
        t (int): Current time step.
        k (int): Step index.
        nu_r, nu_a (ndarray): Nu arrays of shape (T+1, T+1, T+1, 2, N).
        eta (ndarray): Eta array of shape (T+1, T+1, T+1, N).
        w_r_a, w_r_r, w_a_a, w_a_r (ndarray): Weight arrays of shape (T+1, T+1, T+1, 2, N).

    Returns:
        Updated nu and weight arrays.
    """
    # k_ = k - 1

    # Update eta_a and eta_r
    eta_a[t, k, k, 0] = w_a_a[t, k+1, k, 0,: ] * eta_a[k, k, k, 0]
    eta_r[t, k, k, 0] = w_r_a[t, k+1, k, 0,: ] * eta_a[k, k, k, 0]
    # Update eta
    eta_a[t, k, k+1:, 0] = eta_a[t, k, k+1:,1]
    eta_r[t, k, k+1:, 0] = eta_r[t, k, k+1:,1]

    # Update nu
    nu_r[t, k, :, 0] = nu_r[t, k, :, 1]
    nu_a[t, k, :, 0] = nu_a[t, k, :, 1]

    # Update weight arrays
    w_a_a[t, k, :, 0] = w_a_a[t, k, :, 1] + w_a_a[t, k+1, k, 0,: ] * w_a_a[k, k, :, 0]
    w_a_r[t, k, :, 0] = w_a_r[t, k, :, 1] + w_a_a[t, k+1, k, 0,: ] * w_a_r[k, k, :, 0]
    w_r_a[t, k, :, 0] = w_r_a[t, k, :, 1] + w_r_a[t, k+1, k, 0,: ] * w_a_a[k, k, :, 0]
    w_r_r[t, k, :, 0] = w_r_r[t, k, :, 1] + w_r_a[t, k+1, k, 0,: ] * w_a_r[k, k, :, 0]

    return nu_r, nu_a, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r



def solver(T, N, gamma_r, gamma_a, sigma_r, sigma_a, sigma_Z_init=1., sigma_S_init = 1., return_strats=False):
    """
    Given the parameters of the game in batches, find the solution for each game
    
    Parameters:
        T - int - length of the game (same for every within-batch element)
        N - int - size of the batch
        sigma_Z_init, sigma_S_init - variances in prior distribution of strengths  

        other params are 1 X N vectors that contain the intialization params for each batch

    Returns:
        (Derivatives, coefficients)
        Derivatives - a list of consecutive derivatives (from the end to beginning), size T x N
        coefficients - all significant coefficients, defining the actions of players at each step
                (only if the return_strats flag is true) 
    """

    
    Derivatives = []
    # Step 1-3 intitialize constant values
    phi_r, phi_a, sigma_Z, sigma_S = compute_phi(T, sigma_Z_init, sigma_S_init, sigma_r, sigma_a)


    # Step 5 - first optimals
    w_r_a, w_r_r, w_a_a, w_a_r = find_latest_optimal(T, N, phi_r, phi_a, gamma_r, gamma_a)

    eta_r = np.zeros((T + 1, T + 1, T + 1, 2, N)) #+ np.nan
    eta_a = np.zeros((T + 1, T + 1, T + 1, 2, N)) #+ np.nan
    nu_r = np.zeros((T + 1, T + 1, T + 1, 2, N)) #+ np.nan
    nu_a = np.zeros((T + 1, T + 1, T + 1, 2, N)) #+ np.nan

    nu_r[T, T, T, 1] = gamma_r
    for t in range(T+1):
        eta_r[t, t, t, 1] =  0
        nu_a[t, t, t, 0] = 0

    eta_a[T, T, T, 0] =  - 1 / (-1 / gamma_a + gamma_r * phi_a[T, T])


    # Step 6 - final reduction for the first step:
    eta_r, w_r_a, nu_r, w_r_r = reduce_final_r1_to_r0(T, T, N, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r, nu_r)

    # Now the inner cycle:
    for t_0 in range(T)[::-1]:
        # Step 10: calculate optimal r
        w_r_r, w_r_a, nu_r, second_der = \
                find_optimal_r(t_0, T, N, gamma_r, gamma_a, w_r_r, w_r_a, w_a_r, w_a_a, nu_r, nu_a, eta_r, eta_a, phi_a)
        Derivatives.append(second_der)

        # Step 12: calculate optimal reduced forms indexed by 1
        for t in range(t_0+1, T+1)[::-1]:
            nu_r, nu_a, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r =\
                reduce_step_1(t, t_0+1, nu_r, nu_a, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r)
        
        # Step 14: calculate optimal a
        w_a_r, w_a_a, eta_a, second_der =\
                find_optimal_a(t_0, T, N, gamma_r, gamma_a, w_r_r, w_r_a, w_a_r, w_a_a, nu_r, nu_a, eta_r, eta_a, phi_r)
        Derivatives.append(second_der)

        # Step 16: calculate optimal reduced forms indexed by 0
        for t in range(t_0+1, T+1)[::-1]:
            nu_r, nu_a, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r =\
                reduce_step_2(t, t_0, nu_r, nu_a, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r)


        # Step 18: finally, reduce to r(t_0,t)0,0)
        eta_r, w_r_a, nu_r, w_r_r = reduce_final_r1_to_r0(t_0, T, N, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r, nu_r)

    if return_strats: 
        return Derivatives, (nu_r, nu_a, eta_r, eta_a, w_r_a, w_r_r, w_a_a, w_a_r)
    else:
        return Derivatives 