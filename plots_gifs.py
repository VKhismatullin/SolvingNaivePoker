from experiment_code import * 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio


def exists_equil_modified(list_, N):
    """
    Given the list of derivatives, return the inf winner of the game (batched along 1 axis)
        Note that the inf winner is the player who can win infinitely much by 
        overbetting at some stage of the game. 

    Parameters:
        list_ - a list of consecutive derivatives (from the end to beginning), size T x N

    Returns:
        result - list of size N, containing the inf winner of the game 
        1 for reciever-taker, -1 for action-taker, 0 if equilibrium exists 
    """
    result = []
    for i in range(N):
        outcome = 0  # assume all is good for column i
        # iterate over the list in order
        for idx, x in enumerate(list_):
            if idx % 2 == 0:  # even-indexed element: expected positive
                if x[0, i] <= 0:
                    outcome = 1   # violation: even element not positive
                    break
            else:  # odd-indexed element: expected negative
                if x[0, i] >= 0:
                    outcome = -1  # violation: odd element not negative
                    break
        result.append(outcome)
    return result


def run_experiment_in_batches(experiment_func, gamma_r, gamma_a,
                              sigma_r, sigma_a, T, max_batch_size=10_000):
    """
    Evaluates `experiment_func(x1, x2, x3, x4)` in batches over the
    two flattened arrays, and two constants.
    
    Returns the result of the batched call of the same size as x3_flat (and x4_flat).
    """
    N = len(sigma_a)
    results = np.zeros(N)  # placeholder
    
    i, start = 0, 0
    while start < N:
        end = min(start + max_batch_size, N)
        batch_size = end - start
        
        # Construct chunk arrays for x1, x2, x3, x4
        batch_x1 = gamma_r[start:end].reshape(1, -1)
        batch_x2 = gamma_a[start:end].reshape(1, -1)
        batch_x3 = sigma_r[start:end].reshape(1, -1)
        batch_x4 = sigma_a[start:end].reshape(1, -1)
        # Evaluate your experiment
        batch_result = experiment_func(T, batch_size, batch_x1, batch_x2, batch_x3, batch_x4)
        
        # Save the chunk result
        results[start:end] = batch_result
        
        start = end
        i+=1
    
    return results


def plot_given_params(params, T, N, fig_params=None, batch_size=10_000):
    """
        params: {'axis_name_i': {'role':<role>, 'value':<values>}}
            role  -- can take x_axis, y_axis, const
            value -- is either the bound in the first two cases or a const
        
            
        Main idea of this dict is to easily manipulate what to plot on x-axis vs y-axis,
                simplifying the index logic

        T - int - length of the game (same for every within-batch element)
        N - int - size of the batch 
    """
    x_role, y_role, other_params = None, None, []
    grid = {}
    for name in params.keys():
        if params[name]['role'] == 'const':
            grid[name] = params[name]['value']
            other_params.append(name)
        else:
            grid[name] = np.linspace(params[name]['value'][0], params[name]['value'][1], N) 
            if params[name]['role'] == 'x_axis':
                x_role = name
            elif params[name]['role'] == 'y_axis':
                y_role = name
            else:
                raise TypeError('Wrong Params dict')

    # Key simplifying idea: name axis and mix their roles based on the dict, only then create the grid
    x_grid, y_grid = np.meshgrid(grid[x_role], grid[y_role], indexing="ij")
    x_flat, y_flat = x_grid.ravel(), y_grid.ravel()
    
    params_to_pass = {x_role: x_flat, y_role: y_flat, 'T':T}
    for name in other_params:
        params_to_pass[name] = np.ones_like(x_flat) * grid[name]
    
    # We run the experiment for derivatives only and then convert to to the winning player
    solver_to_run = lambda T, N, x1, x2, x3, x4: exists_equil_modified(
                solver(T, N, x1, x2, x3, x4, return_strats=False), 
                batch_size)
    bool_result_2d = run_experiment_in_batches(solver_to_run, max_batch_size=batch_size, **params_to_pass)
    
    res = {x_role: x_flat, y_role: y_flat, 'output': bool_result_2d}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(res)

    # Pivot the DataFrame for 2D visualization
    pivot_table = df.pivot(index=y_role, columns=x_role, values='output')

    cmap = mcolors.ListedColormap(['white', 'grey', 'black'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Visualize the pivot table using the custom colormap and norm
    if fig_params:
        plt.figure(figsize=(fig_params['m'], fig_params['n']))
    else:
        plt.figure(figsize=(9, 7))
    plt.imshow(pivot_table, extent=[x_flat[0], x_flat[-1], y_flat[0], y_flat[-1]], 
               origin='lower', aspect='auto', cmap=cmap, norm=norm)
    plt.xlabel(x_role)
    plt.ylabel(y_role)
    name_1, name_2 = other_params[0], other_params[1]
    plt.title(f'Equilibrium plot (T={T}, {name_1}={grid[name_1]:.2f}, {name_2}={grid[name_2]:.2f})')
    
    if fig_params['to_plot']:
        plt.savefig(f"TEMP_PLOTS/{fig_params['name']}.jpg")



def get_gif_given_params(list_params, list_T, N, gif_path=None, fig_params=None, duration=.2, batch_size=10_000):
    # Initialize the name
    temp_gif_params  = {}
    if gif_path is None:
        gif_path = 'TEMP_GIF.gif'
    
    # Plot the graphs
    for i in range(len(list_params)):
        if fig_params is None:
            temp_gif_params['to_plot'] = True
            temp_gif_params['name'] = f'temp_image_for_gif{i}'
            temp_gif_params['m'] = 9 
            temp_gif_params['n']= 7
        else:
            temp_gif_params = fig_params.copy()
            temp_gif_params['name'] = temp_gif_params['name'] + f'{i}'
        
        plot_given_params(list_params[i], list_T[i], N, fig_params=temp_gif_params, batch_size=10_000)
    
    # Collect the images into a gif
    images = []
    for i in range(len(list_params)):
        if fig_params is None:
            filename = f"TEMP_PLOTS/temp_image_for_gif{i}.jpg"
        else:
            filename = f"TEMP_PLOTS/" + fig_params['name'] + f'{i}.jpg'
        images.append(imageio.imread(filename))
    # Save as GIF with duration-milisecond delay between frames
    imageio.mimsave(gif_path, images, duration=duration)