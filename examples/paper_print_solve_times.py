import csv
import matplotlib.pyplot as plt
import os
import numpy as np


def get_run_details(filename):
    # Split the filename into parts based on the underscore
    parts = filename.split("_")

    # Find the index of 'N' and 'solver' in the list
    n_index = parts.index('N')
    solver_index = parts.index('solver')

    # Get the values after 'N' and 'solver'
    n_value = parts[n_index + 1]
    solver_value = parts[solver_index + 1].split('.')[0]  # Remove the extension

    return n_value, solver_value


def compute_averages(filename, verbose=True):
    # Initialize sums and count
    times = [[], [], []]
    count = 0

    # get solver name
    num_samp, solver = get_run_details(filename)

    # add times
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # next(reader)  # Skip the header row

        for row in reader:
            # Add the values of the first three columns to the sums
            for i in range(len(row)-1):
                times[i].append(float(row[i]) * 1000)  # convert time to ms and append it
            count += 1
    num_cols = len(row)
    # compute average and variance of the times
    times = np.array(times[:num_cols-1])
    averages = np.mean(times, axis=1)
    variances = np.var(times, axis=1)

    round_digits_avrg = 3
    round_digits_var = 4
    avrg = [round(a, round_digits_avrg) for a in averages]
    var = [round(v, round_digits_var) for v in variances]
    std_dev = [round(np.sqrt(v), round_digits_var) for v in variances]

    # Print the averages
    if verbose:
        print("Average solve times with {} for {} samples:".format(solver, num_samp))
        print("TOTAL COUNT: ", count)
    if num_cols == 4:
        latex_summary = (str(num_samp) +
                         " & " + str(avrg[0]) + "(" + str(std_dev[0]) + ")" +
                         " & " + str(avrg[1]) + "(" + str(std_dev[1]) + ")" +
                         " & " + str(avrg[2]) + "(" + str(std_dev[2]) + ")" +
                         " & " + str(solver))
    elif num_cols == 3:
        latex_summary = (str(num_samp) +
                         " & " + str(avrg[0]) + "(" + str(std_dev[0]) + ")" +
                         " & " + str(avrg[1]) + "(" + str(std_dev[1]) + ")" +
                         " & " + str(solver))
    else:
        raise NotImplementedError("Only 3 or 4 column modes implemented")
    if verbose:
        print("Summary: \n", latex_summary)
        print("    cvxpy (ms): ", avrg[0], "std: ", std_dev[0])
        print("    toolbox (ms): ", avrg[1], "std: ", std_dev[1])
        if num_cols == 4:
            print("    toolbox+gen (ms): ", avrg[2], "std: ", std_dev[2])
    else:
        print(latex_summary)


if __name__ == "__main__":
    N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
    solver_list = ['CLARABEL', 'ECOS']
    for solver in solver_list:
        for N in N_list:
            rel_path = os.path.join("paper_simulations", "dr_portfolio")
            if not os.path.exists(rel_path):
                rel_path = "."
            filename = os.path.join(rel_path, "paper_dr_portfolio_opt_results_N_{}_solver_{}.csv".format(N, solver))
            compute_averages(filename, verbose=False)

    N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    solver_list = ["CLARABEL", "OSQP", "SCS"]
    for solver in solver_list:
        for N in N_list:
            rel_path = os.path.join("paper_simulations", "mpc")
            if not os.path.exists(rel_path):
                rel_path = "."
            filename = os.path.join(rel_path, "paper_temp_reg_mpc_N_{}_solver_{}.csv".format(N, solver))
            compute_averages(filename, verbose=False)

    solver_list = ["CLARABEL", "OSQP", "ECOS"]
    N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for solver in solver_list:
        for N in N_list:
            rel_path = os.path.join("paper_simulations", "mhe")
            if not os.path.exists(rel_path):
                rel_path = "."
            filename = os.path.join(rel_path, "paper_mhe_cstr_moment_N_{}_solver_{}.csv".format(N, solver))
            compute_averages(filename, verbose=False)
