import csv
import matplotlib.pyplot as plt

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


def compute_averages(filename):
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

    if count != int(num_samp):
        print("WARNING: counted rows and num samples in file name don't match. Averages are divided by count though.")
        print("TOTAL COUNT: ", count)
    # Print the averages
    print("Average solve times with {} for {} samples:".format(solver, num_samp))
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
    print("Summary: \n", latex_summary)
    print("    cvxpy (ms): ", avrg[0], "std: ", std_dev[0])
    print("    toolbox (ms): ", avrg[1], "std: ", std_dev[1])
    if num_cols == 4:
        print("    toolbox+gen (ms): ", avrg[2], "std: ", std_dev[2])


if __name__ == "__main__":
    # filename = "paper_dr_portfolio_opt_results_N_10_solver_CLARABEL.csv"
    # compute_averages(filename)
    # filename = "paper_dr_portfolio_opt_results_N_10_solver_ECOS.csv"
    # compute_averages(filename)
    # filename = "paper_dr_portfolio_opt_results_N_10_solver_SCS.csv"
    # compute_averages(filename)
    #
    filename = "paper_dr_portfolio_opt_results_N_15_solver_CLARABEL.csv"
    compute_averages(filename)
    filename = "paper_dr_portfolio_opt_results_N_15_solver_ECOS.csv"
    compute_averages(filename)
    filename = "paper_dr_portfolio_opt_results_N_15_solver_SCS.csv"
    compute_averages(filename)
    #
    # filename = "paper_dr_portfolio_opt_results_N_100_solver_CLARABEL.csv"
    # compute_averages(filename)
    # filename = "paper_dr_portfolio_opt_results_N_100_solver_ECOS.csv"
    # compute_averages(filename)
    #
    # filename = "paper_dr_portfolio_opt_results_N_200_solver_ECOS.csv"
    # compute_averages(filename)

    # ######################### #

    filename = "paper_temp_reg_mpc_N_40_solver_CLARABEL.csv"
    compute_averages(filename)
    filename = "paper_temp_reg_mpc_N_40_solver_SCS.csv"
    compute_averages(filename)
    filename = "paper_temp_reg_mpc_N_40_solver_OSQP.csv"
    compute_averages(filename)

    filename = "paper_temp_reg_mpc_N_80_solver_CLARABEL.csv"
    compute_averages(filename)
    filename = "paper_temp_reg_mpc_N_80_solver_SCS.csv"
    compute_averages(filename)
    filename = "paper_temp_reg_mpc_N_80_solver_OSQP.csv"
    compute_averages(filename)


