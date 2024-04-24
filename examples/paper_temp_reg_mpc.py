from cclp_mpc import temp_mpc_regulator_time_varying_constraints
import cvxpy as cp
import csv
import argparse


def paper_temp_reg_mpc(horizon, inputs_per_hr, total_days, keep_init_run, solver):
    t_test = temp_mpc_regulator_time_varying_constraints(horizon, inputs_per_hr, total_days,
                                                         plot_res=False, use_cpg=False, gen_cpg=False,
                                                         keep_init_run=keep_init_run, solver=solver)
    t_test_codegen = temp_mpc_regulator_time_varying_constraints(horizon, inputs_per_hr, total_days,
                                                                 plot_res=False, use_cpg=True, gen_cpg=True,
                                                                 keep_init_run=keep_init_run, solver=solver)
    with open('paper_temp_reg_mpc_N_{}_solver_{}.csv'.format(horizon, solver), 'a', newline='') as file:
        writer = csv.writer(file)
        for t in range(len(t_test_codegen)):
            writer.writerow([t_test[t], t_test_codegen[t], solver])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Temp Regulator.')
    parser.add_argument('--horizon', type=int, default=40, help='MPC horizon')
    parser.add_argument('--inputs_per_hr', type=int, default=4, help='Number of inputs per hour for MPC')
    parser.add_argument('--total_days', type=int, default=5, help='Number of days to run MPC for')
    parser.add_argument('--keep_init_run', type=int, default=0,
                        help='Boolean to keep the initial run (0 or 1) (not when calling generated code though)')
    parser.add_argument('--solver', type=str, default=cp.OSQP,
                        help="CVXPY solver ['CLARABEL', 'CVXOPT', 'ECOS', 'OSQP', 'SCS', ...]")

    args = parser.parse_args()
    horizon = args.horizon
    inputs_per_hr = args.inputs_per_hr
    total_days = args.total_days
    solver = args.solver
    keep_init_run = bool(args.keep_init_run)

    paper_temp_reg_mpc(horizon, inputs_per_hr, total_days, keep_init_run, solver)


