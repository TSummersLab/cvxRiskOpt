from mhe import simple_1d_mhe
import cvxpy as cp
import csv
import argparse


def paper_mhe(horizon, sim_steps, constraint_type, keep_init_run, solver):
    t_test = simple_1d_mhe(horizon, sim_steps, constraint_type,
                           plot_res=False, use_cpg=False, gen_cpg=False,
                           keep_init_run=keep_init_run, solver=solver)
    t_test_codegen = simple_1d_mhe(horizon, sim_steps, constraint_type,
                                   plot_res=False, use_cpg=True, gen_cpg=True,
                                   keep_init_run=keep_init_run, solver=solver)
    if constraint_type is None:
        constraint_type = "none"
    with open('paper_mhe_cstr_{}_N_{}_solver_{}.csv'.format(constraint_type, horizon, solver), 'a', newline='') as file:
        writer = csv.writer(file)
        for t in range(len(t_test_codegen)):
            writer.writerow([t_test[t], t_test_codegen[t], solver])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Temp Regulator.')
    parser.add_argument('--horizon', type=int, default=10, help='MHE horizon')
    parser.add_argument('--sim_steps', type=int, default=200, help='Number fo simulation steps')
    parser.add_argument('--constraint_type', type=str, default="gauss", help='Type of constraints')
    parser.add_argument('--keep_init_run', type=int, default=0,
                        help='Boolean to keep the initial run (0 or 1) (not when calling generated code though)')
    parser.add_argument('--solver', type=str, default=cp.OSQP,
                        help="CVXPY solver ['CLARABEL', 'CVXOPT', 'ECOS', 'OSQP', 'SCS', ...]")

    args = parser.parse_args()
    horizon = args.horizon
    sim_steps = args.sim_steps
    constraint_type = args.constraint_type
    solver = args.solver
    keep_init_run = bool(args.keep_init_run)

    paper_mhe(horizon, sim_steps, constraint_type, keep_init_run, solver)


