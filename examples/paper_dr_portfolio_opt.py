import cvxpy as cp
import csv
from wass_risk_opt_esfahani_portfolio_codegen import esfahani_portfolio_codegen
import argparse


def paper_dr_portfolio_opt(num_samples, num_sim, gen_code, keep_init_run, solver):
    t_ref, t_test, t_test_codegen = (
        esfahani_portfolio_codegen(num_samples=num_samples,
                                   num_sim=num_sim,
                                   gen_code=gen_code,
                                   keep_init_run=keep_init_run,
                                   solver=solver))
    print(t_ref)
    print(t_test)
    print(t_test_codegen)
    with open('paper_dr_portfolio_opt_results_N_{}_solver_{}.csv'.format(str(num_samples), solver), 'a', newline='') as file:
        writer = csv.writer(file)
        for t in range(len(t_test_codegen)):
            writer.writerow([t_ref[t], t_test[t], t_test_codegen[t], solver])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DR Portfolio Optimization.')
    parser.add_argument('--num_samp', type=int, default=10, help='Number of sampled')
    parser.add_argument('--num_sim', type=int, default=5, help='Number of simulations')
    parser.add_argument('--gen_code', type=int, default=1, help='Generate C code flag (0 or 1) (else, use existing)')
    parser.add_argument('--keep_init_run', type=int, default=0,
                        help='Boolean to keep the initial run (0 or 1) (not when calling generated code though)')
    parser.add_argument('--solver', type=str, default=cp.CLARABEL,
                        help="CVXPY solver ['CLARABEL', 'CVXOPT', 'ECOS', 'OSQP', 'SCS', ...]")

    args = parser.parse_args()
    gen_code = bool(args.gen_code)
    keep_init_run = bool(args.keep_init_run)

    paper_dr_portfolio_opt(args.num_samp, args.num_sim, gen_code, keep_init_run, args.solver)
