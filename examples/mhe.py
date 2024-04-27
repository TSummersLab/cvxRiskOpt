import time

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from cvxRiskOpt.cclp_risk_opt import cclp_gauss, cclp_dro_mean_cov
from cvxpygen import cpg


def simple_1d_mhe(horizon=10, sim_steps=200, constraint_type=None, plot_res=False,
                  use_cpg=False, gen_cpg=False,
                  keep_init_run=False, solver=cp.CLARABEL, seed=None):
    if constraint_type not in ["gauss", "moment", "sym_moment"]:
        if constraint_type in ["None", "none"]:
            constraint_type = None
        if constraint_type is not None:
            print("Constraint {} type not supported. Replacing with None".format(constraint_type))

    if seed is not None:
        np.random.seed(seed)

    def dyn(x, u, w):
        return x + u + w

    def meas(x, v):
        return x + v

    N_mhe = horizon
    w_mean, v_mean = 0, 0
    w_var, v_var = 0.01, 10
    u_min, u_max = -0.5, 0.5
    x_min, x_max = 0, 60
    x0 = 5  # initial state

    x_mhe = cp.Variable(N_mhe + 1, 'x_mhe')
    x_mhe0 = cp.Parameter(name='x_mhe0')
    y_mhe = cp.Parameter(N_mhe + 1, 'y_mhe')
    u_mhe = cp.Parameter(N_mhe, 'u_mhe')
    mhe_obj = 0
    constr = [x_mhe[0] == x_mhe0]
    for t in range(N_mhe + 1):
        mhe_obj += cp.square(y_mhe[t] - meas(x_mhe[t], v_mean))
    for t in range(N_mhe):
        mhe_obj += 100 * cp.square(x_mhe[t + 1] - dyn(x_mhe[t], u_mhe[t], w_mean))
        if constraint_type == "gauss":
            constr += [cclp_gauss(eps=0.05, a=-1, b=x_min,
                                  xi1_hat=dyn(x_mhe[t], u_mhe[t], w_mean),
                                  gam11=w_var)]
            constr += [cclp_gauss(eps=0.05, a=1, b=-x_max,
                                  xi1_hat=dyn(x_mhe[t], u_mhe[t], w_mean),
                                  gam11=w_var)]
        elif constraint_type == "moment":
            constr += [cclp_dro_mean_cov(eps=0.05, a=-1, b=x_min,
                                         xi1_hat=dyn(x_mhe[t], u_mhe[t], w_mean),
                                         gam11=w_var)]
            constr += [cclp_dro_mean_cov(eps=0.05, a=1, b=-x_max,
                                         xi1_hat=dyn(x_mhe[t], u_mhe[t], w_mean),
                                         gam11=w_var)]
        elif constraint_type == "sym_moment":
            constr += [cclp_dro_mean_cov(eps=0.05, a=-1, b=x_min,
                                         xi1_hat=dyn(x_mhe[t], u_mhe[t], w_mean),
                                         gam11=w_var, centrally_symmetric=True)]
            constr += [cclp_dro_mean_cov(eps=0.05, a=1, b=-x_max,
                                         xi1_hat=dyn(x_mhe[t], u_mhe[t], w_mean),
                                         gam11=w_var, centrally_symmetric=True)]

    mhe_prob = cp.Problem(cp.Minimize(mhe_obj), constraints=constr)

    if use_cpg:
        if gen_cpg:
            cpg.generate_code(mhe_prob, code_dir='mhe_1d', solver=solver)
        from mhe_1d.cpg_solver import cpg_solve
        mhe_prob.register_solve('cpg', cpg_solve)

    # set up the control signal
    u_hist = np.zeros(sim_steps - 1)
    for t in range(sim_steps - 1):
        u_cur = 0.5 if t < sim_steps / 2 else -0.5
        u_cur = np.clip(u_cur, u_min, u_max)
        u_hist[t] = u_cur

    # get all the data
    x_true_hist = np.zeros(sim_steps)
    y_meas_hist = np.zeros(sim_steps)
    for t in range(sim_steps):
        if t == 0:
            x_true_hist[t] = x0
        else:
            w_cur = np.random.normal(loc=w_mean, scale=np.sqrt(w_var))
            x_true_hist[t] = dyn(x_true_hist[t - 1], u_hist[t - 1], w_cur)
        # measure state
        v_cur = np.random.normal(loc=v_mean, scale=np.sqrt(v_var))
        y_meas_hist[t] = meas(x_true_hist[t], v_cur)

    t_hist = []
    x_est_hist = np.zeros(sim_steps)
    x_kf_est_hist = np.zeros(sim_steps)
    x_kf_est_hist[0] = y_meas_hist[0]
    P, K = 0, 0
    for t in range(1, sim_steps):
        if t >= N_mhe:
            y_mhe.value = y_meas_hist[t - N_mhe:t + 1]
            u_mhe.value = u_hist[t - N_mhe:t]
            x_mhe0.value = x_est_hist[t - N_mhe]
            if use_cpg:
                if solver in [cp.CLARABEL, cp.SCS]:
                    ts = time.time()
                    mhe_prob.solve(method='cpg', updated_params=['y_mhe', 'u_mhe', 'x_mhe0'], verbose=False)
                    te = time.time()
                else:
                    ts = time.time()
                    mhe_prob.solve(method='cpg', updated_params=['y_mhe', 'u_mhe', 'x_mhe0'])
                    te = time.time()
            else:
                ts = time.time()
                mhe_prob.solve(solver=solver)
                te = time.time()

            if t > N_mhe or keep_init_run:
                t_hist.append(te - ts)
            print(mhe_prob.status)
            x_est_hist[t - N_mhe + 1:t + 1] = x_mhe.value[1:]

        # # KF gain:
        P_pred = P + w_var
        K = P_pred / (P_pred + v_var)
        P = (1 - K) * P_pred
        x_kf_est_hist[t] = (1 - K) * dyn(x_kf_est_hist[t - 1], u_hist[t - 1], w_mean) + K * y_meas_hist[t - 1]

    if plot_res:
        plt.plot(range(sim_steps), x_true_hist, color='k')
        plt.plot(range(sim_steps), y_meas_hist, color='tab:red', alpha=0.5, linestyle='--')
        plt.plot(range(sim_steps), x_est_hist, color='tab:green')
        plt.plot(range(sim_steps), x_kf_est_hist, color='tab:blue')
        plt.legend(["true", "meas", "est", "KF"])
        plt.show()

    mean_est_diff = np.mean(np.abs(x_true_hist - x_est_hist)[1:])
    mean_meas_diff = np.mean(np.abs(x_true_hist - y_meas_hist)[1:])
    mean_kf_est_diff = np.mean(np.abs(x_true_hist - x_kf_est_hist)[1:])
    print('MHE diff:', mean_est_diff)
    print('Measurements diff: ', mean_meas_diff)
    print('KF diff: ', mean_kf_est_diff)

    return t_hist


if __name__ == "__main__":
    # TODO: Clarabel output using compiled code is wrong
    solver = cp.OSQP
    t_hist = simple_1d_mhe(horizon=10, sim_steps=200, constraint_type="gauss", plot_res=False,
                           use_cpg=False, gen_cpg=False,
                           keep_init_run=False, solver=solver, seed=2)

    t_hist_gen = simple_1d_mhe(horizon=10, sim_steps=200, constraint_type="gauss", plot_res=True,
                               use_cpg=True, gen_cpg=True,
                               keep_init_run=False, solver=solver, seed=2)

    print(t_hist)
    print(np.mean(t_hist) * 1000, "ms")
    print(t_hist_gen)
    print(np.mean(t_hist_gen) * 1000, "ms")
