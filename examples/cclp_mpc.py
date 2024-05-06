import pickle

import cvxpy as cp
from cvxpygen import cpg
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import csv


def generate_gaussian_samples(mu, std_div, num_samples):
    return np.random.normal(mu, std_div, num_samples)


def simple_1d_mpc(use_cpg=False, gen_cpg=False, with_cclp=False, plot_res=False, seed=None):
    """
    Certainty equivalence MPC. 1D linear dynamics with zero-mean additive noise.
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    T = 5  # horizon
    A, B = 1, 1  # dynamics
    x0_mean = 10  # initial state

    # dynamics
    def nom_dyn(x, u):
        return A * x + B * u

    # params and vars
    x0 = cp.Parameter(name='x0')
    ctrl = cp.Variable(T, 'ctrl')
    state = cp.Variable(T + 1, 'state')

    # sim settings
    steps = 100
    current_state = x0_mean
    # plotting results
    x_hist = [current_state]
    u_hist = []
    # run

    # LQR Objective
    Q = 1
    R = 1
    obj = cp.sum_squares(np.sqrt(Q) * state) + cp.sum_squares(np.sqrt(R) * ctrl)

    # Constraints
    eps = 0.1
    constr = [state[0] == x0,  # initial state
              state[1:T + 1] == A * state[:T] + B * ctrl[:],  # dynamics
              cp.abs(ctrl) <= 0.2,  # control bound
              ]
    if with_cclp:
        # Prob(x > 2) >= 1-eps
        w_var = 0.01  # noise variance
        from cvxRiskOpt.cclp_risk_opt import cclp_gauss
        for t in range(1, T+1):
            constr += [cclp_gauss(eps=eps,
                                  a=1,
                                  b=2,
                                  xi1_hat=-state[t],
                                  gam11=(w_var * t)
                                  )]
        # above for loop replaces this:
        # constr += [state[1:] >= 2 + norm.ppf(1 - eps) * np.array([np.sqrt(w_var * t) for t in range(1, T + 1)])]
        # show how the reformulation matches the known reformulation
        for t in range(1, T+1):
            print(cclp_gauss(eps=eps, a=1, b=2, xi1_hat=-state[t], gam11=(w_var * t)).expr)
            print("x({}) >= {}".format(t, 2 + norm.ppf(1 - eps) * np.sqrt(w_var * t)))
    else:
        # x > 2
        w_var = 0.  # noise variance
        constr += [state[1:] >= 2]

    # formulate the problem
    prob = cp.Problem(cp.Minimize(obj), constr)

    # generate C code
    if use_cpg:
        if gen_cpg:
            cpg.generate_code(prob, code_dir='simple_mpc_1d', solver=cp.OSQP)
        from simple_mpc_1d.cpg_solver import cpg_solve
        prob.register_solve('cpg', cpg_solve)

    # Run MPC
    for t in range(steps):
        x0.value = current_state
        if use_cpg:
            prob.solve(method='cpg', updated_params=['x0'])
        else:
            prob.solve(cp.OSQP)

        if prob.status in ["optimal", "solved"]:
            prev_mpc_ctrl = ctrl.value[:]
            u_now = ctrl.value[0]
        else:
            print(prob.status, " at t = ", t)
            try:
                if len(prev_mpc_ctrl) == 1:
                    raise NotImplementedError("Ran out of controls")
                print("USING OLD RESULTS")
                prev_mpc_ctrl = prev_mpc_ctrl[1:]
                u_now = prev_mpc_ctrl[0]
            except:
                break

        w_now = np.random.normal(loc=0.0, scale=np.sqrt(w_var))
        next_state = nom_dyn(current_state, u_now) + w_now  # np.clip(w_now, -0.2, 0.2)
        x_hist.append(next_state)
        current_state = next_state
        u_hist.append(u_now)

    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)
    if plot_res:
        fig, ax = plt.subplots(2)
        ax[0].plot(range(len(x_hist)), x_hist)
        ax[0].plot(range(len(x_hist)), [0] * len(x_hist), 'k')
        ax[0].set(ylabel='x')
        ax[1].plot(range(len(x_hist)), [0.2] * len(x_hist), 'r')
        ax[1].plot(range(len(x_hist)), [-0.2] * len(x_hist), 'r')
        ax[1].plot(range(len(u_hist)), u_hist)
        ax[1].set(ylabel='u')
        plt.xlabel('t')
        plt.show()

    return x_hist, u_hist


def simple_2d_mpc(use_cpg=False, gen_cpg=False, plot_res=False, keep_init_run=False, solver=cp.CLARABEL):
    """
    Certainty equivalence MPC. Linear dynamics with additive noise.
    :return:
    """
    from cvxRiskOpt.cclp_risk_opt import cclp_gauss
    from cvxRiskOpt.mpc_helpers import lin_mpc_expect_xQx

    T = 7  # horizon
    A, B = np.eye(2), np.eye(2)  # dynamics
    # noise info
    w_mean = np.array([0, 0])
    w_cov = np.diag([0.1, 0.01])
    # initial state
    x0_mean = np.array([-2, -0.8])
    # LQ objective cost matrices
    Q = np.diag([1, 1])
    R = np.diag([1, 1])
    # dynamics
    dyn = lambda x, u: A @ x + B @ u
    # params and vars
    x0 = cp.Parameter(2, 'x0')
    ctrl = cp.Variable((2, T), 'ctrl')
    state = cp.Variable((2, T + 1), 'state')
    # sim settings
    steps = 20
    current_state = x0_mean
    # plotting results
    x_hist = [current_state]
    u_hist = []
    t_hist = []
    # run

    obj = 0
    for t in range(T):
        v, _ = lin_mpc_expect_xQx(t + 1, T, A, B, ctrl, Q, x0, w_cov=w_cov)
        obj += v
        obj += cp.quad_form(ctrl[:, t], R)

    constr = [state[:, 0] == x0]
    for t in range(T):
        constr += [state[:, t + 1] == dyn(state[:, t], ctrl[:, t])]
    constr += [ctrl <= np.expand_dims(np.array([0.2, 0.2]), axis=1),
               ctrl >= np.expand_dims(np.array([-0.2, -0.2]), axis=1)]
    sig = w_cov
    for t in range(T):
        for tt in range(t):
            sig = A @ sig @ A.T + w_cov
        constr += [cclp_gauss(eps=0.05,
                              a=np.array([0, 1]),
                              b=-1,
                              xi1_hat=state[:, t + 1],
                              gam11=sig
                              )]
        constr += [cclp_gauss(eps=0.05,
                              a=np.array([0, -1]),
                              b=-1,
                              xi1_hat=state[:, t + 1],
                              gam11=sig
                              )]
    prob = cp.Problem(cp.Minimize(obj), constr)

    if use_cpg:
        if gen_cpg:
            cpg.generate_code(prob, code_dir='simple_mpc_2d', solver=solver)
        from simple_mpc_2d.cpg_solver import cpg_solve
        prob.register_solve('cpg', cpg_solve)

    for t in range(steps):
        x0.value = current_state
        if use_cpg:
            ts = time.time()
            prob.solve(method='cpg', updated_params=['x0'], verbose=False)
            te = time.time()
        else:
            ts = time.time()
            prob.solve(solver)
            te = time.time()
        if t > 0 or (t == 0 and keep_init_run):
            t_hist.append(te-ts)
        print(prob.status)
        u_now = ctrl.value[:, 0]
        w_now = np.hstack([generate_gaussian_samples(w_mean[0], w_cov[0, 0], 1),
                           generate_gaussian_samples(w_mean[1], w_cov[1, 1], 1)])
        next_state = dyn(current_state, u_now) + w_now
        x_hist.append(next_state)
        current_state = next_state
        print(current_state)
        u_hist.append(ctrl.value[:, 0])

    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)
    if plot_res:
        plt.plot(x_hist[:, 0], x_hist[:, 1])
        plt.scatter(0, 0)
        plt.show()
        fig, axs = plt.subplots(2)
        axs[0].plot(range(steps), u_hist[:, 0])
        axs[1].plot(range(steps), u_hist[:, 1])
        plt.show()
    return t_hist


def hvac_mpc_time_varying_constraints(plot_res=False):  # (use_cpg=False, gen_cpg=False):
    """
    Certainty equivalence MPC. Linear dynamics with additive noise.

    :return:
    """
    from cvxRiskOpt.cclp_risk_opt import cclp_gauss
    from cvxRiskOpt.mpc_helpers import lin_mpc_expect_xQx
    from cvxRiskOpt.mpc_helpers import cp_var_mat_to_list

    def generate_signal(hourly_values, N):
        """
        Generate a reference signal with N values per hour given values at every one-hour mark.

        Parameters:
        hourly_values (list): A list of values at every one-hour mark.
        N (int): The number of values per hour to generate.

        Returns:
        list: A list of reference signal values.
        """
        # Create an array of hourly time points
        hourly_time_points = np.arange(len(hourly_values))

        # Create an array of desired time points
        desired_time_points = np.linspace(0, len(hourly_values) - 1, len(hourly_values) * N)

        # Use linear interpolation to generate the reference signal
        reference_signal = np.interp(desired_time_points, hourly_time_points, hourly_values)

        return reference_signal.tolist()

    # x_t+1 = ax_t + bu_t + fw_t
    # x: indoor temp, u: control, w: outdoor temp
    a = 0.9  # drift in indoor temp
    b = 0.1  # influence of control
    f = 0.1  # influence of outdoor temp

    def nom_dyn(x_mean, ctrl, w_mean):
        return a * x_mean + b * ctrl + f * w_mean

    # number of control steps per hour
    inputs_per_hr = 4
    total_days = 1
    total_days = int(np.ceil(total_days))
    total_inputs = total_days * 24 * inputs_per_hr
    horizon = 4

    # desired reference
    # hours   = [0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    ref_temps = [65, 65, 65, 65, 65, 65, 65, 67, 70, 71, 71, 71, 71, 71, 71, 71, 71, 71, 70, 70, 70, 69, 68, 65]
    ref_temps = generate_signal(ref_temps, inputs_per_hr)
    ref_temps *= total_days
    print('num ref temps: ', len(ref_temps))
    print('total inputs: ', total_inputs)

    # outdoor temp (nominal)
    # hours       = [0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    outdoor_temps = [62, 60, 59, 58, 57, 57, 57, 57, 58, 59, 62, 65, 70, 75, 80, 83, 85, 87, 85, 83, 80, 75, 70, 65]
    outdoor_temps = generate_signal(outdoor_temps, inputs_per_hr)
    outdoor_temps *= total_days

    # min indoor temp
    # hours          = [0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    min_indoor_temps = [60, 60, 60, 60, 60, 60, 63, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 63, 60, 60, 60, 60]
    min_indoor_temps = generate_signal(min_indoor_temps, inputs_per_hr)
    min_indoor_temps *= total_days

    # max indoor temp
    # hours          = [0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    max_indoor_temps = [70, 70, 70, 71, 72, 73, 74, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 74, 73, 72, 71, 70]
    max_indoor_temps = generate_signal(max_indoor_temps, inputs_per_hr)
    max_indoor_temps *= total_days

    if plot_res:
        plt.plot(np.linspace(0, total_days * 24, total_inputs), max_indoor_temps, 'r')
        plt.plot(np.linspace(0, total_days * 24, total_inputs), min_indoor_temps, 'r')
        plt.plot(np.linspace(0, total_days * 24, total_inputs), ref_temps, 'g')
        plt.plot(np.linspace(0, total_days * 24, total_inputs), outdoor_temps, 'b')

    x_cur = outdoor_temps[0]
    w_cov = 1

    # Problem setup
    Q = 100
    R = 1
    u = cp.Variable(horizon, 'u')
    x = cp.Variable(horizon + 1, 'x')
    w = cp.Parameter(horizon, 'w')
    r = cp.Parameter(horizon + 1, 'r')
    xmin = cp.Parameter(horizon + 1, 'xmin')
    xmax = cp.Parameter(horizon + 1, 'xmax')
    x0 = cp.Parameter(name='x0')
    x_hist = [x_cur]
    u_hist = []
    w_hist = []
    # MPC
    obj = 0
    constr = [x[0] == x0]
    eps = 0.1
    for t in range(horizon):
        xQx, data = lin_mpc_expect_xQx(t + 1, horizon, a, b, u, Q, x0,
                                       w_mean=cp_var_mat_to_list(f*w, horizon), w_cov=[f**2 * w_cov])
        obj += xQx
        obj += cp.square(r[t + 1]) * Q
        obj -= 2 * data["x_mean"] * Q * r[t + 1]  # not DPP!
        obj += cp.square(u[t]) * R
        # Note: tracking objective results in product of parameters which is not DPP.
        # manual reformulation is possible into DPP.
        # because *this* formulation is not DPP --> cannot use codegen.
        # might introduce a solution to this problem later

        constr += [x[t + 1] == nom_dyn(x[t], u[t], w[t])]
        constr += [cp.abs(u) <= 15]
        constr += [cclp_gauss(eps=eps,
                              a=-1,
                              b=xmin[t+1],
                              xi1_hat=cp.reshape(data["x_mean"], (1, )),
                              gam11=data["x_cov"],
                              assume_sym=True, assume_psd=True
                              )]
        constr += [cclp_gauss(eps=eps,
                              a=1,
                              b=-xmax[t + 1],
                              xi1_hat=cp.reshape(data["x_mean"], (1,)),
                              gam11=data["x_cov"],
                              assume_sym=True, assume_psd=True
                              )]
    prob = cp.Problem(cp.Minimize(obj), constr)

    for t in range(total_inputs - horizon - 1):
        x0.value = x_cur
        r.value = np.array(ref_temps[t:t + horizon + 1])
        xmin.value = np.array(min_indoor_temps[t:t + horizon + 1])
        xmax.value = np.array(max_indoor_temps[t:t + horizon + 1])
        w.value = np.array(outdoor_temps[t:t+horizon])
        prob.solve(cp.CLARABEL)
        print(prob.status)
        print('x: ', x.value)
        print('r: ', r.value)
        u_now = u.value[0]
        w_now = generate_gaussian_samples(outdoor_temps[t], w_cov, 1)[0]
        x_next = nom_dyn(x_cur, u_now, w_now)

        u_hist.append(u_now)
        x_hist.append(x_next)
        w_hist.append(w_now)
        x_cur = x_next
        print(x_cur)
        print("Objective Value: ", prob.value)

    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)
    if plot_res:
        plt.plot(np.linspace(0, (total_inputs - horizon - 1) / inputs_per_hr, total_inputs - horizon), x_hist, 'tab:orange')
        plt.plot(np.linspace(0, (total_inputs - horizon - 1) / inputs_per_hr, total_inputs - horizon - 1), w_hist,
                 'tab:purple')
        plt.legend(['max', 'min', 'ref', 'outdoor', 'actual indoor', 'actual outdoor'])
        plt.show()
        plt.plot(np.linspace(0, (total_inputs - horizon - 1) / inputs_per_hr, total_inputs - horizon - 1), u_hist,
                 'tab:orange')
        plt.show()


def temp_mpc_regulator_time_varying_constraints(horizon=10, inputs_per_hr=4, total_days=2,
                                                plot_res=False,
                                                use_cpg=False, gen_cpg=True, keep_init_run=False,
                                                solver=cp.CLARABEL):
    """
    Certainty equivalence MPC. Linear dynamics with additive noise.

    :return:
    """
    from cvxRiskOpt.cclp_risk_opt import cclp_gauss
    from cvxRiskOpt.mpc_helpers import lin_mpc_expect_xQx
    from cvxRiskOpt.mpc_helpers import cp_var_mat_to_list

    def generate_signal(hourly_values, N):
        """
        Generate a reference signal with N values per hour given values at every one-hour mark.

        Parameters:
        hourly_values (list): A list of values at every one-hour mark.
        N (int): The number of values per hour to generate.

        Returns:
        list: A list of reference signal values.
        """
        # Create an array of hourly time points
        hourly_time_points = np.arange(len(hourly_values))

        # Create an array of desired time points
        desired_time_points = np.linspace(0, len(hourly_values) - 1, len(hourly_values) * N)

        # Use linear interpolation to generate the reference signal
        reference_signal = np.interp(desired_time_points, hourly_time_points, hourly_values)

        return reference_signal.tolist()

    # x_t+1 = ax_t + bu_t + fw_t
    # x: indoor temp, u: control, w: outdoor temp
    a = 0.9  # drift in indoor temp
    b = 0.1  # influence of control
    f = 0.1  # influence of outdoor temp

    def nom_dyn(x_mean, ctrl, w_mean):
        return a * x_mean + b * ctrl + f * w_mean

    # number of control steps per hour
    total_days = int(np.ceil(total_days))
    inputs_per_hr = int(np.ceil(inputs_per_hr))
    total_inputs = total_days * 24 * inputs_per_hr

    # outdoor temp (nominal)
    outdoor_temps = [-15, -15, -12, -10, -7, -6, -5, -4, -3, -2, 0, 1, 3, 5, 7, 10, 13, 15, 16, 17, 18, 19, 20, 20]
    outdoor_temps = generate_signal(outdoor_temps, inputs_per_hr)
    outdoor_temps *= total_days

    # min indoor temp
    min_indoor_temps = [-8, -8, -8, -7, -7, -6, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -4, -3, -3, -2, -2, -2]
    min_indoor_temps = generate_signal(min_indoor_temps, inputs_per_hr)
    min_indoor_temps *= total_days

    # max indoor temp
    max_indoor_temps = [3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7]
    max_indoor_temps = generate_signal(max_indoor_temps, inputs_per_hr)
    max_indoor_temps *= total_days

    if plot_res:
        plt.plot(np.linspace(0, total_days * 24, total_inputs), max_indoor_temps, 'r')
        plt.plot(np.linspace(0, total_days * 24, total_inputs), min_indoor_temps, 'r')
        plt.plot(np.linspace(0, total_days * 24, total_inputs), [0] * len(np.linspace(0, total_days * 24, total_inputs)), 'g')
        plt.plot(np.linspace(0, total_days * 24, total_inputs), outdoor_temps, 'b')

    x_cur = -11
    w_cov = 1

    # Problem setup
    Q = 100
    R = 1
    u = cp.Variable(horizon, 'u')
    x = cp.Variable(horizon + 1, 'x')
    w = cp.Parameter(horizon, 'w')
    xmin = cp.Parameter(horizon + 1, 'xmin')
    xmax = cp.Parameter(horizon + 1, 'xmax')
    x0 = cp.Parameter(name='x0')
    t_hist = []
    x_hist = [x_cur]
    u_hist = []
    w_hist = []
    # MPC
    obj = 0
    constr = [x[0] == x0]
    eps = 0.1
    for t in range(horizon):
        xQx, data = lin_mpc_expect_xQx(t + 1, horizon, a, b, u, Q, x0,
                                       w_mean=cp_var_mat_to_list(f*w, horizon), w_cov=[f**2 * w_cov])
        obj += xQx
        obj += cp.square(u[t]) * R

        constr += [x[t + 1] == nom_dyn(x[t], u[t], w[t])]
        # constr += [cp.abs(u) <= 15]
        constr += [cclp_gauss(eps=eps,
                              a=-1,
                              b=xmin[t+1],
                              xi1_hat=cp.reshape(data["x_mean"], (1, )),
                              gam11=data["x_cov"],
                              assume_sym=True, assume_psd=True
                              )]
        constr += [cclp_gauss(eps=eps,
                              a=1,
                              b=-xmax[t + 1],
                              xi1_hat=cp.reshape(data["x_mean"], (1,)),
                              gam11=data["x_cov"],
                              assume_sym=True, assume_psd=True
                              )]
    prob = cp.Problem(cp.Minimize(obj), constr)

    if use_cpg:
        if gen_cpg:
            cpg.generate_code(prob, code_dir='temp_reg_mpc', solver=solver)
        from temp_reg_mpc.cpg_solver import cpg_solve
        prob.register_solve('cpg', cpg_solve)

    for t in range(total_inputs - horizon - 1):
        x0.value = x_cur
        xmin.value = np.array(min_indoor_temps[t:t + horizon + 1])
        xmax.value = np.array(max_indoor_temps[t:t + horizon + 1])
        w.value = np.array(outdoor_temps[t:t+horizon])
        if use_cpg:
            if solver in [cp.CLARABEL, cp.SCS]:
                ts = time.time()
                prob.solve(method='cpg', updated_params=['x0', 'xmin', 'xmax', 'w'], verbose=False)
                te = time.time()
            else:
                ts = time.time()
                prob.solve(method='cpg', updated_params=['x0', 'xmin', 'xmax', 'w'])
                te = time.time()
        else:
            ts = time.time()
            prob.solve(solver=solver)
            te = time.time()
        print(prob.status)
        print('x: ', x.value)
        u_now = u.value[0]
        w_now = generate_gaussian_samples(outdoor_temps[t], w_cov, 1)[0]
        x_next = nom_dyn(x_cur, u_now, w_now)

        u_hist.append(u_now)
        x_hist.append(x_next)
        w_hist.append(w_now)
        if t > 0 or (t == 0 and keep_init_run):
            t_hist.append(te-ts)
        x_cur = x_next
        print(x_cur)
        print("Objective Value: ", prob.value)

    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)
    if plot_res:
        plt.plot(np.linspace(0, (total_inputs - horizon - 1) / inputs_per_hr, total_inputs - horizon), x_hist, 'tab:orange')
        plt.plot(np.linspace(0, (total_inputs - horizon - 1) / inputs_per_hr, total_inputs - horizon - 1), w_hist,
                 'tab:purple')
        plt.legend(['max', 'min', 'ref', 'outdoor', 'actual indoor', 'actual outdoor'])
        plt.show()
        plt.plot(np.linspace(0, (total_inputs - horizon - 1) / inputs_per_hr, total_inputs - horizon - 1), u_hist,
                 'tab:orange')
        plt.show()
    return t_hist


if __name__ == "__main__":
    simple_1d_mpc(use_cpg=False, gen_cpg=False, with_cclp=True, plot_res=True)
    simple_2d_mpc(use_cpg=False, gen_cpg=False, plot_res=True)
    hvac_mpc_time_varying_constraints(plot_res=True)
    temp_mpc_regulator_time_varying_constraints(plot_res=True, use_cpg=False, gen_cpg=False)
