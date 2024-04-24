
script="paper_temp_reg_mpc.py"

H=40
inputs_per_hr=4
total_days=5

echo "Solving with CLARABEL with H=$H"
python3 $script --horizon $H --inputs_per_hr $inputs_per_hr --total_days $total_days --keep_init_run 0 --solver "CLARABEL"
sleep .5

echo "Solving with SCS with H=$H"
python3 $script --horizon $H --inputs_per_hr $inputs_per_hr --total_days $total_days --keep_init_run 0 --solver "SCS"
sleep .5

echo "Solving with OSQP with H=$H"
python3 $script --horizon $H --inputs_per_hr $inputs_per_hr --total_days $total_days --keep_init_run 0 --solver "OSQP"
sleep .5


H=80
inputs_per_hr=10

echo "Solving with CLARABEL with H=$H"
python3 $script --horizon $H --inputs_per_hr $inputs_per_hr --total_days $total_days --keep_init_run 0 --solver "CLARABEL"
sleep .5

echo "Solving with SCS with H=$H"
python3 $script --horizon $H --inputs_per_hr $inputs_per_hr --total_days $total_days --keep_init_run 0 --solver "SCS"
sleep .5

echo "Solving with OSQP with H=$H"
python3 $script --horizon $H --inputs_per_hr $inputs_per_hr --total_days $total_days --keep_init_run 0 --solver "OSQP"
sleep .5
