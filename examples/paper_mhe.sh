
script="paper_mhe.py"

H_list=(10 20 30 40 50 60 70 80 90)
sim_steps=200
constraint_type="moment"

for H in "${H_list[@]}"; do
  echo "Solving with ECOS with H=$H"
  python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "ECOS"
  sleep .5

  echo "Solving with CLARABEL with H=$H"
  python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "CLARABEL"
  sleep .5

  echo "Solving with OSQP with H=$H"
  python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "OSQP"
  sleep .5
done

