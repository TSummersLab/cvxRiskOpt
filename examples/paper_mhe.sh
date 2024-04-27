
script="paper_mhe.py"

H=40
sim_steps=200
constraint_type="guass"

#echo "Solving with CLARABEL with H=$H"
#python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "CLARABEL"
#sleep .5

echo "Solving with SCS with H=$H"
python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "SCS"
sleep .5

echo "Solving with OSQP with H=$H"
python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "OSQP"
sleep .5

echo "Solving with ECOS with H=$H"
python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "ECOS"
sleep .5



constraint_type="moment"

#echo "Solving with CLARABEL with H=$H"
#python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "CLARABEL"
#sleep .5

echo "Solving with SCS with H=$H"
python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "SCS"
sleep .5

echo "Solving with OSQP with H=$H"
python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "OSQP"
sleep .5

echo "Solving with ECOS with H=$H"
python3 $script --horizon $H --sim_steps $sim_steps --constraint_type $constraint_type --keep_init_run 0 --solver "ECOS"
sleep .5
