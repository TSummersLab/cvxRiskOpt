
script="paper_dr_portfolio_opt.py"

N_list=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300)
num_sim=100

# CLARABEL
for N in "${N_list[@]}"; do
    echo "Solving with CLARABEL for N = $N samples"
  python3 $script --num_samp $N --num_sim $num_sim --gen_code 1 --keep_init_run 0 --solver "CLARABEL"
  sleep .5
done

# ECOS
# (for some reason, ECOS produces a wrong result, for this problem, when resolving with C code.
# This resolves this issue.)
for N in "${N_list[@]}"; do
  echo "Solving with ECOS for N = $N samples"
  echo "i = 1 / $num_sim"
  python3 $script --num_samp $N --num_sim 1 --gen_code 1 --keep_init_run 0 --solver "ECOS"
  for ((i=2;i<=num_sim;i++)); do
      echo "i = $i / $num_sim"
      python3 $script --num_samp $N --num_sim 1 --gen_code 0 --keep_init_run 0 --solver "ECOS"
      echo "~~~~~"
  done
  sleep .5
done

