
num_sim=100
script="paper_dr_portfolio_opt.py"

#N_list=(10 20 50 100 150 200 250 300)
#N_list=(25 50 75 100 125 150 175 200 225 250 275 300 325)

## CLARABEL
#N_list=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300)
#for N in "${N_list[@]}"; do
#    echo "Solving with CLARABEL for N = $N samples"
#  python3 $script --num_samp $N --num_sim $num_sim --gen_code 1 --keep_init_run 0 --solver "CLARABEL"
#  sleep .5
#done

# ECOS
#N_list=(10 20 50 75 100 125 150 175 200 225 250 275 300 350 400 450 500 600 700 800 900 1000)
#N_list=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300)
N_list=(220 230 250)
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

