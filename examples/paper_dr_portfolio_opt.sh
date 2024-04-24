
N=15
num_sim=20
script="paper_dr_portfolio_opt.py"

echo "Solving with CLARABEL"
python3 $script --num_samp $N --num_sim $num_sim --gen_code 1 --keep_init_run 0 --solver "CLARABEL"
sleep .5

echo "Solving with ECOS"
echo "i = 1 / $num_sim"
python3 $script --num_samp $N --num_sim 1 --gen_code 1 --keep_init_run 0 --solver "ECOS"
for ((i=2;i<=num_sim;i++)); do
    echo "i = $i / $num_sim"
    python3 $script --num_samp $N --num_sim 1 --gen_code 0 --keep_init_run 0 --solver "ECOS"
    echo "~~~~~"
done
sleep .5

echo "Solving with SCS"
python3 $script --num_samp $N --num_sim $num_sim --gen_code 1 --keep_init_run 0 --solver "SCS"
