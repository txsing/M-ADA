source /home/txsing/anaconda3/bin/activate mada

# python main_Digits.py --dataset photo  --cov_weight 0.1 --activate_threshold 0.05 --cover_ratio 1 --GPU_ID 0

ratios=(1.0 0.9 0.6 0.3 0.1 0.01)
thresholds=(0.001 0.01 0.02)
weights=(0.1 1)

for w in ${weights[@]}; do
	for t in ${thresholds[@]}; do
		for r in ${ratios[@]}; do
            echo $w-$t-$r
            python main_Digits.py --cov_weight ${w} --activate_threshold ${t} --cover_ratio ${r} --GPU_ID 5
			echo $res
        done
	done
done
