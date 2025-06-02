
model_name="kaishxu/ARJudge"
test_name_lst=("Auto-J")
export CUDA_VISIBLE_DEVICES=2
for test_name in ${test_name_lst[@]}
do

python analyze.py --model ${model_name} --data_path data/test/${test_name}_test.json --tensor_parallel_size 1 --batch_size 1000 --swap

python generation.py --model Qwen/Qwen2.5-7B-Instruct --data_path results/ARjudge_prompt_w_analysis.json --save_path results/ARjudge_${test_name}_test_refine_results.json --tensor_parallel_size 1 --batch_size 1000 --max_length 4096 --swap

mkdir -p results
python metrics.py --result_path results/ARjudge_${test_name}_test_refine_results.json --swap

done