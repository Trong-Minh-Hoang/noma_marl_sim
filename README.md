# Cài dependencies
pip install torch numpy pyyaml

# Chạy training (full model)
python train.py --run_id 0 --ablation full --episodes 40000

# Ablation studies
python train.py --run_id 1 --ablation no_interference
python train.py --run_id 2 --ablation no_qos
python train.py --run_id 3 --ablation single_phase
