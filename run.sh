

python main.py --task charades --max_pos_len 64 --char_dim 50 --gpu_idx 2 --suffix gt --mode test
python main.py --task anet --max_pos_len 100 --char_dim 100 --gpu_idx 2 --suffix gt --mode test




CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/SeqPAN.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/BaseFast.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/charades/SingleTeacher.yaml

