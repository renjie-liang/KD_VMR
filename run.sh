python main.py --task charades --max_pos_len 64 --char_dim 50 --gpu_idx 2 --suffix gt --mode test
python main.py --task anet --max_pos_len 100 --char_dim 100 --gpu_idx 2 --suffix gt --mode test

CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/charades/SeqPAN.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/charades/BaseFast.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/charades/SingleTeacher.yaml
CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/charades/MultiTeacher.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/MultiTeacherPlus.yaml



CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/anet/SeqPAN.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/BaseFast.yaml
CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/anet/SingleTeacher.yaml
