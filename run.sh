ssh user@8.131.252.85 -p 35789
Pengslab1404~

conda activate deit
conda deactivate


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /home/user/mengxin/image_net --output_dir /home/user/mengxin/deit/checkpoint

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /home/user/mengxin/image_net --output_dir /home/user/mengxin/deit/checkpoint


python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --batch-size 256 --data-path /home/user/mengxin/image_net --output_dir /home/user/mengxin/deit/checkpoint


