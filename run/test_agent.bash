name=agent_test_unseen
flag="--attn soft --train validlistener
      --load snap/agent/state_dict/best_val_unseen 
      --features rn50x4
      --feature_size 640
      --submit
      --batchSize 64
      --featdropout 0.0
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --option_size 8
      --option_step 3
      --subout max --dropout 0.0 --optim adam --lr 1e-4 --iters 200000 --maxAction 35"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 CUDA_LAUNCH_BLOCKING=1 python r2r_src/train.py $flag --name $name