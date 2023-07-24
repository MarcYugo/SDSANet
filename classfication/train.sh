CUDA_VISIBLE_DEVICES=0,1,2,3 
bash distributed_train.sh 4 /path/to/data \
	  --model dsan_t\
      -b 172 \
      --lr 1e-3 \
      --drop-path 0.1 \
      --amp
