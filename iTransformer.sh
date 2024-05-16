export CUDA_VISIBLE_DEVICES=0


python3.9 -u iTransformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path data/iTransformer/ \
  --data_path vix_5m_600000.SH.csv \
  --checkpoints ckpt/ \
  --target redline \
  --model_id vix_128_20 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 128 \
  --label_len 64 \
  --pred_len 20 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --des 'bond50' \
  --d_model 512\
  --d_ff 512\
  --itr 1 
 # --inverse
