if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96  
model_name=X_TST

root_path_name=./dataset/
data_path_name=PEMS08.npz
model_id_name=PEMS08
data_name=PEMS

random_seed=2021

for pred_len in 12 24 48 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 170 \
      --e_layers 2 \
      --n_heads 4 \
      --d_model 128 \
      --d_ff 128 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 300\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 \
      --conv_time 5 \
      --stride1 3 \
      --stride2 5 \
      --stride3 7 \
      --batch_size 32 \
      --learning_rate 0.0005 \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
Done

for pred_len in 24
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 170 \
      --e_layers 2 \
      --n_heads 4 \
      --d_model 128 \
      --d_ff 128 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 300\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 \
      --conv_time 5 \
      --stride1 3 \
      --stride2 5 \
      --stride3 7 \
      --batch_size 32 \
      --learning_rate 0.0005 \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
Done

for pred_len in 48
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 170 \
      --e_layers 2 \
      --n_heads 4 \
      --d_model 128 \
      --d_ff 128 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 300\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 \
      --conv_time 5 \
      --stride1 3 \
      --stride2 5 \
      --stride3 7 \
      --batch_size 32 \
      --learning_rate 0.0005 \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
Done

for pred_len in 96 
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 170 \
      --e_layers 4 \
      --n_heads 16 \
      --d_model 512 \
      --d_ff 512 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 300\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 \
      --conv_time 5 \
      --stride1 3 \
      --stride2 5 \
      --stride3 7 \
      --batch_size 32 \
      --learning_rate 0.0001 \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
