VERSION=CLIP2Video
DATA_PATH=${VERSION}/data/msrvtt_data/
CHECKPOINT=[downloaded trained model path]
MODEL_NUM=2

python ${VERSION}/infer_retrieval.py \
--num_thread_reader=2 \
--val_csv ${DATA_PATH}/MSRVTT_test.full.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path [frame path] \
--output_dir ${CHECKPOINT}/test_${MODEL_NUM}.txt \
--max_words 32 \
--max_frames 12 \
--batch_size_val 64 \
--datatype msrvttfull \
--feature_framerate 2 \
--sim_type seqTransf \
--checkpoint ${CHECKPOINT} \
--do_eval \
--model_num ${MODEL_NUM} \
--temporal_type TDB \
--temporal_proj sigmoid_selfA \
--center_type TAB \
--centerK 5 \
--center_weight 0.5 \
--center_proj TAB_TDB \
--clip_path ${VERSION}/ViT-B-32.pt
