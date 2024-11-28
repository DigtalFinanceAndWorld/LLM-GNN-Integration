import tensorflow as tf

checkpoint_path = "../data/MulDiGraph/delay_5/1/ckpt_dir/bert_finetune"
# checkpoint_path = "../data/MulDiGraph/delay_5/1/ckpt_dir/model_10000"
reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
variable_map = reader.get_variable_to_shape_map()
for key in variable_map:
    print(key)
