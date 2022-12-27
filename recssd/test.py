import os, sys

# Test version of sweep

num_epochs  = 1

num_batches = 400

rt_config_gpu = "--inference_only --inter_op_workers 1 --caffe2_net_type async_dag --use_gpu "

model_config = "--model_type dlrm --arch_mlp_top \"256-64-1\" --arch_mlp_bot \"128-64-32\" --arch_sparse_feature_size 32 --arch_embedding_size \"1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000\" --num_indices_per_lookup 80 --num_indices_per_lookup_fixed True --arch_interaction_op cat "

n = 16

data_config = "--nepochs " + str(num_epochs) + " --num_batches " + str(num_batches) + \
        " --mini_batch_size " + str(n) + " --max_mini_batch_size " + str(n) + " --sls_type " + \
                " --data_generation synthetic"

gpu_command = "python dlrm_s_caffe2.py " + rt_config_gpu + model_config + data_config

print("--------------------Running (RM1) GPU Test with Batch Size " + str(n) +"--------------------\n")
print(gpu_command)
os.system(gpu_command)