#Bench test version by Hyunwoo Moon
#echo $dlrm_extra_option

cpu=1
gpu=1
pt=1

#------------------
ncores=28 #12 #6
nsockets="0"

ngpus="1 2 4 8"

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_s_pytorch.py"

#------Change the data type into sythetic------------
data=synthetic
print_freq=100
rand_seed=727
#------------------
ncores=4

#---------Model Parameter---------
ncores=28 #12 #6
#Model param
mb_size=2048 #1024 #512 #256
nbatches=1000 #500 #100
bot_mlp="512-512-64"
top_mlp="1024-1024-1024-1"
emb_size=64
nindices=100
emb="1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
interaction="dot"
tnworkers=0
tmb_size=16384

#_args="--mini-batch-size="${mb_size}\
_args=" --num-batches="${nbatches}\
" --data-generation="${data}\
" --arch-mlp-bot="${bot_mlp}\
" --arch-mlp-top="${top_mlp}\
" --arch-sparse-feature-size="${emb_size}\
" --arch-embedding-size="${emb}\
" --num-indices-per-lookup="${nindices}\
" --arch-interaction-op="${interaction}\
" --numpy-rand-seed="${rand_seed}\
" --print-freq="${print_freq}\
" --print-time"\
" --enable-profiling "

# GPU Benchmarking
if [ $gpu = 1 ]; then
  echo "--------------------------------------------"
  echo "GPU Benchmarking"
  echo "--------------------------------------------"
    # weak scaling
    # _mb_size=$((mb_size*_ng))
    # strong scaling
    _mb_size=$((mb_size*1))
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    if [ $pt = 1 ]; then
      outf="model1_GPU_PT_$_ng.log" # ng looks like GPU number.
      outpk="dlrm_s_pytorch.prof"
      echo "-------------------------------"
      echo "Running PT (log file: $outf)"
      echo "-------------------------------"
      cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args --use-gpu $dlrm_extra_option > $outf"
      echo $cmd
      eval $cmd
      min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
      echo "Min time per iteration = $min"
      # move profiling file(s)
      mv $outp ${outf//".log"/".prof"}
      mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
    fi
fi