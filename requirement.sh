conda install -c anaconda scikit-learn -y
conda install -c conda-forge tensorboard -y
conda install -c conda-forge tqdm -y
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
