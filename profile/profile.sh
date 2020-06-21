#PYTHONPATH=../ python -m cProfile -o output.pstats $1
#PYTHONPATH=../ python -m torch.utils.bottleneck $1
#./gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png
FSP_DATA_DIR=/home/max/MERF_FSP/data/RawData /usr/local/cuda/bin/nvprof -f --print-gpu-trace /home/max/miniconda3/envs/MERF-FSP/bin/python -m dog.dog
