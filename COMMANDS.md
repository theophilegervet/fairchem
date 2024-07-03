Env:
```bash
wget https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/packages/env.gpu.yml
conda env create -f env.gpu.yml
pip install mendeleev
```

Download data:
```bash
cd /home/code/fairchem/src/fairchem/core
NUM_WORKERS=50
for SPLIT in "all" "val_id"; do
    python scripts/download_data.py --task s2ef --split "$SPLIT" --num-workers $NUM_WORKERS --ref-energy
    python scripts/make_lmdb_sizes.py --data-path /mnt/vast/home/theo/code/fairchem/src/fairchem/data/s2ef/all/$SPLIT --num-workers $NUM_WORKERS
done
```

On 1 GPU to debug:
```bash
CONFIG="configs/s2ef/all/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml"
CONFIG="configs/s2ef/all/gemnet/gemnet-oc.yml"
CONFIG="configs/s2ef/all/schnet/schnet.yml"
CONFIG="configs/s2ef/all/painn/painn_h512.yml"
CONFIG="configs/s2ef/all/baseline/baseline.yml"
CONFIG="configs/s2ef/all/baseline/frame_averaging.yml"
python main.py --mode train --config-yml $CONFIG
```

With submitit:
```bash
CONFIG="configs/s2ef/all/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml"
EXP=e
CONFIG="configs/s2ef/all/gemnet/gemnet-oc.yml"
EXP=g
CONFIG="configs/s2ef/all/schnet/schnet.yml"
EXP=s
CONFIG="configs/s2ef/all/painn/painn_h512.yml"
EXP=p
CONFIG="configs/s2ef/all/baseline/baseline.yml"
EXP=b
CONFIG="configs/s2ef/all/baseline/frame_averaging.yml"
EXP=f

python main.py --distributed --num-gpus 8 --num-nodes 1 \
    --identifier "$(date +%y%m%d)_$EXP" --submit --mode train --config-yml $CONFIG
```
