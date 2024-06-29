Env:
```bash
wget https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/packages/env.gpu.yml
conda env create -f env.gpu.yml
pip install mendeleev
```

Download data:
```bash
cd /home/code/fairchem/src/fairchem/core
for SPLIT in "2M" "all" "val_id"; do
    python scripts/download_data.py --task s2ef --split "$SPLIT" --num-workers 8 --ref-energy
done
```

On 1 GPU to debug:
```bash
CONFIG="configs/s2ef/all/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml"
CONFIG="configs/s2ef/all/baseline/baseline.yml"
python main.py --mode train --config-yml $CONFIG
```

With submitit:
```bash
CONFIG="configs/s2ef/all/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml"
EXP=equi

CONFIG="configs/s2ef/all/baseline/baseline.yml"
EXP=base

python main.py --distributed --num-gpus 8 --num-nodes 1 \
    --identifier "$(date +%y%m%d)_$EXP" --submit --mode train --config-yml $CONFIG
```
