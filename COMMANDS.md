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

2M:
```bash
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml
python main.py --mode train --config-yml configs/s2ef/2M/baseline/baseline.yml
```

All:
```bash
python main.py --mode train --config-yml configs/s2ef/all/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml --num_gpus 8
python main.py --mode train --config-yml configs/s2ef/all/baseline/baseline.yml --num_gpus 8
```
