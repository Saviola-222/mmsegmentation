### deeplabv3+
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_1 configs_dsn/deeplabv3plus/deeplabv3plus-r101_d8-832x832-contract_true-align_true-80k-bs_1-cityscapes-stem_64.py --seed=0

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_2 configs_dsn/deeplabv3plus/deeplabv3plus-r101_d8-832x832-contract_false-align_true-80k-bs_1-cityscapes-stem_64.py --seed=0

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_3 configs_dsn/deeplabv3plus/deeplabv3plus-r101_d8-832x832-contract_true-align_false-80k-bs_1-cityscapes-stem_64.py --seed=0

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_4 configs_dsn/deeplabv3plus/deeplabv3plus-r101_d8-832x832-contract_false-align_false-80k-bs_1-cityscapes-stem_64.py --seed=0

### sep_dsn
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_5 configs_dsn/sep_dsn/deeplabv3plus-r101_d8-832x832-contract_true-align_false-stem_channels_64-80k-load_ckpt_mmseg-cityscapes.py --seed=0

### dsn
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_6 configs_dsn/dsn/deeplabv3plus-r101_d8-832x832-contract_true-align_false-80k-bs_1-cityscapes-stem_64.py --seed=0
