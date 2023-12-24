** Total Recall in Industrial Anomaly Detection with PatchCore**

```markdown


## Overview
PatchCore is a powerful method for image-level anomaly detection and pixel-level anomaly localization. This repository provides the implementation along with various pretrained models that achieve high-performance metrics.

## Quick Guide

**Clone the Repository:**
```bash
git clone https://github.com/your-username/patchcore.git
cd patchcore
```

**Set PYTHONPATH:**
```bash
export PYTHONPATH=src
```

**Train PatchCore on MVTec AD:**
```bash
datapath=/path_to_mvtec_folder/mvtec
datasets=('bottle' 'cable' 'capsule' ...)
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model \
--log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_online --log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
--pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
```

**Evaluate Pretrained Models:**
```bash
loadpath=/path_to_pretrained_patchcores_models
modelfolder=IM224_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S0
savefolder=evaluated_results'/'$modelfolder

model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
```

## In-Depth Description

### Requirements
- Python 3.8
- Packages listed in `requirements.txt`
- GPU with at least 11GB memory

### Setting up MVTec AD
1. Download the [MVTec AD benchmark](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place it in the specified datapath.

### Training PatchCore
- Run `bin/run_patchcore.py` with appropriate parameters to train PatchCore on MVTec AD.

### Evaluating a Pretrained PatchCore Model
- Run `bin/load_and_evaluate_patchcore.py` with the path to pretrained models for evaluation.
```
-@misc{roth2021total,
 - title={Towards Total Recall in Industrial Anomaly Detection},
 - author={Karsten Roth and Latha Pemula and Joaquin Zepeda and Bernhard Schölkopf and Thomas Brox and Peter Gehler},
 - year={2021},
 - eprint={2106.08265},
 - archivePrefix={arXiv},
 - primaryClass={cs.CV}
}

