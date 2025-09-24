# EgoPER Downstream Analysis
This is ***modified*** [EgoPER](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Error_Detection_in_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf) for ["What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning"](https://arxiv.org/abs/2503.21055) for downstream analysis. Full credits for EgoPER goes to the original authors. For details about the methodoloty, please refer the original paper/ code.

## Preparation
For preparation, please follow the original [EgoPER code](https://github.com/robert80203/EgoPER_official).

## Training
Train model using features extracted from different feature extractors. Depending on the feature extractor, the feature dimension can change, so we added user argument to take dynamic feature size in `train.py`.

### Notes
Note that `train.py` in this repo is revised version from the origianl. The differences include that new train.py can...
1. take dynamic `--input_dim` user argument to compare different models
2. take `--cp_dir` for different checkpoint directories
3. take `--seed` for random seed

### How-to-run
#### train.py
`Stage 1 Trainig`
```
python train.py {config_yaml_file} --output {output_name} --feat_dirname {saved_feature_path} --input_dim {input_feature_dimension} --cp_dir {save_checkpoint_dir} --seed {random_seed}
```
Example:
```
python train_others.py ./configs/EgoPER/tea_aod_bgr1.0.yaml --output final_output --feat_dirname features_method1 --input_dim 768 --cp_dir ./ckpt/method1 --data_root_dir ./data
```

`Stage 2 Training`
```
python train.py {config_yaml_file} --resume {saved_model_weight_path} --output {output_name} --feat_dirname {saved_feature_path} --input_dim {input_feature_dimension} --cp_dir {save_checkpoint_dir}
```
Example:
```
python train.py ./configs/EgoPER/tea_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/method1/tea_aod_bgr1.0_final/epoch_105.pth.tar --output final_output --feat_dirname features_method1 --input_dim 768 --cp_dir ./ckpt/method1 --data_root_dir ./data
```

## Testing
Same as training, feature dimension can vary depending on the feature extraction method, so we added `--input_dim` user argument in `test_ed.py` and `test.py`.

### Notes
Note that this is also different from the original code (takes `--input_dim` to compare multiple models).

### How-to-run
#### test.py
This runs test evaluation script for action segmentation.
```
python test.py {config_yaml_file} {saved_checkpoint_dir} --feat_dirname {input_feature_dimension} --data_root_dir {data_root_directory}
```
Example:
```
python test.py ./configs/EgoPER/tea_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/method1/tea_aod_cspl_p2_neg2_bgr1.0_final --feat_dirname features_method1 --data_root_dir ./data
```
#### test_ed.py
This runs test evaluation script for error detection.
```
python test_ed.py {config_yaml_file} {saved_checkpoint_dir} --feat_dirname {input_feature_dimension} --data_root_dir {data_root_directory}
```
Example:
```
python test_ed.py ./configs/EgoPER/tea_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/method1/tea_aod_cspl_p2_neg2_bgr1.0_final --feat_dirname features_method1 --data_root_dir ./data
```