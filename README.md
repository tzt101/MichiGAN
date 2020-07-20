[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# MichiGAN: Multi-Input-Conditioned Hair Image Generation for Portrait Editing


## Installation

Clone this repo.
```bash
git clone https://github.com/tzt101/HairSynthesisCode.git
cd hairSynthesis/
```

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

This code also requires the Synchronized-BatchNorm-PyTorch rep.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

## Dataset Preparation

The FFHQ dataset can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1jI0EThBSgVRB_bgPype8pg) with the extracted code `ichc`, you should specify the dataset root from through `--data_dir`.

## Generating Images Using Pretrained Model

Once the dataset is ready, the result images can be generated using pretrained models.

1. Download the pretrained models from the [Google Drive Folder](https://drive.google.com/open?id=1Vxilcb82ax1Zlwy9wqHRu5-DCJuZFc_C), save it in 'checkpoints/MichiGAN/'

2. Generate single image using the pretrained model.
    ```bash
    python inference.py --name MichiGAN --gpu_ids 0 --inference_ref_name 67172 --inference_tag_name 67172 --inference_orient_name 67172 --netG spadeb --which_epoch 50 --use_encoder --noise_background --expand_mask_be --expand_th 5 --use_ig --load_size 512 --crop_size 512 --add_feat_zeros --data_dir [path_to_dataset]
    ```
3. The outputs images are stored at `./inference_samples/` by default.

## Interactive System

If you want to generate images through interactive system, please run
```bash
python demo.py
```

## Training New Models

New models can be trained with the following command.

```bash
python train.py --name [name_experiment] --batchSize 8 --no_confidence_loss --gpu_ids 0,1,2,3,4,5,6,7 --no_style_loss --no_rgb_loss --no_content_loss --use_encoder --wide_edge 2 --no_background_loss --noise_background --random_expand_mask --use_ig --load_size 568 --crop_size 512 --data_dir [pah_to_dataset] ----checkpoints_dir ./checkpoints
```
`[name_experiment]` is the directory name of the checkpoint file saved. if you want to train the model with orientation inpainting model, please download the pretrained inpainting model from [Google Drive Folder](https://drive.google.com/open?id=1Vxilcb82ax1Zlwy9wqHRu5-DCJuZFc_C) and save them in `./checkpoints/[name_experiment]/` firstly.


## Code Structure

- `train.py`, `inference.py`: the entry point for training and inferencing.
- `trainers/pix2pix_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading datas.


## Acknowledgments
This code borrows heavily from SPADE. We thank Jiayuan Mao for his Synchronized Batch Normalization code.
