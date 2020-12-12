
# MichiGAN: Multi-Input-Conditioned Hair Image Generation for Portrait Editing
![Teaser](data/teaser.jpg)
### [Paper](https://mlchai.com/files/tan2020michigan.pdf)

Zhentao Tan, [Menglei Chai](https://mlchai.com/), [Dongdong Chen](http://www.dongdongchen.bid/), [Jing Liao](https://liaojing.github.io/html/index.html), [Qi Chu](https://scholar.google.com/citations?user=JZjOMdsAAAAJ&hl=en), Lu Yuan, [Sergey Tulyakov](http://www.stulyakov.com/), [Nenghai Yu](https://scholar.google.com/citations?user=7620QAMAAAAJ&hl=zh-CN)

## Abstract
>Despite the recent success of face image generation with GANs, conditional hair editing remains challenging due to the under-explored complexity of its geometry and appearance. In this paper, we present MichiGAN (Multi-Input-Conditioned Hair Image GAN), a novel conditional image generation method for interactive portrait hair manipulation. To provide user control over every major hair visual factor, we explicitly disentangle hair into four orthogonal attributes, including shape, structure, appearance, and background. For each of them, we design a corresponding condition module to represent, process, and convert user inputs, and modulate the image generation pipeline in ways that respect the natures of different visual attributes. All these condition modules are integrated with the backbone generator to form the final end-to-end network, which allows fully-conditioned hair generation from multiple user inputs. Upon it, we also build an interactive portrait hair editing system that enables straightforward manipulation of hair by projecting intuitive and high-level user inputs such as painted masks, guiding strokes, or reference photos to well-defined condition representations. Through extensive experiments and evaluations, we demonstrate the superiority of our method regarding both result quality and user controllability.


## Installation

Clone this repo.
```bash
git clone https://github.com/tzt101/MichiGAN.git
cd MichiGAN/
```

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

Please download the Synchronized-BatchNorm-PyTorch rep.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

## Dataset Preparation

The FFHQ dataset can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1jI0EThBSgVRB_bgPype8pg) with the extracted code `ichc` or [OneDrive (RAR)](https://mailustceducn-my.sharepoint.com/:u:/g/personal/tzt_mail_ustc_edu_cn/ES2Ig_Nmmh1Jglv_T1VJzBgBbbxgdAnjDVVhJU1SzqIugA) or [OneDrive (ZIP)](https://mailustceducn-my.sharepoint.com/:u:/g/personal/tzt_mail_ustc_edu_cn/ES_I8Z09JZVJocoFvo-1aKEB3Ah7uI9C56JuMPMZhpXNqQ?e=9n61Lx), you should specify the dataset root from through `--data_dir`. Please follow the [license](https://github.com/NVlabs/ffhq-dataset) when you use the FFHQ dataset.

## Generating Images Using Pretrained Model

Once the dataset is ready, the result images can be generated using pretrained models.

1. Download the pretrained models from the [Google Drive Folder](https://drive.google.com/open?id=1Vxilcb82ax1Zlwy9wqHRu5-DCJuZFc_C), save it in 'checkpoints/MichiGAN/'. You can also download the pretrained models with the following commands:
```bash
cd checkpoints/MichiGAN/
bash download_model_G.sh
bash download_model_IG.sh
bash download_model_SIG.sh
```

2. Generate single image using the pretrained model.
    ```bash
    python inference.py --name MichiGAN --gpu_ids 0 --inference_ref_name 67172 --inference_tag_name 67172 --inference_orient_name 67172 --netG spadeb --which_epoch 50 --use_encoder --noise_background --expand_mask_be --expand_th 5 --use_ig --load_size 512 --crop_size 512 --add_feat_zeros --data_dir [path_to_dataset]
    ```
3. The outputs images are stored at `./inference_samples/` by default. If you just want to test this single image without download the whole dataset, please set `--data_dir ./datasets/FFHQ_single/`. We give a sample image (67172) here.

## Training New Models

New models can be trained with the following command.

```bash
python train.py --name [name_experiment] --batchSize 8 --no_confidence_loss --gpu_ids 0,1,2,3,4,5,6,7 --no_style_loss --no_rgb_loss --no_content_loss --use_encoder --wide_edge 2 --no_background_loss --noise_background --random_expand_mask --use_ig --load_size 568 --crop_size 512 --data_dir [pah_to_dataset] ----checkpoints_dir ./checkpoints
```
`[name_experiment]` is the directory name of the checkpoint file saved. if you want to train the model with orientation inpainting model (with the option --use_ig), please download the pretrained inpainting model from [Google Drive Folder](https://drive.google.com/open?id=1Vxilcb82ax1Zlwy9wqHRu5-DCJuZFc_C) and save them in `./checkpoints/[name_experiment]/` firstly.

## UI

You can direct run demo.py to use the Interactive systems. This UI code borrows from [MaskGAN](https://github.com/switchablenorms/CelebAMask-HQ.git).

## Orientation for New Dataset

Once the image and the corresponding hair mask is provided, you can use the following command to extract dense hair orientaiton map.
```bash
    python cal_orientation.py --image_path [your image path] --hairmask_path [you hair mask path] --orientation_root [save root]
```
For ease of use, we have rewritten the original c++ code into python. The results of this code are slightly different from the C++ version, but does not affect usage.

## Code Structure

- `train.py`, `inference.py`: the entry point for training and inferencing.
- `trainers/pix2pix_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading datas.

## Citation
If you use this code for your research, please cite our papers.
```
@article{tan2020michigan,
  title={MichiGAN: Multi-Input-Conditioned Hair Image Generation for Portrait Editing},
  author={Zhentao Tan, Menglei Chai, Dongdong Chen, Jing Liao, Qi Chu, Lu Yuan, Sergey Tulyakov and Nenghai Yu},
  journal={ACM Transactions on Graphics (TOG)},
  volume={39},
  number={4},
  pages={1--13},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```

## Acknowledgments
This code borrows heavily from [SPADE](https://github.com/NVlabs/SPADE.git). We thank Jiayuan Mao for his [Synchronized Batch Normalization](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) code.
