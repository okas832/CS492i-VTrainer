# CS492i-VTrainer

  

This repository for CS492(I) final project.  

  

We implemented Work-out pose evaluator for personal training.  

  
  

## How to run

  

### Create data split

To split dataset into training and test images, you should run this code.

All parameters is hard-coded in code(i.e., input image dir path, output image dir, split ratio)

In this code, we manually set split ratio as 7:3.

```

cd main

python split.py --number {specified dataset number}

```

  

Since the directory path of images is hard-coded, you should put images in 'CS492i-VTrainer/dataset/'

```

For good plank images: 'CS492i-VTrainer/dataset/good/*.png' or 'CS492i-VTrainer/dataset/good/*.jpg'

For bad plank images: 'CS492i-VTrainer/dataset/bad/bad_plank/*.png' or 'CS492i-VTrainer/dataset/bad/bad_plank/*.jpg'

For no-plank images: 'CS492i-VTrainer/dataset/bad/bad_not_plank/*.png' or 'CS492i-VTrainer/dataset/bad/bad_plank/*.jpg'

```

  

The splited dataset will be saved at CS492i-VTrainer/dataset/

  

Training data consist of ...

```

'CS492i-VTrainer/dataset/train_{number}/good/'

'CS492i-VTrainer/dataset/train_{number}/bad/bad_not_plank'

'CS492i-VTrainer/dataset/train_{number}/bad/bad_plank'

```

  

Testing data

```

CS492i-VTrainer/dataset/test_{number}/bad/bad_plank/'

'CS492i-VTrainer/dataset/test_{number}/bad/bad_not_plank'

'CS492i-VTrainer/dataset/test_{number}/bad/bad_plank'

```

  

### Train

```

cd main

python vtrainer_train.py \

--gpu {specified yout gpu number} \

--dataset_num {split number} \

--extra_tag {specify output directory name} \

--lr {you can set learning rate(default:0.001)}

--batch_size {you can set batch size(default: 32)}

--epochs {you can set epochs(default:30)}

```

  

For example

```

python vtrainer_train.py --gpu 0 --dataset_num 1

```

Model weight will be saved at 'CS492i-VTrainer/weights/classfier/default/'

  

### Evaluation

```

python vtrainer_test.py --gpu {specified yout gpu number} --dataset_num {number of split} --cls_weight {classifier model path}

```

For example

```

python vtrainer_test.py --gpu 0 --dataset_num 1 --cls_weight ../weights/classfier/1/best_model.pt

```

  

### For visualization

```

cd main

python vtrainer_test.py --gpu {specified yout gpu number} --dataset_num {number of split} --cls_weight {classifier model path} --visualization

```

For example

```

python vtrainer_test.py --gpu 0 --dataset_num 1 --cls_weight ../weights/classfier/1/best_model.pt --visualization

```

  

### dependencies

We run at

```

python 3.6

pytorch 1.8.0

numpy 1.19.2

matplotlib 3.3.4

opencv-python 4.5.4.60

Pillow 8.4.0

PyYAML 6.0

requests 2.26.0

scipy 1.5.4

  

```

### Dataset

[Dataset download link](https://drive.google.com/drive/folders/1KlArONIR_a7sOjV_Z7iOLWCNh8liJxSu?usp=sharing)

[additional, bad, good] are raw dataset.

You can created train-test split using split.py

[train1, test1], [train2, test2], [train3, test3] are our split datasets.

  

Directory should be look like this.

'CS492i-VTrainer/dataset/additional'

'CS492i-VTrainer/dataset/bad'

'CS492i-VTrainer/dataset/good'

  

### Weights

You can download our classifier weights.

[classifier Weights download link](https://drive.google.com/drive/folders/1rOdO8O3uxCYF4nAN-Q6THSXMelr-XyPF?usp=sharing)

1, 2, 3 are trained by train1, train2, train3, respectively.

[best model for dataset 1](https://drive.google.com/file/d/18XKrtnpi0z9waNqkkRYDHLJx0SPODNCN/view?usp=sharing)

[best model for dataset 2](https://drive.google.com/file/d/1u4RM9q9fHXBsbeulG2JdmArYMW0-8tRL/view?usp=sharing)

[best model for dataset 3](https://drive.google.com/file/d/1wl-PftWBtoarU8KPM2kd1kRCX7tN8QQC/view?usp=sharing)

  

Your weight should be in

'CS492i-VTrainer/weights/classifier/1/best_model.pt'

'CS492i-VTrainer/weights/classifier/2/best_model.pt'

'CS492i-VTrainer/weights/classifier/3/best_model.pt'

For rootnet and posenet,

You need to download both pretrained model weights.

[posenet](https://drive.google.com/file/d/1GV60hVpKKRpXjBBbYJ6poWJeDl_Ln1fD/view?usp=sharing)

[rootnet](https://drive.google.com/file/d/1heFZLbm1GEMOEjvBB5Z3nvVREwYSPuGe/view?usp=sharing)

  

Put the weight in

'CS492i-VTrainer/weights/posenet_weight.pth.tar'

'CS492i-VTrainer/weights/rootnet_weight.pth.tar'

  

Weights for ablation study

[Only joints angle info](https://drive.google.com/file/d/1y_cdxolRRPfxiF3XOBLByCc0v7NktZMT/view?usp=sharing)

[Only image features](https://drive.google.com/file/d/1CHSMehTILtxEJOrWsQ6NZjB0HA6B9WzU/view?usp=sharing)

  

'CS492i-VTrainer/weights/classifier/only_feature/best_model.pt'

'CS492i-VTrainer/weights/classifier/only_joint/best_model.pt'

  

### We provide the conda list(library list) in conda_env.txt