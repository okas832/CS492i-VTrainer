# CS492i-VTrainer 

## How to run
### Train 
```
cd main
python vtrainer_train.py --gpu {gpu number} --dataset_num {split number}

For example 
python vtrainer_train.py --gpu 0 --dataset_num 1 

```
### Evaluation 
```
Will be updated
```
### For visualization
```
cd main
python vtrainer_train.py --gpu {gpu number} --dataset_num {split number} --visualization

```
### Create randomly splited datase
```
cd main
python split.py --number {number_tag}
```
### Install dependency
```
Will be updated
```
### Dataset
[Dataset download link](https://drive.google.com/drive/folders/1KlArONIR_a7sOjV_Z7iOLWCNh8liJxSu?usp=sharing)
[additional, bad, good] are raw dataset.
You can created train-test split using split.py
[train1, test1], [train2, test2], [train3, test3] are our split datasets. 

### Weights
You can download our classifier weights.
[Weights download link](https://drive.google.com/drive/folders/1rOdO8O3uxCYF4nAN-Q6THSXMelr-XyPF?usp=sharing)
1, 2, 3 are trained by train1, train2, train3, respectively.

Your weight should be in
  
'CS492i-VTrainer/weights/classifier/1/{weight name}.pt'
'CS492i-VTrainer/weights/classifier/2/{weight name}.pt'
'CS492i-VTrainer/weights/classifier/3/{weight name}.pt'
 
For rootnet and posenet,
You need to download both pretrained model weights.
[posenet](https://drive.google.com/drive/folders/1SKzmLk21mo3o24q_eB-4z7t1Sa9ozJ80?usp=sharing)
[rootnet](https://drive.google.com/drive/folders/1SKzmLk21mo3o24q_eB-4z7t1Sa9ozJ80?usp=sharing)

 Put the weight in 'CS492i-VTrainer/weights/{weight name}.pth.tar'


