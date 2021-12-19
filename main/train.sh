# train our proposed model with dataset number 1
python vtrainer_train.py --gpu 1 --dataset_num 1 --extra_tag test_1
# train our proposed model with dataset number 2
python vtrainer_train.py --gpu 1 --dataset_num 2 --extra_tag test_2
# train our proposed model with dataset number 3
python vtrainer_train.py --gpu 1 --dataset_num 3 --extra_tag test_3
# train joint only model with dataset number 1
python vtrainer_train_only_joint.py --gpu 1 --dataset_num 1 --extra_tag only_joint/
# train image feature only model with dataset number 1
python vtrainer_train_only_feature.py --gpu 1 --dataset_num 1 --extra_tag only_feature/