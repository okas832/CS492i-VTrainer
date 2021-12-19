# Evaluate our model with dataset number 1 
python vtrainer_test.py --gpu 1 --dataset_num 1 --cls_weight ../weights/classfier/1/best_model.pt;
# Evaluate our model with dataset number 2
python vtrainer_test.py --gpu 1 --dataset_num 2 --cls_weight ../weights/classfier/2/best_model.pt;
# Evaluate our model with dataset number 3 
python vtrainer_test.py --gpu 1 --dataset_num 3 --cls_weight ../weights/classfier/3/best_model.pt;
# Evaluate only image feature model with dataset number 1
python vtrainer_train_only_feature.py --gpu 1 --dataset_num 1 --cls_weight ../weights/classfier/only_feature/best_model.pt --eval;
# Evaluate only image feature model with dataset number 1
python vtrainer_train_only_joint.py --gpu 1 --dataset_num 1 --cls_weight ../weights/classfier/only_joint/best_model.pt --eval;
