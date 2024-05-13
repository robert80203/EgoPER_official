# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/pinwheels_bgr1.0.yaml --output 1130
# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/pinwheels_cspl_bg2_np2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/pinwheels_bgr1.0_1130/epoch_105.pth.tar --output max10_1130 --maxvideos 10
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/pinwheels_gcn_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/pinwheels_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/pinwheels_gcn_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/quesadilla_bgr1.0.yaml --output 1201
# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/quesadilla_cspl_bg2_np2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/quesadilla_bgr1.0_1201/epoch_105.pth.tar --output max10_1201 --maxvideos 10
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/quesadilla_gcn_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/quesadilla_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/quesadilla_gcn_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/tea_bgr1.0.yaml --output 1201
# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/tea_cspl_bg2_np2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/tea_bgr1.0_1201/epoch_105.pth.tar --output max10_1201 --maxvideos 10
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/tea_gcn_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/tea_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/tea_gcn_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/oatmeal_bgr1.0.yaml --output 1201
# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/oatmeal_cspl_bg2_np2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/oatmeal_bgr1.0_1201/epoch_105.pth.tar --output max10_1201 --maxvideos 10
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/oatmeal_gcn_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/oatmeal_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/oatmeal_gcn_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/coffee_bgr0.1.yaml --output 1201
# CUDA_VISIBLE_DECVICES=4 python train.py ./configs/EgoPER/coffee_cspl_bg2_np2_neg2_bgr0.1.yaml --resume ./ckpt/EgoPER/coffee_bgr0.1_1201/epoch_105.pth.tar --output max5_1201 --maxvideos 5
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/coffee_gcn_bgr0.1.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/coffee_gcn_cspl_bg2_np2_neg2_bgr0.1.yaml --resume ./ckpt/EgoPER/coffee_gcn_bgr0.1_final/epoch_105.pth.tar --output final --maxvideos 5