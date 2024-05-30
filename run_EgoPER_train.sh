CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/pinwheels_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/pinwheels_aod_cspl_p4_neg4_bgr1.0.yaml --resume ./ckpt/EgoPER/pinwheels_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10
# CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/pinwheels_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/pinwheels_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/quesadilla_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/quesadilla_aod_cspl_p4_neg4_bgr1.0.yaml --resume ./ckpt/EgoPER/quesadilla_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10
# CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/quesadilla_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/quesadilla_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/tea_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/tea_aod_cspl_p4_neg4_bgr1.0.yaml --resume ./ckpt/EgoPER/tea_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10
# CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/tea_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/tea_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/oatmeal_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/oatmeal_aod_cspl_p4_neg4_bgr1.0.yaml --resume ./ckpt/EgoPER/oatmeal_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10
# CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/oatmeal_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/oatmeal_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/coffee_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/coffee_aod_cspl_p4_neg4_bgr1.0.yaml --resume ./ckpt/EgoPER/coffee_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10
# CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/coffee_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/coffee_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10