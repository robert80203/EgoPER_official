CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/pinwheels_vc_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/pinwheels_vc_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/pinwheels_vc_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/quesadilla_vc_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/quesadilla_vc_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/quesadilla_vc_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/tea_vc_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/tea_vc_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/tea_vc_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/oatmeal_vc_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/oatmeal_vc_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/oatmeal_vc_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/coffee_vc_aod_bgr1.0.yaml --output final
CUDA_VISIBLE_DECVICES=3 python train.py ./configs/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr1.0.yaml --resume ./ckpt/EgoPER/coffee_vc_aod_bgr1.0_final/epoch_105.pth.tar --output final --maxvideos 10

CUDA_VISIBLE_DECVICES=5 python train.py ./configs/EgoPER/coffee_vc_aod_bgr0.1.yaml --output final
CUDA_VISIBLE_DECVICES=5 python train.py ./configs/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr0.1.yaml --resume ./ckpt/EgoPER/coffee_vc_aod_bgr0.1_final/epoch_105.pth.tar --output final --maxvideos 10