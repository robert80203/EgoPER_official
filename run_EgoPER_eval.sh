python test_ed.py ./configs/EgoPER/pinwheels_aod_cspl_p4_neg4_bgr1.0.yaml ./ckpt/EgoPER/pinwheels_aod_cspl_p4_neg4_bgr1.0_final
python metric_vis.py --task pinwheels --dirname pinwheels_aod_cspl_p4_neg4_bgr1.0_final/ -ed
python test_ed.py ./configs/EgoPER/pinwheels_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/pinwheels_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task pinwheels --dirname pinwheels_aod_cspl_p2_neg2_bgr1.0_final/ -ed

python test_ed.py ./configs/EgoPER/quesadilla_aod_cspl_p4_neg4_bgr1.0.yaml ./ckpt/EgoPER/quesadilla_aod_cspl_p4_neg4_bgr1.0_final
python metric_vis.py --task quesadilla --dirname quesadilla_aod_cspl_p4_neg4_bgr1.0_final/ -ed
python test_ed.py ./configs/EgoPER/quesadilla_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/quesadilla_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task quesadilla --dirname quesadilla_aod_cspl_p2_neg2_bgr1.0_final/ -ed

python test_ed.py ./configs/EgoPER/tea_aod_cspl_p4_neg4_bgr1.0.yaml ./ckpt/EgoPER/tea_aod_cspl_p4_neg4_bgr1.0_final
python metric_vis.py --task tea --dirname tea_aod_cspl_p4_neg4_bgr1.0_final/ -ed
python test_ed.py ./configs/EgoPER/tea_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/tea_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task tea --dirname tea_aod_cspl_p2_neg2_bgr1.0_final/ -ed

python test_ed.py ./configs/EgoPER/oatmeal_aod_cspl_p4_neg4_bgr1.0.yaml ./ckpt/EgoPER/oatmeal_aod_cspl_p4_neg4_bgr1.0_final
python metric_vis.py --task oatmeal --dirname oatmeal_aod_cspl_p4_neg4_bgr1.0_final/ -ed
python test_ed.py ./configs/EgoPER/oatmeal_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/oatmeal_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task oatmeal --dirname oatmeal_aod_cspl_p2_neg2_bgr1.0_final/ -ed

python test_ed.py ./configs/EgoPER/coffee_aod_cspl_p4_neg4_bgr1.0.yaml ./ckpt/EgoPER/coffee_aod_cspl_p4_neg4_bgr1.0_final
python metric_vis.py --task coffee --dirname coffee_aod_cspl_p4_neg4_bgr1.0_final/ -ed
python test_ed.py ./configs/EgoPER/coffee_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/coffee_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task coffee --dirname coffee_aod_cspl_p2_neg2_bgr1.0_final/ -ed