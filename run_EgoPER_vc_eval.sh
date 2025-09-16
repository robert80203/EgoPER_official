python test.py ./configs/EgoPER/pinwheels_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/pinwheels_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task pinwheels --dirname pinwheels_vc_aod_cspl_p2_neg2_bgr1.0_final/ -as
python test_ed.py ./configs/EgoPER/pinwheels_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/pinwheels_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task pinwheels --dirname pinwheels_vc_aod_cspl_p2_neg2_bgr1.0_final/ -ed --threshold 0.0
python metric_vis.py --task pinwheels --dirname pinwheels_vc_aod_cspl_p2_neg2_bgr1.0_final/ -od

python test.py ./configs/EgoPER/quesadilla_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/quesadilla_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task quesadilla --dirname quesadilla_vc_aod_cspl_p2_neg2_bgr1.0_final/ -as --vis
python test_ed.py ./configs/EgoPER/quesadilla_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/quesadilla_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task quesadilla --dirname quesadilla_vc_aod_cspl_p2_neg2_bgr1.0_final/ -ed --threshold 0.0 --vis
python metric_vis.py --task quesadilla --dirname quesadilla_vc_aod_cspl_p2_neg2_bgr1.0_final/ -od

python test.py ./configs/EgoPER/tea_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/tea_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task tea --dirname tea_vc_aod_cspl_p2_neg2_bgr1.0_final/ -as
python test_ed.py ./configs/EgoPER/tea_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/tea_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task tea --dirname tea_vc_aod_cspl_p2_neg2_bgr1.0_final/ -ed --threshold 0.0
python metric_vis.py --task tea --dirname tea_vc_aod_cspl_p2_neg2_bgr1.0_final/ -od

python test.py ./configs/EgoPER/oatmeal_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/oatmeal_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task oatmeal --dirname oatmeal_vc_aod_cspl_p2_neg2_bgr1.0_final/ -as --vis
python test_ed.py ./configs/EgoPER/oatmeal_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/oatmeal_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task oatmeal --dirname oatmeal_vc_aod_cspl_p2_neg2_bgr1.0_final/ -ed --threshold 0.0 --vis
python metric_vis.py --task oatmeal --dirname oatmeal_vc_aod_cspl_p2_neg2_bgr1.0_final/ -od

python test.py ./configs/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task coffee --dirname coffee_vc_aod_cspl_p2_neg2_bgr1.0_final/ -as
python test_ed.py ./configs/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr1.0_final
python metric_vis.py --task coffee --dirname coffee_vc_aod_cspl_p2_neg2_bgr1.0_final/ -ed --threshold 0.0
python metric_vis.py --task coffee --dirname coffee_vc_aod_cspl_p2_neg2_bgr1.0_final/ -od

python test.py ./configs/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr0.1.yaml ./ckpt/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr0.1_final
python metric_vis.py --task coffee --dirname coffee_vc_aod_cspl_p2_neg2_bgr0.1_final/ -as
python test_ed.py ./configs/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr0.1.yaml ./ckpt/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr0.1_final
python metric_vis.py --task coffee --dirname coffee_vc_aod_cspl_p2_neg2_bgr0.1_final/ -ed --threshold 0.0
python metric_vis.py --task coffee --dirname coffee_vc_aod_cspl_p2_neg2_bgr0.1_final/ -od

# python test_ed.py ./configs/EgoPER/tea_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/tea_vc_aod_cspl_p2_neg2_bgr1.0_final --mode error_recognition
# python test_ed.py ./configs/EgoPER/pinwheels_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/pinwheels_vc_aod_cspl_p2_neg2_bgr1.0_final --mode error_recognition
# python test_ed.py ./configs/EgoPER/quesadilla_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/quesadilla_vc_aod_cspl_p2_neg2_bgr1.0_final --mode error_recognition
# python test_ed.py ./configs/EgoPER/oatmeal_vc_aod_cspl_p2_neg2_bgr1.0.yaml ./ckpt/EgoPER/oatmeal_vc_aod_cspl_p2_neg2_bgr1.0_final --mode error_recognition
# python test_ed.py ./configs/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr0.1.yaml ./ckpt/EgoPER/coffee_vc_aod_cspl_p2_neg2_bgr0.1_final --mode error_recognition

# python evaluation_as_ed_er.py --task tea --dirname tea_vc_aod_cspl_p2_neg2_bgr1.0_final/ --vis
# python evaluation_as_ed_er.py --task oatmeal --dirname oatmeal_vc_aod_cspl_p2_neg2_bgr1.0_final/ --vis
# python evaluation_as_ed_er.py --task pinwheels --dirname pinwheels_vc_aod_cspl_p2_neg2_bgr1.0_final/ --vis
# python evaluation_as_ed_er.py --task quesadilla --dirname quesadilla_vc_aod_cspl_p2_neg2_bgr1.0_final/ --vis
# python evaluation_as_ed_er.py --task coffee --dirname coffee_vc_aod_cspl_p2_neg2_bgr0.1_final/ --vis