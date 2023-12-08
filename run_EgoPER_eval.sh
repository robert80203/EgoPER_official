python test.py ./configs/EgoPER/pinwheels_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/pinwheels_cspl_bg2_np2_neg2_bgr1.0_max10_1130
python metric_vis.py --task pinwheels --dirname pinwheels_cspl_bg2_np2_neg2_bgr1.0_max10_1130/ -as
python test_ed.py ./configs/EgoPER/pinwheels_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/pinwheels_cspl_bg2_np2_neg2_bgr1.0_max10_1130
python metric_vis.py --task pinwheels --dirname pinwheels_cspl_bg2_np2_neg2_bgr1.0_max10_1130/ -ed -od
python test.py ./configs/EgoPER/pinwheels_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/pinwheels_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1130
python metric_vis.py --task pinwheels --dirname pinwheels_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1130/ -as
python test_ed.py ./configs/EgoPER/pinwheels_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/pinwheels_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1130
python metric_vis.py --task pinwheels --dirname pinwheels_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1130/ -ed -od

python test.py ./configs/EgoPER/quesadilla_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/quesadilla_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task quesadilla --dirname quesadilla_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -as
python test_ed.py ./configs/EgoPER/quesadilla_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/quesadilla_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task quesadilla --dirname quesadilla_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -ed -od
python test.py ./configs/EgoPER/quesadilla_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/quesadilla_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task quesadilla --dirname quesadilla_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -as
python test_ed.py ./configs/EgoPER/quesadilla_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/quesadilla_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task quesadilla --dirname quesadilla_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -ed -od

python test.py ./configs/EgoPER/tea_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/tea_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task tea --dirname tea_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -as
python test_ed.py ./configs/EgoPER/tea_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/tea_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task tea --dirname tea_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -ed -od
python test.py ./configs/EgoPER/tea_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/tea_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task tea --dirname tea_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -as
python test_ed.py ./configs/EgoPER/tea_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/tea_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task tea --dirname tea_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -ed -od

python test.py ./configs/EgoPER/oatmeal_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/oatmeal_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task oatmeal --dirname oatmeal_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -as
python test_ed.py ./configs/EgoPER/oatmeal_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/oatmeal_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task oatmeal --dirname oatmeal_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -ed -od
python test.py ./configs/EgoPER/oatmeal_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/oatmeal_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task oatmeal --dirname oatmeal_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -as
python test_ed.py ./configs/EgoPER/oatmeal_gcn_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/EgoPER/oatmeal_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201
python metric_vis.py --task oatmeal --dirname oatmeal_gcn_cspl_bg2_np2_neg2_bgr1.0_max10_1201/ -ed -od

python test.py ./configs/EgoPER/coffee_cspl_bg2_np2_neg2_bgr0.1.yaml ./ckpt/EgoPER/coffee_cspl_bg2_np2_neg2_bgr0.1_max5_1201
python metric_vis.py --task coffee --dirname coffee_cspl_bg2_np2_neg2_bgr0.1_max5_1201/ -as
python test_ed.py ./configs/EgoPER/coffee_cspl_bg2_np2_neg2_bgr0.1.yaml ./ckpt/EgoPER/coffee_cspl_bg2_np2_neg2_bgr0.1_max5_1201
python metric_vis.py --task coffee --dirname coffee_cspl_bg2_np2_neg2_bgr0.1_max5_1201/ -ed -od
python test.py ./configs/EgoPER/coffee_gcn_cspl_bg2_np2_neg2_bgr0.1.yaml ./ckpt/EgoPER/coffee_gcn_cspl_bg2_np2_neg2_bgr0.1_max5_1201
python metric_vis.py --task coffee --dirname coffee_gcn_cspl_bg2_np2_neg2_bgr0.1_max5_1201/ -as
python test_ed.py ./configs/EgoPER/coffee_gcn_cspl_bg2_np2_neg2_bgr0.1.yaml ./ckpt/EgoPER/coffee_gcn_cspl_bg2_np2_neg2_bgr0.1_max5_1201
python metric_vis.py --task coffee --dirname coffee_gcn_cspl_bg2_np2_neg2_bgr0.1_max5_1201/ -ed -od