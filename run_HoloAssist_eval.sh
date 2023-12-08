# python test.py ./configs/HoloAssist/verb_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/HoloAssist/verb_cspl_bg2_np2_neg2_bgr1.0_max150_1205
# python metric_vis.py --dataset HoloAssist --task verb --dirname verb_cspl_bg2_np2_neg2_bgr1.0_max150_1205/ -as
# python test_ed.py ./configs/HoloAssist/verb_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/HoloAssist/verb_cspl_bg2_np2_neg2_bgr1.0_max150_1205
python metric_vis.py --dataset HoloAssist --task verb --dirname verb_cspl_bg2_np2_neg2_bgr1.0_max150_1205/ -ed
# python test.py ./configs/HoloAssist/noun_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/HoloAssist/noun_cspl_bg2_np2_neg2_bgr1.0_max150_1205
# python metric_vis.py --dataset HoloAssist --task noun --dirname noun_cspl_bg2_np2_neg2_bgr1.0_max150_1205/ -as
# python test_ed.py ./configs/HoloAssist/noun_cspl_bg2_np2_neg2_bgr1.0.yaml ./ckpt/HoloAssist/noun_cspl_bg2_np2_neg2_bgr1.0_max150_1205
python metric_vis.py --dataset HoloAssist --task noun --dirname noun_cspl_bg2_np2_neg2_bgr1.0_max150_1205/ -ed
