#python tools/test.py --cfg experiments/free/face_alignment_free_hrnet_w18.yaml --model-file HR18-WFLW.pth
#python tools/draw_result.py

#python tools/convert.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml --model-file HR18-WFLW.pth
#python tools/convert.py --cfg experiments/300w/face_alignment_300w_hrnet_w18.yaml --model-file HR18-WFLW.pth
#python tools/convert.py --cfg experiments/wflwe70/face_alignment_wflwe70_hrnet_w18.yaml --model-file HR18-WFLW.pth


#python tools/train.py --cfg experiments/wflwe70/face_alignment_wflwe70_hrnet_w18.yaml
python tools/test.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml --model-file HR18-WFLW.pth
