export PATH="/home/czh/anaconda3/bin:$PATH"

python tools/test.py --cfg experiments/free/face_alignment_free_hrnet_w18.yaml --model-file HR18-WFLW.pth
python tools/draw_result.py
