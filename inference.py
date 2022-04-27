import os


# path = './inference_data'
# infer_data_location = os.listdir(path)

# print(infer_data_location)
# os.system(f"CUDA_VISIBLE_DEVICES='0' python tools/infer.py configs/infer/tinaface/tinaface_r50_fpn_bn.py")
# CUDA_VISIBLE_DEVICES="0" python tools/infer.py configs/infer/tinaface/tinaface_r50_fpn_bn.py ./inference_data/42_Car_Racing_Nascar_42_333.jpg


# os.system(f"CUDA_VISIBLE_DEVICES='0' python tools/infer_Thread.py configs/infer/tinaface/tinaface_r50_fpn_bn.py")
#  file_name으로 하시면 file_name이미지 경로만 input으로 주면 값을 반환합니다.


os.system(f"CUDA_VISIBLE_DEVICES='0' python tools/infer_Thread.py configs/infer/tinaface/tinaface_r50_fpn_bn.py")