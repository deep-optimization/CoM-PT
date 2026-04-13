import torch

from tensorboard.backend.event_processing import event_accumulator
 
#加载日志数据
ea_baseline=event_accumulator.EventAccumulator('/home/ayao/events.out.tfevents.1724997715.vail01.2580001.0') 
ea_baseline.Reload()
# print(ea_baseline.scalars.Keys())
# assert 0 == 1


ea_ours_1=event_accumulator.EventAccumulator('/home/disk1/ayao/fjw/CLIP-KD/src/logs/2024_09_19-15_41_20-t_model_ViT-T-16-text-w256-s_model_ViT-B-16-lr_0.001-b_128-tag_distill-new/tensorboard/events.out.tfevents.1726731689.VAIL03.3493927.0') 
ea_ours_1.Reload()

ea_ours = event_accumulator.EventAccumulator('/home/disk1/ayao/fjw/CLIP-KD/src/logs/2024_09_14-11_22_21-t_model_ViT-T-16-text-w256-s_model_ViT-B-16-lr_0.001-b_128-tag_distill-new/tensorboard/events.out.tfevents.1726284150.VAIL03.3287409.0')
ea_ours.Reload()

baseline_in_1k_top1 = ea_baseline.scalars.Items('val/imagenet-zeroshot-val-top1')
ours_in_1k_top1 = ea_ours.scalars.Items('val/imagenet-zeroshot-val-top1')

# baseline_cc3m_i2t = ea_baseline.scalars.Items('val/image_to_text_R@1')
# ours_cc3m_i2t = ea_ours.scalars.Items('val/image_to_text_R@1')

baseline_cc3m_i2t = ea_baseline.scalars.Items('val/text_to_image_R@1')
ours_cc3m_i2t = ea_ours.scalars.Items('val/text_to_image_R@1')

print(len(baseline_in_1k_top1))
print(len(ours_in_1k_top1))


import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0,128,129)
x_s = np.linspace(0,32,33)

# print(x.shape)
# assert 0 == 1
print(baseline_in_1k_top1)
baseline_value = [0]
our_value = [0]

# for ele in baseline_in_1k_top1:
#     baseline_value.append(ele.value)
# for ele in ours_in_1k_top1:
#     our_value.append(ele.value)


# plt.plot(x, baseline_value,c = 'r', label='baseline')
# plt.plot(x_s, our_value,c = 'b', linestyle='dotted', label='ours')
# plt.legend()
# plt.savefig('./in-1k.png')

# plt

baseline_value = [0]
our_value = [0]
for ele in baseline_cc3m_i2t:
    baseline_value.append(ele.value)
for ele in ours_cc3m_i2t:
    our_value.append(ele.value)

plt.plot(x, baseline_value,c = 'r', label='baseline')
plt.plot(x_s, our_value,c = 'b', linestyle='dotted', label='ours')
plt.legend()
plt.savefig('./cc3m_1.png')

# print([(i.step,i.value) for i in val_psnr])