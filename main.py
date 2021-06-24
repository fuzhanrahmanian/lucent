from __future__ import absolute_import, division, print_function

import torch
import timm
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1, inception_v3
import lucent.modelzoo as mdz
import matplotlib.pyplot as plt
import torchvision.models as torchmods
from efficientnet_pytorch import EfficientNet
from transformers import AutoTokenizer, ViTForImageClassification
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from collections import OrderedDict


models = [
    EfficientNet.from_pretrained("efficientnet-b0"),
    torchmods.mobilenet_v2(pretrained=True),
    torchmods.shufflenet_v2_x1_0(pretrained=True),
    # torchmods.mobilenet_v2(pretrained=True),
    # torchmods.mobilenet_v2(pretrained=True),
    # torchmods.mobilenet_v2(pretrained=True),
]

layers = {"EfficientNet" : ['_blocks_13', '_blocks_14', '_blocks_15'], "MobileNetV2": ['features_7', 'features_10', 'features_14', 'features_17'],
            "shuffleNetV2": ["stage2_2","stage3_2","stage3_7", "stage4_3"]}

def get_model_layers(model, getLayerRepr=False):
    """
    If getLayerRepr is True, return a OrderedDict of layer names, layer representation string pair.
    If it's False, just return a list of layer names
    """
    layers = OrderedDict() if getLayerRepr else []
    # recursive function to get layers
    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if getLayerRepr:
                    layers["_".join(prefix+[name])] = layer.__repr__()
                else:
                    layers.append("_".join(prefix + [name]))
                get_layers(layer, prefix=prefix+[name])

    get_layers(model)
    return layers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
#model = ViTForImageClassification.from_pretrained("nateraw/baked-goods")
#model = inceptionv1(pretrained=True)
#model = models.inception_v3(pretrained=True)
#model = inception_v3(pretrained=True)
models[1].to(device).eval()
get_model_layers(models[1])
output_file = r"C:\Users\lucaz\Documents\Fuzhi\GitHub\lucent\experimental_results\shufflenetV2"
layers_name = ["stage2_2","stage3_2","stage3_7", "stage4_3"]
channel_num = ["7", "15", "65", "54"]
model_name = "shufflenetV2"
for layer, channel in zip(layers_name, channel_num):
    images, loss_record = render.render_vis(models[2], "{}:{}".format(layer, channel), verbose= True,
                                save_image=True, image_name= "{}/{}_{}.png".format(output_file, layer, channel),
                                show_inline=True)
    print(loss_record)
    # layer_name_comp = ["Mixed_6b", "Mixed_6c", "Mixed_6c", "Mixed_6d", "Mixed_6d"]
    # channel_num_comp = [651, 672, 30, 651, 42]
    # model_name = "inceptionV3"
    # output_file = r"C:\Users\lucaz\Documents\Fuzhi\GitHub\lucent\lucent_comparison"
    # for layer, channel in zip(layer_name_comp, channel_num_comp):
    #     images, loss_record = render.render_vis(model, "{}:{}".format(layer, channel), verbose= True,
    #                             save_image=True, image_name= "{}/{}_{}_{}.png".format(output_file, model_name, layer, channel),
    #                             show_inline=True)
    #     print(loss_record)



    plt.plot([loss_record[i] for i in range(1, len(loss_record), 2)])
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.savefig("{}/loss_{}_{}_{}.png".format(output_file, model_name, layer, channel))
    plt.close()

# Mixed_6b ||| mixed4
# 651

# torch: Mixed_6c ||| tf : mixed5
# channel num: 672, 30

# Mixed_6d ||| mixed6
# 651, 42