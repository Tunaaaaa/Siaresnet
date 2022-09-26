# Siaresnet
A deep learning method for automatic evaluation of diagnostic information from multi-stained histopathological images

# Abstract
Manual screening of large-scale histopathological images is an extremely time-consuming, laborious and subjective procedure. Accurate evaluation of diagnostic information from multi-color stained images requires expertise due to the complex nature of histopathology and the lack of quantifiable measurement. In this work, a novel deep learning method is developed based on a convolutional siamese network, in which the information quantification task is transformed into a similarity assessment between lesion and non-lesion patterns on histopathological images. The subtle changes underlying the microstructure of tissue biopsies can be captured through an optimization of training loss within a low-to-high-level feature space. A new information score is introduced to quantify the abnormality in tissue appearance and stain pattern. Experiments on 3 independent data cohorts including 5 types of color-stained images demonstrate that our method can achieve promising performance compared with state-of-the-art methods. Results show that the proposed information score can serve as an effective measure to evaluate the importance of multi-stained images, and ultimately facilitate automatic diagnosis for clinical multi-stained histopathology.

**OpenVINO™** is a high-performance solution developed by Intel for deep learning deployments at the edge or in the cloud, enabling accelerated optimization of models. In smart medical scenarios, fast and timely data processing is crucial. This project builds a multi-stain pathology image information evaluation system based on the OpenVINO™ platform to achieve fast inference.

More information about **OpenVINO™** [https://docs.openvino.ai/latest/index.html](https://docs.openvino.ai/latest/index.html)

[Attention] Due to ethical requirements, we can only provide three groups of multi-stain image patches.

# How to use
## for inference

- in your favourite environment `pip install openvino`
- in your work directory `git clone https://github.com/Tunaaaaa/Siaresnet.git`
- `cd Siaresnet`
- Download models from [model.xml](https://1drv.ms/u/s!Aq_9-jGCnfL-kX0aIUfjjQ67-TFq?e=E0BkLY "model.xml"). Unzip and put them into /Siaresnet. If the link is not working, contact `jjy_ji@qq.com`
- `python infer.py --test <test data path>` The path should be similar to the example we provide, you can also set classi_model / siamese_model manually

## for training
to be continued

# To do
Release Pytorch training/infer code

# Questions and Suggestions
email: `jjy_ji@qq.com`