
import argparse
import os
import numpy as np
import torch
from PIL import Image
from openvino.runtime import Core
import torchvision.transforms as transforms

def get_imlist(path):  # 此函数读取特定文件夹下的PNG格式图像
    l=[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    l.sort()
    return l

def compare(imgA, imgH, imgM, imgS, imgV,modelpath):
    '''
    compare the input images' vectors with database's vectors by euclidean distance.
    :param imgA:
    :param imgH:
    :param imgM:
    :param imgS:
    :param imgV:
    :param modelpath: siamese model path (openvino IR xml)
    :return: normalized information score(weight)
    '''

    # average vectors calculated on training set.
    outputA = [[0.0114, -0.0330, 1.0054, 0.6460, 0.6708, -1.0433, -0.0167, 0.8037, -0.0294, 0.1007, -0.1259]]
    outputH = [[0.0067, -0.0217, 0.6217, 0.3929, 0.7517, -0.2383, -0.0192, 0.5947, -0.0313, 0.0873, -0.0907]]
    outputM = [[0.0070, -0.0286, 0.4332, 0.0616, 0.0924, -0.5519, -0.0078, 0.2260, -0.0299, 0.0976, -0.1033]]
    outputS = [[0.0079, 0.0062, 0.2653, 0.7307, 0.5512, -0.8934, -0.0228, 0.7778, -0.0571, 0.0553, 0.0155]]
    outputV = [[0.0046, 0.0032, -0.0391, 0.2752, 0.2284, -0.4139, -0.0181, 0.3642, -0.0509, 0.0594, 0.0068]]

    loader = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    #
    imgA = loader(imgA.convert("RGB")).unsqueeze(0)
    imgH = loader(imgH.convert("RGB")).unsqueeze(0)
    imgM = loader(imgM.convert("RGB")).unsqueeze(0)
    imgS = loader(imgS.convert("RGB")).unsqueeze(0)
    imgV = loader(imgV.convert("RGB")).unsqueeze(0)


    ie = Core()
    model_ir = ie.read_model(model=modelpath)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    # Get input and output layers
    # input_layer_ir = next(iter(compiled_model_ir.inputs))
    output_layer_ir = next(iter(compiled_model_ir.outputs))

    # Run inference on the input image
    res_irA = compiled_model_ir([imgA])[output_layer_ir]
    res_irH = compiled_model_ir([imgH])[output_layer_ir]
    res_irM = compiled_model_ir([imgM])[output_layer_ir]
    res_irS = compiled_model_ir([imgS])[output_layer_ir]
    res_irV = compiled_model_ir([imgV])[output_layer_ir]


    distance_H = np.sqrt(np.sum(np.square(outputH - res_irH)))
    distance_A = np.sqrt(np.sum(np.square(outputA - res_irA)))
    distance_S = np.sqrt(np.sum(np.square(outputS - res_irS)))
    distance_M = np.sqrt(np.sum(np.square(outputM - res_irM)))
    distance_V = np.sqrt(np.sum(np.square(outputV - res_irV)))

    sum = abs(distance_H) + abs(distance_A) + abs(distance_S) + abs(distance_M) + abs(distance_V)

    p_H = abs(distance_H) / sum
    p_A = abs(distance_A) / sum
    p_S = abs(distance_S) / sum
    p_M = abs(distance_M) / sum
    p_V = abs(distance_V) / sum
    # print('weights:',p_H, p_A, p_S, p_M, p_V)

    return p_A, p_H, p_M, p_S, p_V

def get_imlist(path):  # 此函数读取特定文件夹下的PNG格式图像
    l=[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    l.sort()
    return l

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--siamese', action='store_true', help="use siamese weight")
    parser.add_argument('--classi_model', type=str, default='./model.xml/embracenet.xml',
                        help='Openvino classification model path. \'xxx.xml\'')
    parser.add_argument('--siamese_model', type=str, default='./model.xml/siamese.xml',
                        help='Openvino siamese model path. \'xxx.xml\'')
    parser.add_argument('--test', type=str, default='./data/',
                        help='test data path')
    args, remaining_args = parser.parse_known_args()

    dic={0:'Normal',
         1:'MEMA',
         2:'SMCNL'}


    ie = Core()
    model_ir = ie.read_model(model=args.classi_model)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    # Get input and output layers
    # input_layer_ir = next(iter(compiled_model_ir.inputs))
    output_layer_ir = next(iter(compiled_model_ir.outputs))

    # data prepare
    path_A_list = get_imlist(os.path.join(args.test, 'A'))
    path_H_list = get_imlist(os.path.join(args.test, 'H'))
    path_M_list = get_imlist(os.path.join(args.test, 'M'))
    path_S_list = get_imlist(os.path.join(args.test, 'S'))
    path_V_list = get_imlist(os.path.join(args.test, 'V'))
    lens = len(path_H_list)


    for i in range(lens):
        print('No. img: ',i)
        print('Path of H: ', path_H_list[i])
        imgA = Image.open(path_A_list[i])
        imgH = Image.open(path_H_list[i])
        imgS = Image.open(path_S_list[i])
        imgM = Image.open(path_M_list[i])
        imgV = Image.open(path_V_list[i])


        pA, pH, pM, pS, pV = compare(imgA, imgH, imgM, imgS, imgV, args.siamese_model)
        # p = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float, device='cuda:0')
        p = torch.tensor([[pA, pH, pM, pS, pV]], dtype=torch.float, device='cuda:0')

        modality_indices = torch.multinomial(p, num_samples=512,
                                             replacement=True)

        imgA = (imgA.resize((56, 56), Image.Resampling.LANCZOS)).convert('L')  # resize, gray
        imgH = (imgH.resize((56, 56), Image.Resampling.LANCZOS)).convert('L')  # resize, gray
        imgS = (imgS.resize((56, 56), Image.Resampling.LANCZOS)).convert('L')  # resize, gray
        imgM = (imgM.resize((56, 56), Image.Resampling.LANCZOS)).convert('L')  # resize, gray
        imgV = (imgV.resize((56, 56), Image.Resampling.LANCZOS)).convert('L')  # resize, gray

        image_A = (np.asarray(imgA, dtype='float64') / 256.0) * 2.0 - 1.0
        image_H = (np.asarray(imgH, dtype='float64') / 256.0) * 2.0 - 1.0
        image_S = (np.asarray(imgS, dtype='float64') / 256.0) * 2.0 - 1.0
        image_M = (np.asarray(imgM, dtype='float64') / 256.0) * 2.0 - 1.0
        image_V = (np.asarray(imgV, dtype='float64') / 256.0) * 2.0 - 1.0

        input_data = [np.array([image_A]), np.array([image_H]), np.array([image_M]), np.array([image_S]),
                      np.array([image_V])]
        input_data = np.repeat(np.array([input_data]), repeats=1, axis=0)

        # Run inference on the input image
        res_ir = compiled_model_ir([modality_indices.cpu(),input_data])[output_layer_ir]

        # print(res_ir)
        # print(res_ir.argmax())
        print('Result:',dic[res_ir.argmax()])
        print('-------------------------------------------------------------')

main()