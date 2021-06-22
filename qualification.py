import os
import utils
import torch
import torch.nn.functional as F
from options import Options
from my_transforms import get_transforms
from FullNet import FullNet
import torch.backends.cudnn as cudnn
from PIL import Image
import glob


# 得切1000*1000的
def main():
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    opt.print_options()

    img_dir = opt.test['img_dir']
    label_dir = opt.test['label_dir']  # 之所以用label instance是因为指标需要根据实例来计算
    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']
    tta = opt.test['tta']

    test_transform = get_transforms(opt.transform['test'])

    # load model
    model = FullNet(opt.model['in_c'], opt.model['out_c'], n_layers=opt.model['n_layers'],
                    growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                    dilations=opt.model['dilations'], is_hybrid=opt.model['is_hybrid'],
                    compress_ratio=opt.model['compress_ratio'], layer_type=opt.model['layer_type'])
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(best_checkpoint['epoch']))
    # model = model.module
    model.eval()

    img_normal = glob.glob('D:/multiorgan/qulication_data/normal_norm/*.png')
    for img_path in img_normal:
        # load test image
        img_name = os.path.basename(img_path).split('.')[0]
        print('=> Processing image {:s}'.format(img_name))
        img = Image.open(img_path)

        input = test_transform((img,))[0].unsqueeze(0)

        print('\tComputing output probability maps...')
        prob_maps = get_probmaps(input, model, opt)


def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda())
    else:
        output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])
    output = output.squeeze(0)
    prob_maps = F.softmax(output, dim=0).cpu().numpy()

    return prob_maps


if __name__ == '__main__':
    main()
