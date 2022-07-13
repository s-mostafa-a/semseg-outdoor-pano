import argparse
import numpy as np
from tqdm import tqdm
import torch
from dataloader import maverick_loader
from models.unet import UNet
from models.unet_equiconv import UNetEquiconv
from utils.metrics import Evaluator
from matplotlib import image as mat_image
from pathlib import Path

BATCH_SIZE = 4


class Test(object):
    """UNet-stdconv and UNet-equiconv test"""

    def __init__(self, args):
        self.args = args
        self.test_loader = maverick_loader(args)
        self.model = None

        if args.conv_type == 'Std':
            print("UNet-stdconv")
            self.model = UNet(args.num_classes)
            state_dict = torch.load("UNet-stdconv/stdconv.pth.tar")
        elif args.conv_type == "Equi":
            print("UNet-equiconv")
            layerdict, offsetdict = torch.load('UNet-equiconv/layer_256x512.pt'), torch.load(
                'UNet-equiconv/offset_256x512.pt')
            self.model = UNetEquiconv(args.num_classes, layer_dict=layerdict, offset_dict=offsetdict)
            state_dict = torch.load("UNet-equiconv/equiconv.pth.tar")
        else:
            raise Exception

        self.model.load_state_dict(state_dict)
        self.evaluator = Evaluator(args.num_classes)
        if args.cuda:
            self.model = self.model.cuda()
        Path(f"./results/{self.args.dataset_path.split('/')[-1]}").mkdir(parents=True, exist_ok=True)

    def run(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        with torch.no_grad():
            for _, sample in enumerate(tbar):
                image, file_name = sample['image'], sample['file_name']
                if self.args.cuda:
                    image = image.cuda()
                    output = self.model(image)

                pred = output.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                for bi in range(BATCH_SIZE):
                    mat_image.imsave(f"./results/{self.args.dataset_path.split('/')[-1]}/{file_name[bi]}.jpg",
                                     pred[bi, :, :])


def main():
    parser = argparse.ArgumentParser(description="PyTorch UNet-stdconv and UNet-equiconv")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_witdh", type=int, default=512)
    parser.add_argument("--conv_type", type=str, default='Std', choices=['Std', 'Equi'])
    parser.add_argument("--copy_weights", type=str, default=True)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--dataset_path', type=str, default="./dataset/Maverick/subset", help='path to dataset')
    args = parser.parse_args()
    test = Test(args)
    test.run()


if __name__ == "__main__":
    main()
