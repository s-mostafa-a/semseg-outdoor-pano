import argparse
import numpy as np
from tqdm import tqdm
import torch
from dataloader import maverick_loader, maverick_raw_loader
from models.unet import UNet
from models.unet_equiconv import UNetEquiconv
from utils.metrics import Evaluator
from matplotlib import image as mat_image
from pathlib import Path
from PIL import Image
from skimage.measure import label
from skimage.morphology import convex_hull_image

BATCH_SIZE = 48
ORIGINAL_IMAGE_WIDTH = 2000
ORIGINAL_IMAGE_HEIGHT = 1000


class MaverickTest(object):
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
        Path(f"./results/semantic_segmentation/{self.args.dataset_path.split('/')[-1]}").mkdir(parents=True,
                                                                                               exist_ok=True)
        Path(f"./results/moving_semantic_mask/{self.args.dataset_path.split('/')[-1]}").mkdir(parents=True,
                                                                                              exist_ok=True)

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
                self.do_saving_tasks(prediction=pred, file_names=file_name)

    def do_saving_tasks(self, prediction, file_names, save_moving_semantics=True, save_semantic_segmentation=False):
        for bi in range(prediction.shape[0]):
            res = prediction[bi, :, :]
            semantic_path = f"""./results/semantic_segmentation/{
            self.args.dataset_path.split('/')[-1]}/{file_names[bi]}.jpg"""
            moving_path = f"""./results/moving_semantic_mask/{
            self.args.dataset_path.split('/')[-1]}/{file_names[bi]}.jpg"""
            if save_semantic_segmentation:
                self.save_semantic_segmentation(img=res, path=semantic_path)
            if save_moving_semantics:
                self.save_moving_semantics_black_and_white(img=res, path=moving_path)

    @staticmethod
    def save_semantic_segmentation(img, path):
        mat_image.imsave(path, img, cmap='gray')

    @staticmethod
    def save_moving_semantics_black_and_white(img, path):
        img = np.where(img < 6, 0, 1)
        img = label(img)
        labels = np.unique(img)
        resulting = np.zeros_like(img)
        for lbl in labels:
            if lbl == 0:
                continue
            lbl_img = np.where(img == lbl, 1, 0)
            cvx_lbl = np.where(convex_hull_image(lbl_img) > 0, 1, 0)
            resulting += cvx_lbl
        img = Image.fromarray(np.uint8(resulting) * 255)
        img = img.resize((ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_HEIGHT), Image.Resampling.BILINEAR)
        img.save(path)

    def calculate_mean_and_std(self):
        raw_loader = maverick_raw_loader(self.args)

        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)
        tbar = tqdm(raw_loader, desc='\r')
        for _, sample in enumerate(tbar):
            images = sample['image']
            b, c, h, w = images.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2,
                                      dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
        return mean, std
    # test_2_block_a: (tensor([128.0318, 135.2806, 138.2639]), tensor([85.2109, 87.3013, 94.7959]))


def main():
    parser = argparse.ArgumentParser(description="PyTorch UNet-stdconv and UNet-equiconv")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_witdh", type=int, default=512)
    parser.add_argument("--conv_type", type=str, default='Std', choices=['Std', 'Equi'])
    parser.add_argument("--copy_weights", type=str, default=True)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--dataset_path', type=str, default="./dataset/Maverick/single", help='path to dataset')
    args = parser.parse_args()
    test = MaverickTest(args)
    test.run()
    # print(test.calculate_mean_and_std())


if __name__ == "__main__":
    main()
