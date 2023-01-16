import numpy as np
from PIL import Image

if __name__ == '__main__':
    name = "img0049128.jpg"
    file = f"./results/semantic_segmentation/test_2_block_a/{name}"

    img = np.array(Image.open(file))
    print(np.unique(img))
