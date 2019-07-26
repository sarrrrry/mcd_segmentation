from PIL import Image
from torch.utils.data import Dataset as torch_dataset


class GTA5Dataset(torch_dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, item):
        img_path = "/raid/datasets/gta5/data/images/24966.png"
        lbl_path = "/raid/datasets/gta5/data/labels/24966.png"
        img = Image.open(str(img_path))
        lbl = Image.open(str(lbl_path))
        return img, lbl


if __name__ == '__main__':
    import numpy as np

    gta5 = GTA5Dataset()
    img, lbl = gta5.__getitem__(0)
    img = np.array(lbl)
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()
