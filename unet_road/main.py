import unet_road
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

import torch

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "roads"
    
    ds = unet_road.RoadsDataset(path)

    model = unet_road.UNet()
    model.load_state_dict(torch.load('unet_road.pth'))
    
    img, mask = ds[0]
    img, mask = img, mask

    result = model(img.unsqueeze(0))

    mask = mask.detach().squeeze().cpu().numpy()
    result = result.detach().squeeze().cpu().numpy()
    result[result > 0] = 1
    result[result <= 0] = 0
    

    plt.subplot(131)
    plt.imshow(mask)
    plt.title('orig')
    plt.subplot(132)
    plt.imshow(result)
    plt.title('result')
    plt.subplot(133)
    plt.imshow(mask - result)
    plt.title('diff')
    
    plt.savefig('compare.png')
    plt.show()