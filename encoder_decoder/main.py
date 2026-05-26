from train import Encoder, Decoder, ImageDataset
import torch
import matplotlib.pyplot as plt

for curr_mode in [1, 2, 3, 4]:
    encoder = Encoder()
    decoder = Decoder()

    encoder.load_state_dict(torch.load(f"encoder_{curr_mode}.pth", map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(f"decoder_{curr_mode}.pth", map_location=torch.device('cpu')))

    dataset = ImageDataset(10, 256, mode=curr_mode)
    image, _ = dataset[0]

    with torch.no_grad():

        latent = encoder(image.unsqueeze(0))
        result = decoder(latent)
        plt.subplot(131)
        plt.imshow(image.squeeze().cpu().numpy())
        plt.subplot(132)
        plt.imshow(result.squeeze().cpu().detach().numpy())
        plt.subplot(133)
        plt.imshow(image.squeeze()-result.squeeze())
        plt.savefig(f"comparison_mode_{curr_mode}.png", dpi=300, bbox_inches='tight')
        plt.show()