import torch
from utils import *
from PIL import Image
import glob
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srresnet_checkpoint = "./pth_6/checkpoint_srresnet.pth.tar"

# Load models
srresnet = torch.load(srresnet_checkpoint)["model"].to(device)
srresnet.eval()

path = "./data/testing_lr_images/"
imagenames = sorted(
    glob.glob(path + "*.png"),
    key=lambda x: [
        int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
    ],
)
# print(imagenames[:3])

for filename in imagenames:
    lr_img = Image.open(filename, mode="r")
    lr_img = lr_img.convert("RGB")

    sr_img_srresnet = srresnet(
        convert_image(lr_img, source="pil", target="imagenet-norm")
        .unsqueeze(0)
        .to(device)
    )
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(
        sr_img_srresnet, source="[-1, 1]", target="pil"
    )

    name = filename.split("/")[3]
    sname = name.split(".")[0]
    sr_img_srresnet.save(os.path.join("output_6/" + sname + "_pred.png"))
