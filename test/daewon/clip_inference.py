
import pickle
import os
import torch
import clip
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image


device = 'cuda'
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

vit = clip.load('ViT-L/14', device=device)[0]
vit.eval()

image_folder_path = r"D:\workspace\Dataset\my_room\long_frame"
image_save_folder_path = r"D:\workspace\Dataset\my_room\long_pre_image"
feature_save_folder_path = r"D:\workspace\Dataset\my_room\long_feature"

for i in range(3526):
    raw_image = Image.open(os.path.join(image_folder_path, f"{i}.png")).convert('RGB')

    vis_processor = transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor()])
    # vis_processor = transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    im1 = to_pil_image(vis_processor(raw_image)).save(os.path.join(image_save_folder_path,f"{i}.png")) 
    image = vis_processor(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = vit.encode_image(image)

        with open(file=os.path.join(feature_save_folder_path, f"{i}.pickle"), mode='wb') as f:
            pickle.dump(feature, f)

    print(f"{i} END")