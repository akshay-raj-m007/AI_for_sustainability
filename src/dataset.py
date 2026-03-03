import os
from PIL import Image
from torch.utils.data import Dataset


# This class:
#     - Stores image filenames and corresponding labels
#     - Loads images from disk when requested
#     - Applies transformations (augmentation + normalization)
#     - Returns (image_tensor, label)
class LandUseDataset(Dataset):
    def __init__(self, image_list, label_list, image_dir, transform=None):
        self.image_list = image_list     #List of image filenames
        self.label_list = label_list     #List of corresponding numeric labels
        self.image_dir = image_dir       #Directory where images are stored
        self.transform = transform       # torchvision.transforms
        
    def __len__(self):
        return len(self.image_list)   #Returns total number of samples in dataset.
    


    #  Steps(__getitem__):
    #     1. Get image filename
    #     2. Construct full file path
    #     3. Open image
    #     4. Convert to RGB
    #     5. Apply transforms
    #     6. Return image tensor + label
    def __getitem__(self, idx):     # Returns one sample at index 'idx'.
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        label = self.label_list[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label