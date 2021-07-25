import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def preprocess(image, res=None):
    """
    Resizes, normalizes, and converts image to float32.
    """
    # resize image to model input size
    if res is not None:
        image = TF.resize(image, res)

    # convert image to float
    image = image.to(torch.float32) / 255

    # normalize image to default torchvision values
    image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    return image


def augment(image, rois):
    """
    Applies rotation, color jitter, and a flip.
    Runs *much* faster on GPU than CPU, so try to avoid using on CPU.
    """
    # horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0]

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image)

    # random rotation
    image, rois = random_image_rotation(image, rois, 15)
    
    return image, rois


def random_image_rotation(image, points, max_angle=30.0):
    """
    Randomly rotates an image and its annotations by [-max_angle, max_angle].
    Runs *much* faster on GPU than CPU, so try to avoid using on CPU.
    """
    device = image.device
    
    # check that the points are within [0, 1]
    assert points.min() >= 0, points.min()
    assert points.max() <= 1, points.max()

    # generate random rotation angle in range [-max_angle, max_angle]
    angle_deg = (2*torch.rand(1).item() - 1) * max_angle
    
    # rotate the image and note the change in resolutions
    _, H1, W1 = image.shape
    image = TF.rotate(image, angle_deg, expand=True)
    _, H2, W2 = image.shape
    
    # create rotation matrix
    angle_rad = torch.tensor((angle_deg / 180.0) * 3.141592)
    R = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)],
                      [torch.sin(angle_rad),  torch.cos(angle_rad)]], dtype=torch.float, device=device)
    
    # move points to an absolute cooridnate system with [0, 0] as the center of the image
    points = points.clone()
    points -= 0.5
    points[..., 0] *= (W1 - 1)
    points[..., 1] *= (H1 - 1)
    
    # rotate the points
    points = points @ R
    
    # move points back to the relative coordinate system
    points[..., 0] /= (W2 - 1)
    points[..., 1] /= (H2 - 1)
    points += 0.5
    
    # check that the points remain within [0, 1]
    assert points.min() >= 0, points.min()
    assert points.max() <= 1, points.max()
    
    return image, points