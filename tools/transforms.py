from torchvision import transforms

# DOWNSCALE RESOLUTION TRANSFORMS

TRAIN_TRANSFORMS_DOWNSCALE = lambda downscale, upscale: transforms.Compose([
            transforms.Resize(downscale),
            transforms.Resize(upscale),
            transforms.RandomCrop(upscale, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

TEST_TRANSFORMS_DOWNSCALE = lambda downscale, upscale:transforms.Compose([
        transforms.Resize(downscale),
        transforms.Resize(upscale),
        transforms.CenterCrop(upscale),
        transforms.ToTensor()
    ])


# DEFAULT TRANSFORMS

TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
        ])

# BLACK AND WHITE TRANSFORMS

BLACK_N_WHITE_DOWNSCALE = lambda downscale, size: transforms.Compose([
        transforms.Resize(downscale),
        transforms.Resize(size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        ])

BLACK_N_WHITE = lambda size: transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        ])


