model_path = "outputs/efficientnet_b0/efficientnet_b0_best.pth"

name = model_path.split('/')[-1].split('.pth')[0]
print(name)