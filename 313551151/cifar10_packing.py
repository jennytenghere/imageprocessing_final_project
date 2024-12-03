import torch
import torchvision
import torchvision.transforms as transforms


def main():
    # Define transformation to convert CIFAR-10 images to tensor of type uint8
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte())
    ])

    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # Create a DataLoader for easy iteration
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    # Extract all images and labels
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Reshape images to desired format (N, H, W, 3)
    images = images.permute(0, 2, 3, 1)  # Convert from (N, 3, H, W) to (N, H, W, 3)

    # Save the tensor
    output_path = f"cifar10_data_{images.shape[0]}_{images.shape[1]}_{images.shape[2]}.pt"
    torch.save(images, output_path)
    print(f"Saved CIFAR-10 tensor of shape {images.shape} to {output_path}")


if __name__ == "__main__":
    main()