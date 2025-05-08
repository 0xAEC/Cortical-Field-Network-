# project-root/cfn/data/tasks.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os

class NoisyMNISTDataset(Dataset):
    """
    A dataset that wraps MNIST, adds noise to images, and returns (noisy_image, clean_image).
    """
    def __init__(self, mnist_root='./data/mnist_data', train=True, download=True, noise_factor=0.3):
        self.mnist_dataset = datasets.MNIST(
            root=mnist_root,
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)) # Normalizing makes target harder if noise is 0-1
            ])
        )
        self.noise_factor = noise_factor
        print(f"NoisyMNISTDataset created. Train={train}. Num samples: {len(self.mnist_dataset)}. Noise factor: {noise_factor}")


    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        clean_image, _ = self.mnist_dataset[idx] # (1, 28, 28), ignore label
        
        # Add Gaussian noise
        noisy_image = clean_image + self.noise_factor * torch.randn_like(clean_image)
        noisy_image = torch.clamp(noisy_image, 0., 1.) # Ensure values are in [0,1]
        
        return noisy_image, clean_image


def get_denoising_mnist_loaders(batch_size=64, noise_factor=0.3, data_root_dir='./data'):
    """
    Provides DataLoader instances for the NoisyMNISTDataset.
    """
    mnist_data_path = os.path.join(data_root_dir, 'mnist_data')
    os.makedirs(mnist_data_path, exist_ok=True)

    train_dataset = NoisyMNISTDataset(
        mnist_root=mnist_data_path,
        train=True,
        download=True,
        noise_factor=noise_factor
    )
    test_dataset = NoisyMNISTDataset(
        mnist_root=mnist_data_path,
        train=False,
        download=True,
        noise_factor=noise_factor
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train loader: {len(train_loader)} batches of size {batch_size}")
    print(f"Test loader: {len(test_loader)} batches of size {batch_size}")
    return train_loader, test_loader

if __name__ == '__main__':
    print("Testing NoisyMNISTDataset and DataLoaders...")
    train_loader, test_loader = get_denoising_mnist_loaders(batch_size=4, noise_factor=0.5)

    # Fetch a batch
    noisy_batch, clean_batch = next(iter(train_loader))
    print("Noisy batch shape:", noisy_batch.shape) # Expected: (4, 1, 28, 28)
    print("Clean batch shape:", clean_batch.shape) # Expected: (4, 1, 28, 28)
    assert noisy_batch.shape == (4, 1, 28, 28)
    assert clean_batch.shape == (4, 1, 28, 28)

    # # Optional: Visualize
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    # for i in range(4):
    #     axes[0, i].imshow(noisy_batch[i, 0].cpu().numpy(), cmap='gray')
    #     axes[0, i].set_title(f"Noisy {i+1}")
    #     axes[0, i].axis('off')
    #     axes[1, i].imshow(clean_batch[i, 0].cpu().numpy(), cmap='gray')
    #     axes[1, i].set_title(f"Clean {i+1}")
    #     axes[1, i].axis('off')
    # plt.tight_layout()
    # plt.show()

    print("Data loading test successful.")
