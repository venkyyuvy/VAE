import random
import torch
import matplotlib.pyplot as plt

def plot_image_grid(
        data, label, wrong_labels, 
        class_names,
        gray_image=True,
    ):

    batch_size = data.shape[0]
    num_rows = int(batch_size**0.5)
    num_cols = int(batch_size**0.5)
    _, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    for i in range(batch_size):
        row = i // num_cols 
        col = i % num_cols - 1 

        if gray_image:
            axs[row, col].imshow(data[i, 0], cmap='gray')
        else:
            axs[row, col].imshow(data[i].transpose(1, 2, 0))
        axs[row, col].set_title(
            f"{class_names[label]} | {class_names[wrong_labels[i]]}")
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Input_class | wrong_label", fontsize=16)


def plot_resconstr(model, batch: list, n_images: int=36):
    with torch.no_grad():
        X, y = batch
        idx = random.randint(0, X.shape[0] - 1)
        sample, y = X[idx: idx + 1], y[idx: idx + 1]
        random_y = torch.randint(0, 9, [n_images, ]).to(model.device)
        test_batch = sample[0].expand(
            n_images, model.x_channel, model.x_dim, model.x_dim)\
        .to(model.device), random_y
        reconstr = model(*test_batch)[0]\
            .view(n_images, model.x_channel, model.x_dim, model.x_dim)\
            .cpu().numpy()
        
        plot_image_grid(
            reconstr,
            y[0].item(),
            random_y,
            model.class_names,
            model.x_channel == 1,
        )
        plt.savefig(f'./samples/sample_{model.current_epoch}' + '.png')
        return 
