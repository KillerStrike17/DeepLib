import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

torch.manual_seed(1)


def get_device() -> tuple:
    """Get Device type

    Returns:
        tuple: Device type
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (use_cuda, device)


def print_summary(model, input_size=(1, 28, 28)):
    """Print Model summary

    Args:
        model (Net): Model Instance
        input_size (tuple, optional): Input size. Defaults to (1, 28, 28).
    """
    summary(model, input_size=input_size)


def print_modal_summary(model):
    """Print Model summary

    Args:
        model (Net): Model Instance
    """
    print(f'--------------------------------------------------------')
    print(f'| {"Name":25}\t{"Shape":15}\tParams |')
    print(f'--------------------------------------------------------')
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = np.prod(list(param.data.shape))
            total += count
            print(f'| {name:25}\t{str(list(param.data.shape)):15}\t{count:6} |')
    print(f'--------------------------------------------------------')
    print(f'| {"Total":25}\t{"":15}\t{total:6} |')
    print(f'--------------------------------------------------------')


def initialize_weights(m):
    """Function to initialize random weights

    Args:
        m (nn.Module): Layer instance
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


def load_weights_from_path(model, path):
    """load weights from file

    Args:
        model (Net): Model instance
        path (str): Path to weights file

    Returns:
        Net: loaded model
    """
    model.load_state_dict(torch.load(path))
    return model


def get_all_predictions(model, loader, device):
    """Get All predictions for model

    Args:
        model (Net): Trained Model 
        loader (Dataloader): instance of dataloader
        device (str): Which device to use cuda/cpu

    Returns:
        tuple: all predicted values and their targets
    """
    model.eval()
    all_preds = torch.tensor([]).to(device)
    all_targets = torch.tensor([]).to(device)
    with torch.no_grad():
        for data, target in loader:
            data, targets = data.to(device), target.to(device)
            all_targets = torch.cat(
                (all_targets, targets),
                dim=0
            )
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds = torch.cat(
                (all_preds, preds),
                dim=0
            )

    return all_preds, all_targets


def prepare_confusion_matrix(all_preds, all_targets, class_map):
    """Prepare Confusion matrix

    Args:
        all_preds (list): List of all predictions
        all_targets (list): List of all actule labels
        class_map (dict): Class names

    Returns:
        tensor: confusion matrix for size number of classes * number of classes
    """
    stacked = torch.stack((
        all_targets, all_preds
    ),
        dim=1
    ).type(torch.int64)

    no_classes = len(class_map)

    # Create temp confusion matrix
    confusion_matrix = torch.zeros(no_classes, no_classes, dtype=torch.int64)

    # Fill up confusion matrix with actual values
    for p in stacked:
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

    return confusion_matrix

def plot_predictions_gradcam(model,predictions, class_map,target_layers,use_cuda,count=10):
    """Plot Incorrect predictions

    Args:
        predictions (list): List of all incorrect predictions
        class_map (dict): Lable mapping
        count (int, optional): Number of samples to print, multiple of 5. Defaults to 10.
    """
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    # Denormalize the data using test mean and std deviation
    inv_normalize = transforms.Normalize(
        mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
        std=[1/0.23, 1/0.23, 1/0.23]
    )

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    classes = list(class_map.values())

    fig = plt.figure(figsize=(10, 5))
    for i, (d, t, p, o) in enumerate(predictions):
        orig_img = d.cpu().numpy()
        orig_img = np.transpose(orig_img, (1, 2, 0))
        input_image = orig_img
        transform_to_tensor = transforms.ToTensor()
        input_image = transform_to_tensor(input_image)
        input_image = input_image.unsqueeze(0)
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')
        grayscale_cam = cam(input_tensor=input_image)
        grayscale_cam = grayscale_cam[0, :]       
        val = inv_normalize(d.cpu()).numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(val, grayscale_cam, use_rgb=True,image_weight = 0.5)
        plt.imshow(visualization)
        if i+1 == 5*(count/5):
            break



def plot_predictions(predictions, class_map, correct = True,count=10):
    """Plot Incorrect predictions

    Args:
        predictions (list): List of all incorrect predictions
        class_map (dict): Lable mapping
        count (int, optional): Number of samples to print, multiple of 5. Defaults to 10.
    """
    if correct:
      print(f'Total correct Predictions {len(predictions)}')
    else:
      print(f'Total incorrect Predictions {len(predictions)}')
    # Denormalize the data using test mean and std deviation
    inv_normalize = transforms.Normalize(
        mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
        std=[1/0.23, 1/0.23, 1/0.23]
    )

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    classes = list(class_map.values())

    fig = plt.figure(figsize=(10, 5))
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')
        plt.imshow(inv_normalize(d.cpu()).numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break
        
def get_predictions(model, loader, device,correct = True):
    """Get all predictions

    Args:
        model (Net): Trained model
        loader (DataLoader): instance of data loader
        device (str): Which device to use cuda/cpu

    Returns:
        list: list of all incorrect predictions and their corresponding details
    """
    model.eval()
    vals = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == correct:
                    vals.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return vals