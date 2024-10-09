import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from skimage.metrics import structural_similarity

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3']):
        """
        Initializes the VGGFeatureExtractor.

        Args:
            layers (list): List of layer names from VGG19 to extract features from.
        """
        super(VGGFeatureExtractor, self).__init__()
        # Load the pre-trained VGG19 model
        vgg = models.vgg19(pretrained=True).features
        self.selected_layers = layers
        self.layer_names = {
            'conv1_1': 0,
            'conv1_2': 2,
            'conv2_1': 5,
            'conv2_2': 7,
            'conv3_1': 10,
            'conv3_2': 12,
            'conv3_3': 14,
            'conv3_4': 16,
            'conv4_1': 19,
            'conv4_2': 21,
            'conv4_3': 23,
            'conv4_4': 25,
            'conv5_1': 28,
            'conv5_2': 30,
            'conv5_3': 32,
            'conv5_4': 34,
        }
        # Include all layers up to index 35 (VGG19 has 36 layers in features)
        self.features = nn.ModuleList([vgg[i] for i in range(36)])
        
        # Set the model to evaluation mode
        self.eval()
        
        # Freeze the parameters to prevent training
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Extracts features from the input tensor x.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list: Extracted feature maps.
        """
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [self.layer_names[name] for name in self.selected_layers]:
                features.append(x)
        return features

def mse_loss(output, target):
    """
    Computes Mean Squared Error loss between output and target.

    Args:
        output (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.

    Returns:
        torch.Tensor: MSE loss.
    """
    return F.mse_loss(output, target)

def psnr_loss(output, target):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between output and target.

    Args:
        output (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.

    Returns:
        float: PSNR value.
    """
    mse = mse_loss(output, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(1 / mse.item())
    return psnr

def ssim_loss(output, target):
    """
    Computes Structural Similarity Index Measure (SSIM) loss between output and target.

    Args:
        output (torch.Tensor): Predicted tensor with shape (N, C, H, W).
        target (torch.Tensor): Ground truth tensor with shape (N, C, H, W).

    Returns:
        float: SSIM loss (1 - average SSIM over the batch).
        float: Average SSIM over the batch.
    """
    # Ensure the tensors are in CPU and detached from the computation graph
    output_np = output.permute(0, 2, 3, 1).detach().cpu().numpy()
    target_np = target.permute(0, 2, 3, 1).detach().cpu().numpy()
    ssim_total = 0
    for i in range(output_np.shape[0]):
        ssim = structural_similarity(
            output_np[i],
            target_np[i],
            multichannel=True,
            win_size=11,
            channel_axis=3,
            data_range=1.0
        )
        ssim_total += ssim
    average_ssim = ssim_total / output_np.shape[0]
    # Convert SSIM to a loss value
    return 1 - average_ssim, average_ssim

def perceptual_loss(output, target, feature_extractor):
    """
    Computes Perceptual Loss between output and target using a feature extractor.

    Args:
        output (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        feature_extractor (nn.Module): Feature extractor model.

    Returns:
        torch.Tensor: Perceptual loss.
    """
    output_features = feature_extractor(output)
    target_features = feature_extractor(target)
    loss = 0
    for of, tf in zip(output_features, target_features):
        loss += F.mse_loss(of, tf)
    return loss

def total_loss(output, target, feature_extractor, alpha=0.5):
    """
    Computes the total loss as a weighted sum of MSE loss and Perceptual loss.

    Args:
        output (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        feature_extractor (nn.Module): Feature extractor model.
        alpha (float): Weight for perceptual loss.

    Returns:
        tuple: (MSE loss, Perceptual loss, Total loss)
    """
    p_loss = perceptual_loss(output, target, feature_extractor)
    m_loss = mse_loss(output, target)
    total = alpha * p_loss + (1 - alpha) * m_loss
    return m_loss, p_loss, total
