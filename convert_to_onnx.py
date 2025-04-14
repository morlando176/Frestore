import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import os

# === Standard U-Net Model Definition ===
# (Model definition code remains the same - DoubleConv, Down, Up, OutConv, UNet)
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__();
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__(); self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__();
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1); diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1); return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(); self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__(); self.n_channels = n_channels; self.n_classes = n_classes; self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64); self.down1 = Down(64, 128); self.down2 = Down(128, 256); self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1; self.down4 = Down(512, 1024 // factor); self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear); self.up3 = Up(256, 128 // factor, bilinear); self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1); logits = self.outc(x); return logits

# === ONNX Conversion Settings ===
INPUT_CHANNELS = 3
NUM_CLASSES = 6
# *** UPDATED INPUT AND OUTPUT PATHS ***
MODEL_WEIGHTS_PATH = "models/standard_unet_model_trained_v2.pth" # Use the new weights file
OUTPUT_ONNX_PATH = "models/unet_model_v2.onnx"             # Save with a new ONNX name
# *************************************
INPUT_HEIGHT = 256
INPUT_WIDTH = 256
OPSET_VERSION = 12

def convert_model_to_onnx(n_channels, n_classes, weights_path, output_path, height, width, opset_v):
    """Loads trained PyTorch U-Net, converts to ONNX, saves."""
    print(f"Starting ONNX conversion...")
    print(f"Input Channels: {n_channels}, Num Classes: {n_classes}")
    print(f"Loading weights from: {weights_path}")
    print(f"Output ONNX path: {output_path}")
    print(f"Expected Input Size: ({height}, {width})")
    print(f"Using ONNX Opset Version: {opset_v}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True)
    print("Model instantiated.")

    device = torch.device('cpu')
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Successfully loaded model weights onto CPU.")
    except FileNotFoundError:
        print(f"❌ ERROR: Weights file not found at '{weights_path}'. Did training save correctly?")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to load weights: {e}")
        return

    model.eval()
    print("Model set to evaluation mode.")

    dummy_input = torch.randn(1, n_channels, height, width, requires_grad=False).to(device)
    print(f"Created dummy input tensor with shape: {dummy_input.shape}")

    try:
        print("Exporting model to ONNX...")
        torch.onnx.export(
            model, dummy_input, output_path, export_params=True,
            opset_version=opset_v, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes=None # Keep batch fixed for simplicity
        )
        print(f"✅ Model successfully converted and saved to '{output_path}'")

        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model check passed.")
        except ImportError: print("ℹ️ 'onnx' library not found. Skipping model verification.")
        except Exception as e: print(f"❌ ONNX model verification failed: {e}")

    except Exception as e:
        print(f"❌ ERROR during ONNX export: {e}")

if __name__ == "__main__":
    convert_model_to_onnx(
        n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES,
        weights_path=MODEL_WEIGHTS_PATH, output_path=OUTPUT_ONNX_PATH,
        height=INPUT_HEIGHT, width=INPUT_WIDTH, opset_v=OPSET_VERSION
    )
