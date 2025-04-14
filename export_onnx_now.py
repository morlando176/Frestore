# --- export_onnx_now.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import os # Added for path joining

print("Starting ONNX export process...")

# === Copied U-Net Model Definition ===
# --- U-Net Building Blocks ---
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """1x1 Convolution for final output mapping"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# --- The Full U-Net Model ---
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
# === End of Copied U-Net Model Definition ===


# === Export Configuration ===
N_CHANNELS = 3 # From your script analysis
N_CLASSES = 6  # From your script analysis
INPUT_HEIGHT = 256 # From your script analysis
INPUT_WIDTH = 256  # From your script analysis

# --->>> !!! IMPORTANT: SET THE CORRECT PATH TO YOUR TRAINED .pth FILE !!! <<<---
# Example: If it's in the same folder as this script and named 'standard_unet_model_trained.pth'
# PATH_TO_TRAINED_MODEL = r"standard_unet_model_trained.pth"
# Example: If it's elsewhere (use the full path)
PATH_TO_TRAINED_MODEL = r"C:\Users\marko\Desktop\foreskin_tracker\models\standard_unet_model_trained.pth" # <-- Make it look like this (or your actual .pth path)" # <-- CHANGE THIS

# --->>> Set the output path for the NEW .onnx file (points to your project folder) <<<---
OUTPUT_FOLDER = r"C:\Users\marko\Desktop\foreskin_tracker"
OUTPUT_ONNX_FILENAME = "unet_model.onnx"
OUTPUT_ONNX_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_ONNX_FILENAME)

# Ensure output directory exists (optional, good practice)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load Model ===
print(f"Loading trained model from: {PATH_TO_TRAINED_MODEL}")
if not os.path.exists(PATH_TO_TRAINED_MODEL):
    print(f"❌ ERROR: Trained model file not found at '{PATH_TO_TRAINED_MODEL}'. Please check the path.")
    exit() # Stop if the model file isn't found

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
try:
    model.load_state_dict(torch.load(PATH_TO_TRAINED_MODEL, map_location=device))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
except Exception as e:
    print(f"❌ ERROR loading model state_dict: {e}")
    exit() # Stop if loading fails

# === Create Dummy Input ===
dummy_input = torch.randn(1, N_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, requires_grad=False).to(device)
print(f"Created dummy input with shape: {dummy_input.shape}")

# === Export to ONNX ===
print(f"Exporting model to ONNX at: {OUTPUT_ONNX_PATH}")
try:
    torch.onnx.export(
        model,                     # model being run
        dummy_input,               # model input
        OUTPUT_ONNX_PATH,          # where to save the model
        export_params=True,        # store the trained parameter weights
        opset_version=12,          # ONNX version (11 is generally compatible)
        do_constant_folding=True,  # optimize
        input_names = ['input'],   # model's input names
        output_names = ['output'], # model's output names
        # Add dynamic axes if your model supports variable batch size, otherwise keep it simple
        # dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
    )
    print(f"✅ Model successfully exported to {OUTPUT_ONNX_PATH}")
except Exception as e:
    print(f"❌ ERROR during ONNX export: {e}")

print("ONNX export process finished.")