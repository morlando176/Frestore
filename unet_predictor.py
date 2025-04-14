import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt # Needed for saving visualized mask

# --- Configuration ---
MODEL_PATH = "models/unet_model_v2.onnx"
INPUT_HEIGHT = 256
INPUT_WIDTH = 256
# !!! CONFIRM THESE CLASS INDICES ARE CORRECT FOR YOUR TRAINING MASKS !!!
CLASS_BACKGROUND = 0
CLASS_SHAFT_SKIN = 1
CLASS_INNER_FORESKIN = 2
CLASS_EXPOSED_GLANS = 3
CLASS_COVERED_GLANS = 4
CLASS_SCROTAL_SKIN = 5
NUM_CLASSES = 6 # Should match model output

# --- Output Options ---
SAVE_PREDICTED_MASK = True # Set to True to save the mask image for inspection
PREDICTED_MASK_SAVE_PATH = "predicted_mask.png" # Saves in the project root

# --- Load ONNX model session ---
session = None
INPUT_NAME = None
OUTPUT_NAME = None
try:
    available_providers = ort.get_available_providers()
    provider = 'CUDAExecutionProvider' if 'CUDAExecutionProvider' in available_providers else 'CPUExecutionProvider'
    session = ort.InferenceSession(MODEL_PATH, providers=[provider])
    INPUT_NAME = session.get_inputs()[0].name
    OUTPUT_NAME = session.get_outputs()[0].name
    print(f"✅ ONNX Runtime session loaded successfully using {provider}.")
    print(f"   Input Name: {INPUT_NAME}, Output Name: {OUTPUT_NAME}")
except Exception as e:
    print(f"❌ ERROR loading ONNX model '{MODEL_PATH}': {e}")
    print("   Ensure the model file exists and onnxruntime is installed correctly.")

def preprocess_image(image_path):
    """Loads and preprocesses image for ONNX model."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.LANCZOS)
        img_np = np.array(img, dtype=np.float32) # Convert to float32
        img_np = img_np / 255.0 # Scale pixel values to 0-1
        # Normalize (Uncomment and use same values as training if needed)
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1) # Transpose axes from (H, W, C) to (C, H, W)
        input_tensor = np.expand_dims(img_np, axis=0) # Add batch dimension (1, C, H, W)
        return input_tensor
    except Exception as e:
        print(f"❌ Error preprocessing image {image_path}: {e}")
        return None

# *** CORRECTED save_mask_visualization function ***
def save_mask_visualization(mask_np, save_path):
    """Saves the predicted mask using a colormap for visual inspection."""
    try:
        # Using your provided colors
        # Ensure this list definition is syntactically correct
        colors = [
            [0, 0, 0],         # 0: Background
            [210, 180, 140],   # 1: Shaft Skin
            [255, 192, 203],   # 2: Inner Foreskin
            [135, 206, 235],   # 3: Exposed Glans
            [147, 112, 219],   # 4: Covered Glans
            [255, 255, 0]      # 5: Scrotal Skin (Other)
        ] # <-- Make sure this closing bracket is present and correct
        colors = np.array(colors[:NUM_CLASSES], dtype=np.uint8) # Ensure correct length

        # Create RGB image from mask indices
        height, width = mask_np.shape
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(NUM_CLASSES):
            # Ensure index i is within the bounds of the colors array
            if i < len(colors):
                 color_mask[mask_np == i] = colors[i]
            else:
                 print(f"Warning: Class index {i} found in mask, but no color defined. Using black.")
                 color_mask[mask_np == i] = [0,0,0] # Default to black if color missing


        img_to_save = Image.fromarray(color_mask)
        img_to_save.save(save_path)
        print(f"✅ Saved predicted mask visualization to: {save_path}")
    except Exception as e:
        print(f"❌ Error saving mask visualization: {e}")
# *** END OF CORRECTED function ***

def calculate_coverage_percentage(mask_np):
    """Calculates coverage based on class indices from NumPy mask."""
    try:
        pixels_exposed = np.sum(mask_np == CLASS_EXPOSED_GLANS)
        pixels_covered = np.sum(mask_np == CLASS_COVERED_GLANS)
        total_glans_pixels = pixels_exposed + pixels_covered

        print(f"DEBUG POSTPROCESS: Pixels Exposed (Class {CLASS_EXPOSED_GLANS}): {pixels_exposed}")
        print(f"DEBUG POSTPROCESS: Pixels Covered (Class {CLASS_COVERED_GLANS}): {pixels_covered}")
        print(f"DEBUG POSTPROCESS: Total Glans Pixels: {total_glans_pixels}")

        if total_glans_pixels == 0:
            print("Warning: No glans pixels (exposed or covered) detected in the mask.")
            return 0.0

        coverage_pct = (pixels_covered / total_glans_pixels) * 100
        return round(coverage_pct, 2)

    except Exception as e:
        print(f"❌ Error calculating coverage percentage: {e}")
        return None

def run_unet_analysis(image_path):
    """Runs the full U-Net analysis pipeline on a single image."""
    if session is None or INPUT_NAME is None or OUTPUT_NAME is None:
        print("❌ ONNX session not loaded. Cannot run analysis.")
        return None

    print(f"--- Running analysis on: {image_path} ---")
    input_tensor = preprocess_image(image_path)
    if input_tensor is None: return None

    try:
        print(f"Running ONNX inference...")
        outputs = session.run([OUTPUT_NAME], {INPUT_NAME: input_tensor})
        output_mask_logits = outputs[0]
        print(f"Inference successful. Output shape: {output_mask_logits.shape}")

        predicted_mask_np = np.argmax(output_mask_logits, axis=1).squeeze()
        print(f"Predicted mask generated. Shape: {predicted_mask_np.shape}")
        print(f"DEBUG POSTPROCESS: Unique values in predicted mask: {np.unique(predicted_mask_np)}")

        if SAVE_PREDICTED_MASK:
            # Ensure mask_np is passed correctly
            save_mask_visualization(predicted_mask_np, PREDICTED_MASK_SAVE_PATH)

        coverage_pct = calculate_coverage_percentage(predicted_mask_np)
        print(f"Calculated Coverage Percentage: {coverage_pct}")
        return coverage_pct

    except Exception as e:
        print(f"❌ Error during ONNX inference or postprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage (for testing this script directly)
if __name__ == '__main__':
    test_image_path = "test_image.png"
    try:
        print(f"Attempting to create dummy test image: {test_image_path}")
        dummy_img = Image.new('RGB', (300, 300), color = 'red')
        dummy_img.save(test_image_path)
        print("Running test analysis...")
        result = run_unet_analysis(test_image_path)
        print(f"\nTest Result: {result}")
        os.remove(test_image_path)
    except Exception as test_e:
        print(f"Error during test execution: {test_e}")

