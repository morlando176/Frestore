print("DEBUG: train_unet.py script started execution.") # ADDED FOR DEBUGGING

# Wrap main logic in try-except to catch potential import or early setup errors
try:
    # === Imports ===
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    # Import specific augmentation transforms
    from torchvision.transforms import functional as TF
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import numpy as np
    import os
    import random
    import matplotlib.pyplot as plt
    import glob # For finding files
    import shutil # For potentially copying pre-resized files
    from collections import Counter # For counting class pixels

    print("--- Local U-Net Training Script (Handles Mixed Data, Augmentation, Class Weights) ---")

    # === Configuration ===
    # --- Paths ---
    SOURCE_IMAGES_DIR = "data/source_images/"
    SOURCE_MASKS_DIR = "data/source_masks/"
    RESIZED_IMAGES_DIR = "data/resized_images/"
    RESIZED_MASKS_DIR = "data/resized_masks/"
    MODEL_SAVE_DIR = "models/"
    MODEL_SAVE_NAME = "standard_unet_model_trained_v2.pth" # Changed name for new version
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)

    # --- Training Parameters ---
    INPUT_CHANNELS = 3
    NUM_CLASSES = 6 # Background, Shaft, Inner, ExposedG, CoveredG, Scrotal
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    BATCH_SIZE = 4 # Adjust based on your GPU memory
    NUM_EPOCHS = 50 # *** Consider increasing significantly (e.g., 100, 150, 200+) ***
    LEARNING_RATE = 1e-4

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === Step 1: Data Preparation Function ===
    # (resize_image, resize_mask, prepare_local_data functions remain the same)
    def resize_image(image_path, output_path, new_width, new_height):
        """Resizes a single image."""
        try: image = Image.open(image_path).convert("RGB"); resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS); resized_image.save(output_path); return True
        except Exception as e: print(f"❌ Error resizing image {image_path}: {e}"); return False

    def resize_mask(mask_path, output_path, new_width, new_height):
        """Resizes a single mask using nearest neighbor."""
        try:
            mask = Image.open(mask_path)
            if mask.mode != 'L' and mask.mode != 'P': mask = mask.convert('L')
            resized_mask = mask.resize((new_width, new_height), resample=Image.Resampling.NEAREST); resized_mask.save(output_path); return True
        except Exception as e: print(f"❌ Error resizing mask {mask_path}: {e}"); return False

    def prepare_local_data(source_img_dir, source_mask_dir, target_img_dir, target_mask_dir, target_w, target_h):
        """Ensures all images/masks from source dirs have a correctly sized version in the target dirs."""
        print("--- Preparing Local Data ---")
        os.makedirs(target_img_dir, exist_ok=True); os.makedirs(target_mask_dir, exist_ok=True)
        print(f"Ensuring target directories exist:\n  {target_img_dir}\n  {target_mask_dir}")
        if not os.path.isdir(source_img_dir): print(f"❌ ERROR: Source images directory not found: {source_img_dir}"); return False
        if not os.path.isdir(source_mask_dir): print(f"❌ ERROR: Source masks directory not found: {source_mask_dir}"); return False
        # Process Images
        print(f"\nProcessing images from: {source_img_dir}")
        resized_img_count = 0; skipped_img_count = 0; error_img_count = 0
        source_image_files = glob.glob(os.path.join(source_img_dir, "*.*"))
        for source_path in source_image_files:
            filename = os.path.basename(source_path)
            if filename.startswith('.') or not filename.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            target_path = os.path.join(target_img_dir, filename); needs_resizing = True
            if os.path.exists(target_path):
                try:
                    with Image.open(target_path) as img:
                        if img.size == (target_w, target_h): needs_resizing = False; skipped_img_count += 1
                except Exception as e: print(f"Warning: Could not read existing target image '{filename}'. Will overwrite. Error: {e}")
            if needs_resizing:
                if resize_image(source_path, target_path, target_w, target_h): resized_img_count += 1
                else: error_img_count += 1
        print(f"Image Processing Summary: Resized={resized_img_count}, Skipped/CorrectSize={skipped_img_count}, Errors={error_img_count}")
        # Process Masks
        print(f"\nProcessing masks from: {source_mask_dir}")
        resized_mask_count = 0; skipped_mask_count = 0; error_mask_count = 0
        source_mask_files = glob.glob(os.path.join(source_mask_dir, "*.png"))
        for source_path in source_mask_files:
            filename = os.path.basename(source_path)
            if filename.startswith('.'): continue
            target_path = os.path.join(target_mask_dir, filename); needs_resizing = True
            if os.path.exists(target_path):
                try:
                    with Image.open(target_path) as img:
                        if img.size == (target_w, target_h): needs_resizing = False; skipped_mask_count += 1
                except Exception as e: print(f"Warning: Could not read existing target mask '{filename}'. Will overwrite. Error: {e}")
            if needs_resizing:
                if resize_mask(source_path, target_path, target_w, target_h): resized_mask_count += 1
                else: error_mask_count +=1
        print(f"Mask Processing Summary: Resized={resized_mask_count}, Skipped/CorrectSize={skipped_mask_count}, Errors={error_mask_count}")
        if error_img_count > 0 or error_mask_count > 0: print("⚠️ WARNING: Errors occurred during resizing.")
        print("\n--- Data Preparation Finished ---")
        return True

    # === Step 2: Define Custom Dataset ===
    # *** UPDATED __getitem__ to include Augmentations ***
    class CustomDataset(Dataset):
        def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, augmentation=False,
                     image_extensions=('.jpg', '.jpeg', '.png'), mask_suffix='_mask', mask_extension='.png'):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.image_transform = transform # Renamed for clarity
            self.mask_transform = mask_transform
            self.augmentation = augmentation # Flag to control if augmentation is applied
            self.mask_suffix = mask_suffix
            self.mask_extension = mask_extension

            self.image_filenames = []
            self.mask_filenames = []

            print(f"\nDataset Init: Searching for images in {image_dir} with extensions {image_extensions}")
            all_image_files = []
            for ext in image_extensions: all_image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
            all_image_files.sort()
            print(f"Dataset Init: Found {len(all_image_files)} potential image files.")
            print("--- Starting Pairing Check ---")
            found_masks = 0
            for img_path in all_image_files:
                img_basename = os.path.splitext(os.path.basename(img_path))[0]
                expected_mask_name = f"{img_basename}{self.mask_suffix}{self.mask_extension}"
                mask_path = os.path.join(self.mask_dir, expected_mask_name)
                mask_exists = os.path.exists(mask_path)
                if mask_exists:
                    self.image_filenames.append(img_path)
                    self.mask_filenames.append(mask_path)
                    found_masks += 1
            print("--- Finished Pairing Check ---")
            print(f"Dataset Init: Found {len(self.image_filenames)} matching image/mask pairs using pattern '{self.mask_suffix}{self.mask_extension}'.")
            if len(self.image_filenames) == 0:
                 print("❌ ERROR: No matching image/mask pairs found!")

        def __len__(self):
            return len(self.image_filenames)

        def __getitem__(self, idx):
            img_path = self.image_filenames[idx]
            mask_path = self.mask_filenames[idx]

            try:
                image = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path) # Open mask

                # --- Data Augmentation ---
                # Apply augmentations that affect both image and mask identically first
                if self.augmentation:
                    # Example: Random Horizontal Flip
                    if random.random() > 0.5:
                        image = TF.hflip(image)
                        mask = TF.hflip(mask)

                    # Example: Random Rotation (use NEAREST for mask interpolation)
                    # angle = random.uniform(-10, 10) # degrees
                    # image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                    # mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

                # --- Mask Processing ---
                # Use your customized Scenario A or B logic here to get mask_np
                # !!! This example uses Scenario B - CUSTOMIZE class_color_map !!!
                class_color_map = {
                    0: [0, 0, 0], 1: [210, 180, 140], 2: [255, 192, 203],
                    3: [135, 206, 235], 4: [147, 112, 219], 5: [255, 255, 0]
                } # USE YOUR ACTUAL COLORS
                mask_rgb = np.array(mask.convert("RGB"))
                mask_np = np.full((mask_rgb.shape[0], mask_rgb.shape[1]), 0, dtype=np.int64)
                for class_idx, color in class_color_map.items():
                    if class_idx == 0: continue
                    matches = np.all(mask_rgb == np.array(color), axis=-1)
                    mask_np[matches] = class_idx
                # --- Verification ---
                min_val, max_val = np.min(mask_np), np.max(mask_np)
                if max_val >= NUM_CLASSES or min_val < 0:
                     print(f"❌ ERROR: Mask {os.path.basename(mask_path)} contains invalid class indices! Min: {min_val}, Max: {max_val}. Allowed: 0-{NUM_CLASSES-1}.")
                     placeholder_mask = torch.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.long)
                     if self.image_transform: image = self.image_transform(image)
                     else: image = transforms.ToTensor()(image)
                     return image, placeholder_mask
                # --- End Mask Processing ---

                # --- Apply Image-Specific Transforms (including ToTensor) ---
                if self.image_transform:
                    image = self.image_transform(image) # Apply ToTensor, ColorJitter etc.
                else: # Apply basic ToTensor if no other transform provided
                    image = transforms.ToTensor()(image)

                # --- Convert Processed Mask to Tensor ---
                mask = torch.from_numpy(mask_np).long() # Shape: (H, W)

                # --- Apply Mask Transform (if any) ---
                if self.mask_transform:
                     mask = self.mask_transform(mask) # Usually None

                return image, mask

            except Exception as e:
                print(f"❌ ERROR loading item {idx}: Image='{img_path}', Mask='{mask_path}'")
                print(f"   Error details: {e}")
                placeholder_image = torch.zeros((INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
                placeholder_mask = torch.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.long)
                return placeholder_image, placeholder_mask


    # === Step 3: Define Image Transforms ===
    # Now includes augmentations - apply only during training
    train_transform = transforms.Compose([
        # Augmentations that only affect the image:
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        # Convert PIL image [0, 255] to Tensor [0.0, 1.0] - MUST BE LAST for PIL transforms
        transforms.ToTensor(),
        # Add normalization IF your model was trained with it or you want to add it now
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define separate transform for validation/testing if needed (usually no augmentation)
    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize(...) # Use same normalization as training
    # ])


    # === Step 4: U-Net Model Definition ===
    # (Includes DoubleConv, Down, Up, OutConv, UNet classes - same as before)
    class DoubleConv(nn.Module):
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
    print("✅ Standard U-Net model architecture defined.")


    # === Step 5: Class Weight Calculation ===
    def calculate_class_weights(dataset, num_classes):
        """Calculates class weights based on inverse frequency."""
        print("\n--- Calculating Class Weights (may take a moment)... ---")
        pixel_counts = Counter()
        total_pixels = 0
        num_samples = len(dataset)
        if num_samples == 0:
            print("Warning: Dataset is empty, cannot calculate weights.")
            return None

        for i in range(num_samples):
            try:
                # Load only the mask to count pixels
                _, mask_tensor = dataset[i] # Get the processed mask tensor
                if mask_tensor is None: continue # Skip if loading failed

                mask_np = mask_tensor.numpy() # Convert tensor back to numpy
                unique, counts = np.unique(mask_np, return_counts=True)
                pixel_counts.update(dict(zip(unique, counts)))
                total_pixels += mask_np.size
            except Exception as e:
                print(f"Warning: Error processing mask {i} during weight calculation: {e}")
                continue # Skip problematic mask

        if total_pixels == 0:
            print("Warning: No valid pixels found in masks, cannot calculate weights.")
            return None

        print(f"Total pixels counted: {total_pixels}")
        print(f"Pixel counts per class: {dict(pixel_counts)}")

        # Calculate weights using inverse frequency balancing
        # weight = total_pixels / (num_classes * count_per_class)
        # Add smoothing (epsilon) to avoid division by zero for absent classes
        epsilon = 1e-6
        weights = []
        for i in range(num_classes):
            count = pixel_counts.get(i, 0) # Get count, default to 0 if class absent
            weight = total_pixels / (num_classes * (count + epsilon))
            weights.append(weight)

        # Normalize weights (optional, but can help stability)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / np.sum(weights) # Normalize to sum to 1
        weights = weights * num_classes # Scale so average weight is roughly 1

        print(f"Calculated Class Weights: {weights}")
        print("--- Class Weight Calculation Finished ---")
        return torch.tensor(weights, dtype=torch.float32)


    # === Step 6: Training Setup ===

    # --- Instantiate Model ---
    model = UNet(n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES, bilinear=True).to(device)
    print("✅ Standard U-Net model instantiated.")

    # --- Create Dataset and DataLoader FIRST ---
    # We need the dataset to calculate weights BEFORE defining the criterion
    train_dataset = None
    train_loader = None
    class_weights = None
    try:
        print(f"\nAttempting to create dataset from RESIZED dirs:\n  Images: {RESIZED_IMAGES_DIR}\n  Masks: {RESIZED_MASKS_DIR}")
        # Create dataset WITH augmentation enabled
        train_dataset = CustomDataset(
            image_dir=RESIZED_IMAGES_DIR,
            mask_dir=RESIZED_MASKS_DIR,
            transform=train_transform,
            augmentation=True # Enable augmentations defined in __getitem__
        )

        if len(train_dataset) > 0:
             # --- Calculate Class Weights ---
             class_weights = calculate_class_weights(train_dataset, NUM_CLASSES)

             # --- Create DataLoader ---
             train_loader = DataLoader(
                 train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                 num_workers=2, pin_memory=True # Adjust as needed
             )
             print(f"✅ DataLoader created with {len(train_dataset)} samples, batch size {BATCH_SIZE}.")
        else:
             print("❌ Dataset is empty after preparation. Cannot create DataLoader or calculate weights.")

    except Exception as e:
         print(f"❌ ERROR creating Dataset or DataLoader: {e}")


    # --- Loss Function (Now with weights) ---
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"Using loss function: {criterion} with calculated class weights.")
    else:
        print("Warning: Could not calculate class weights. Using standard CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()
        print(f"Using loss function: {criterion}")


    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Using optimizer: {optimizer}")


    # === Step 7: Training Loop ===
    # (run_training function remains the same, uses global criterion/optimizer/model)
    def run_training():
        print(f"--- Starting Training ---"); print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}"); print(f"Saving model to: {MODEL_SAVE_PATH}")
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        if not train_loader: print("❌ Training loop skipped because DataLoader is not available."); return False
        if os.path.exists(MODEL_SAVE_PATH):
             print(f"Found existing model weights at {MODEL_SAVE_PATH}. Loading state...")
             try: state_dict = torch.load(MODEL_SAVE_PATH, map_location=device); model.load_state_dict(state_dict); print("Successfully loaded existing weights.")
             except Exception as e: print(f"Warning: Could not load existing weights (error: {e}). Training from scratch.")
        print(f"Starting training on device: {device}")
        for epoch in range(NUM_EPOCHS):
            model.train(); epoch_loss = 0.0; batch_count = 0
            for i, (images, masks) in enumerate(train_loader):
                if images is None or masks is None or images.shape[0] == 0 or masks.shape[0] == 0 : print(f"Warning: Skipping potentially problematic batch at index {i}"); continue
                images = images.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
                try:
                    outputs = model(images)
                    if masks.ndim == 4 and masks.shape[1] == 1: masks = masks.squeeze(1)
                    if masks.dtype != torch.long: masks = masks.long()
                    loss = criterion(outputs, masks); optimizer.zero_grad(); loss.backward(); optimizer.step()
                    epoch_loss += loss.item(); batch_count += 1
                except Exception as batch_e: print(f"❌ ERROR during training step {i} in epoch {epoch+1}: {batch_e}"); print(f"   Image batch shape: {images.shape}, Mask batch shape: {masks.shape}"); continue
            if batch_count > 0: avg_epoch_loss = epoch_loss / batch_count; print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] Completed, Average Loss: {avg_epoch_loss:.4f} ---")
            else: print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] Completed, but no batches were processed successfully. ---")
        print("✅ Training completed!")
        try: torch.save(model.state_dict(), MODEL_SAVE_PATH); print(f"✅ Trained model state_dict saved successfully to {MODEL_SAVE_PATH}"); return True
        except Exception as e: print(f"❌ ERROR saving model: {e}"); return False

    # === Main Execution ===
    if __name__ == "__main__":
        # --- Prepare Data (Resize if needed) ---
        data_ready = prepare_local_data(
            source_img_dir=SOURCE_IMAGES_DIR,
            source_mask_dir=SOURCE_MASKS_DIR,
            target_img_dir=RESIZED_IMAGES_DIR,
            target_mask_dir=RESIZED_MASKS_DIR,
            target_w=IMAGE_WIDTH,
            target_h=IMAGE_HEIGHT
        )

        # --- Run Training if Data is Ready ---
        if data_ready:
            # Re-create dataset and dataloader (this was already done above, but ensures they exist)
            # This time, we ensure the dataset is created *before* the loss function needs the weights
            if train_loader: # Check if DataLoader was successfully created earlier
                 print("\nDataLoader already created. Proceeding to training.")
                 run_training() # Execute the training loop
            else:
                 # Attempt to create again if it failed earlier but data is now ready
                 print("\nAttempting to create dataset/dataloader again...")
                 try:
                     train_dataset = CustomDataset(
                         image_dir=RESIZED_IMAGES_DIR, mask_dir=RESIZED_MASKS_DIR,
                         transform=train_transform, augmentation=True
                     )
                     if len(train_dataset) > 0:
                          class_weights = calculate_class_weights(train_dataset, NUM_CLASSES) # Recalculate weights
                          if class_weights is not None:
                              criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) # Update criterion
                              print("Updated criterion with class weights.")
                          else:
                              criterion = nn.CrossEntropyLoss()
                              print("Using standard criterion (no weights).")

                          train_loader = DataLoader(
                              train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True
                          )
                          print("DataLoader created.")
                          run_training() # Execute the training loop
                     else:
                          print("❌ Dataset is empty after preparation. Cannot train.")
                 except Exception as e:
                      print(f"❌ ERROR setting up Dataset/DataLoader before training: {e}")

        else:
            print("❌ Data preparation failed. Skipping training.")

        print("--- Script Finished ---")

# === Top Level Exception Handler ===
except Exception as top_level_e:
    print("\n" + "="*30)
    print("❌ An unexpected error occurred at the top level:")
    import traceback
    traceback.print_exc() # Print the full traceback
    print("="*30 + "\n")

