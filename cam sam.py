import os
import cv2
import numpy as np
import torch
# from segment_anything import sam_model_registry, SamPredictor
from mobile_sam import sam_model_registry, SamPredictor


def isolate_main_object(input_folder, output_folder, checkpoint_path, model_type="vit_h"):
    print("Loading SAM model into GPU...")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        print(f"Processing: {filename}")
        filepath = os.path.join(input_folder, filename)
        
        image_bgr = cv2.imread(filepath)
        if image_bgr is None:
            continue
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        
        height, width = image_rgb.shape[:2]
        
        # ==========================================
        # STAGE 1: Smart Reconnaissance (The 5-Point Net)
        # ==========================================
        # Cast a net to catch objects even if they are off-center
        points_to_test = [
            [width // 2, height // 2],          # Dead Center
            [int(width * 0.35), height // 2],   # Left Center
            [int(width * 0.65), height // 2],   # Right Center
            [width // 2, int(height * 0.35)],   # Top Center
            [width // 2, int(height * 0.65)]    # Bottom Center
        ]
        
        candidate_masks = []
        
        # Gather guesses from all 5 points
        for pt in points_to_test:
            input_point = np.array([pt])
            input_label = np.array([1])
            
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            for i in range(len(masks)):
                candidate_masks.append({
                    'mask': masks[i],
                    'confidence': scores[i],
                })
                
        # ==========================================
        # STAGE 2: Filter and Score Candidates
        # ==========================================
        best_score = -999
        best_recon_mask = None
        total_area = width * height
        
        for candidate in candidate_masks:
            mask = candidate['mask']
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) == 0:
                continue
                
            area_ratio = len(y_coords) / total_area
            
            # Rule 1: Ignore tiny stickers (<3%) and massive background walls (>65%)
            if not (0.03 < area_ratio < 0.65):
                continue
                
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            
            # Rule 2: Calculate how far the object is from the center
            dist_x = abs(centroid_x - width / 2) / (width / 2)
            dist_y = abs(centroid_y - height / 2) / (height / 2)
            distance_penalty = (dist_x**2 + dist_y**2)**0.5
            
            # Final Score: Weigh AI confidence heavily, reward size, penalize extreme edges
            score = (candidate['confidence'] * 2.0) + (area_ratio * 0.8) - (distance_penalty * 1.5)
            
            if score > best_score:
                best_score = score
                best_recon_mask = mask
                
        # ==========================================
        # STAGE 3 & 4: Bounding Box Extraction (The Magic Fix)
        # ==========================================
        if best_recon_mask is not None:
            # We found the main object! Now find its outer edges.
            y_coords, x_coords = np.where(best_recon_mask)
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # Add 5% padding around the box so SAM doesn't cut off the cap or bottom
            pad_x = int(width * 0.05)
            pad_y = int(height * 0.05)
            
            box = np.array([
                max(0, x_min - pad_x),
                max(0, y_min - pad_y),
                min(width, x_max + pad_x),
                min(height, y_max + pad_y)
            ])
            
            # RE-PROMPT SAM: Feed it the bounding box. 
            # This forces SAM to grab the whole, solid object inside the box cleanly.
            final_masks, final_scores, _ = predictor.predict(
                box=box,
                multimask_output=True
            )
            
            # The highest confidence mask from a Box prompt is almost always perfect
            best_mask = final_masks[np.argmax(final_scores)]
            
        else:
            print(f"Warning: Could not clearly identify main object in {filename}. Falling back.")
            input_point = np.array([[width // 2, height // 2]])
            input_label = np.array([1])
            masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
            best_mask = masks[np.argmax(scores)]
            
        # ==========================================
        # Create Output Image
        # ==========================================
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[..., :3] = image_rgb
        rgba_image[..., 3] = best_mask * 255
        
        bgra_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
        
        output_filename = os.path.splitext(filename)[0] + "_isolated.png"
        output_filepath = os.path.join(output_folder, output_filename)
        
        cv2.imwrite(output_filepath, bgra_image)
        print(f"Successfully saved cleanly extracted object to: {output_filepath}")

if __name__ == "__main__":
    INPUT_DIR = "input_images"       
    OUTPUT_DIR = "output_images"     
    # CHECKPOINT = "sam_vit_h_4b8939.pth" # Ensures the heavy, accurate model is used
    CHECKPOINT = "mobile_sam.pt" # Ensures the heavy, accurate model is used
    
    # isolate_main_object(INPUT_DIR, OUTPUT_DIR, CHECKPOINT, model_type="vit_h")
    isolate_main_object(INPUT_DIR, OUTPUT_DIR, CHECKPOINT, model_type="vit_t")