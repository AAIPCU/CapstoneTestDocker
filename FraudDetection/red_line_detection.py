import cv2
import numpy as np
import glob

def extract_specific_color(image_path, target_bgr=(148, 135, 139), tolerance=30):
    """
    Extract pixels that match a specific color within a tolerance range.
    
    Args:
        image_path: Path to the input image
        target_bgr: Target color in BGR format (default: #8B8794)
        tolerance: Color difference tolerance (0-255). Can be a scalar or a tuple (B, G, R).
        
    Returns:
        Binary mask where matching pixels are white (255) and others are black (0)
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to numpy arrays
    target = np.array(target_bgr, dtype=np.float32)
    lower_bound = np.maximum(target - tolerance, 0).astype(np.uint8)
    upper_bound = np.minimum(target + tolerance, 255).astype(np.uint8)
    print(f"Base: {target}, Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    
    # Create mask for pixels within the color range
    mask = cv2.inRange(img, lower_bound, upper_bound)
    
    return mask


def extract_specific_color_result(image_path, target_bgr=(148, 135, 139), tolerance=30):
    """
    Extract pixels that match a specific color and return as color image.
    
    Args:
        image_path: Path to the input image
        target_bgr: Target color in BGR format (default: #8B8794)
        tolerance: Color difference tolerance (0-255). Can be a scalar or a tuple (B, G, R).
        
    Returns:
        Color image with only matching pixels preserved, others are black
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Get the binary mask
    mask = extract_specific_color(image_path, target_bgr, tolerance)
    
    # Apply mask to original image
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result


if __name__ == "__main__":
    # Example usage
    input_image_folder = "data/line/*"  # Change this to your image folder path
    
    # TARGET_COLOR_BGR = (191, 172, 185)  #B9ACBF in BGR format
    TARGET_COLOR_BGR = (119, 116, 130)  #777482 in BGR format
    # TARGET_COLOR_BGR = (109, 115, 142)  #8E736D in BGR format
    # Tolerance range: 10 for Blue/Green, 25 for Red (BGR order)
    COLOR_TOLERANCE = (45, 40, 45)

    # l = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    # for i in l:
    #     for j in l:
    #         for k in l:
    #             print(f"Testing tolerance: B={i}, G={j}, R={k}")
    #             try:
    #                 specific_mask = extract_specific_color(input_image, TARGET_COLOR_BGR, (i, j, k))
    #                 print(f"Success with tolerance B={i}, G={j}, R={k}")
    #                 cv2.imshow("Specific Color Mask (#8B8794)", specific_mask)
    #                 cv2.waitKey(0)
    #             except Exception as e:
    #                 print(f"Failed with tolerance B={i}, G={j}, R={k}: {e}")
    
    # get image from folder
    image_paths = glob.glob(input_image_folder)
    for input_image in image_paths:

        try:
            # Method 5: Specific color detection (binary mask) - #8B8794
            specific_mask = extract_specific_color(input_image, TARGET_COLOR_BGR, COLOR_TOLERANCE)
            cv2.imwrite(f"bin/specific_color_mask_{(input_image.split('.')[0].split('/')[-1].split('\\')[-1])}.jpg", specific_mask)
            print(f"Specific color mask saved (color={TARGET_COLOR_BGR}, tolerance={COLOR_TOLERANCE})")
            
            # Method 6: Specific color detection (color result)
            print(input_image.split('.')[0].split('/')[-1].split('\\')[-1])
            specific_color_result = extract_specific_color_result(input_image, TARGET_COLOR_BGR, COLOR_TOLERANCE)
            cv2.imwrite(f"bin/specific_color_result.jpg", specific_color_result)
            print("Specific color result saved as 'specific_color_result.jpg'")
            
            # Display the results
            cv2.imshow("Original", cv2.imread(input_image))
            cv2.imshow("Specific Color Mask (#8B8794)", specific_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error: {e}")
