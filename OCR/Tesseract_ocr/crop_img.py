import cv2
import numpy as np
from typing import List, Optional, Tuple
import json
from pathlib import Path

DEFAULT_SIFT_RATE = 25000
FLANN_INDEX_KDTREE = 0
current_dir = Path(__file__).parent
config_path = str(current_dir / "config.json")
template_path = str(current_dir / "test_data/idcard_front_template.jpg")

def read_img(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at path: {image_path}")
    return img


def create_feature_matcher(sift_rate: int = DEFAULT_SIFT_RATE):
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    sift = cv2.SIFT_create(sift_rate)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return sift, flann


def load_template(template_path: str, sift) -> Tuple[np.ndarray, List[cv2.KeyPoint], np.ndarray]:
    template_img = read_img(template_path)
    template_kp, template_des = sift.detectAndCompute(template_img, None)
    return template_img, template_kp, template_des


def compare_template_similarity(
    flann,
    query_des: np.ndarray,
    train_des: np.ndarray,
    template_threshold: float,
) -> List[cv2.DMatch]:
    if query_des is None or train_des is None:
        return []
    matches = flann.knnMatch(query_des, train_des, k=2)
    good_matches: List[cv2.DMatch] = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < template_threshold * n.distance:
            good_matches.append(m)
    return good_matches


def find_and_warp_object(
    image: np.ndarray,
    template_shape: Tuple[int, int],
    matches: List[cv2.DMatch],
    process_kp: List[cv2.KeyPoint],
    template_kp: List[cv2.KeyPoint],
    flann,
    process_des: np.ndarray,
    template_des: np.ndarray,
    relaxed_threshold: float,
    min_matches: int,
    reproj_thresh: float,
    min_inliers: int,
    margin: int,
) -> np.ndarray:
    tpl_h, tpl_w = template_shape

    good_matches = matches
    if len(good_matches) < min_matches:
        relaxed_matches = compare_template_similarity(flann, process_des, template_des, relaxed_threshold)
        if len(relaxed_matches) > len(good_matches):
            good_matches = relaxed_matches

    if not good_matches or len(good_matches) < 4:
        return image

    proc_pts = np.float32([process_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([template_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    methods = [(cv2.RANSAC, reproj_thresh * 0.5), (cv2.LMEDS, reproj_thresh)]
    best_M = None
    best_mask = None
    best_inliers = 0

    for method, method_thresh in methods:
        try:
            temp_M, temp_mask = cv2.findHomography(proc_pts, src_pts, method, method_thresh)
        except cv2.error:
            temp_M, temp_mask = None, None
        if temp_M is None or temp_mask is None:
            continue
        inliers = int(temp_mask.sum())
        if inliers > best_inliers:
            best_M = temp_M
            best_mask = temp_mask
            best_inliers = inliers

    if best_M is None or best_inliers < min_inliers:
        try:
            temp_M, temp_mask = cv2.findHomography(proc_pts, src_pts, cv2.RANSAC, reproj_thresh * 2)
        except cv2.error:
            temp_M, temp_mask = None, None
        if temp_M is not None:
            best_M = temp_M
            best_mask = temp_mask
            best_inliers = int(temp_mask.sum()) if temp_mask is not None else 0

    if best_M is None:
        return image

    dst_w, dst_h = tpl_w + margin * 2, tpl_h + margin * 2
    if margin > 0:
        translation = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], dtype=np.float64)
        best_M = translation @ best_M

    try:
        warped = cv2.warpPerspective(
            image,
            best_M,
            (dst_w, dst_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except cv2.error:
        return image

    if margin > 0:
        warped = warped[margin:margin + tpl_h, margin:margin + tpl_w]

    return warped


def crop_img(
    image: np.ndarray,
    output_path: Optional[str] = None,
    sift_rate: int = DEFAULT_SIFT_RATE,
    template_threshold: float = 0.7,
    relaxed_threshold: float = 0.85,
    min_matches: int = 10,
    reproj_thresh: float = 3.0,
    min_inliers: int = 8,
    margin: int = 10,
) -> np.ndarray:
    # image = read_img(img_path)
    sift, flann = create_feature_matcher(sift_rate)
    template_img, template_kp, template_des = load_template(template_path, sift)
    tpl_h, tpl_w = template_img.shape[:2]

    process_kp, process_des = sift.detectAndCompute(image, None)

    matches = compare_template_similarity(flann, process_des, template_des, template_threshold)

    warped = find_and_warp_object(
        image,
        (tpl_h, tpl_w),
        matches,
        process_kp,
        template_kp,
        flann,
        process_des,
        template_des,
        relaxed_threshold,
        min_matches,
        reproj_thresh,
        min_inliers,
        margin,
    )

    if output_path:
        cv2.imwrite(output_path, warped)
        # print("Saved cropped image to:", output_path)

    return warped

def crop_img_save(img, iqs, save_path=None): 
    # keep original and resized separately so we can restore crop size correctly
    # orig_img = cv2.imread(img_path)
    orig_img = crop_img(img)
    resized_img, _ = iqs.resize_image_to_fit_screen(orig_img.copy())
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    cropped_images = {}
    for item in config:
        x1,y1,x2,y2 = config[item]['points']
        img = resized_img[y1:y2, x1:x2]
        re_img = iqs.improveQuality(img)
        if item == 'image' :
            continue
        if save_path:
            cv2.imwrite(f"{save_path}/{item}.png", re_img)
        cropped_images[item] = re_img
    
    return cropped_images

if __name__ == "__main__":
    img = read_img("C:\\Users\\User\\OneDrive\\Desktop\\Aomsin\\year3\\capstone\\train\\test_data\\idcard17.jpg")
    preview = crop_img( image=img, output_path="C:\\Users\\User\\OneDrive\\Desktop\\Aomsin\\year3\\capstone\\train\\test_data\\idcard17_cropped.jpg")
    cv2.imshow("cropped", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()