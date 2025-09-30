
import sys
sys.path.append('/home/dex/Open CV project/speedplusbaseline')

import torch
import cv2
import numpy as np
import scipy.io
import json
import os
from torchvision import transforms
from src.nets.park2019 import KeypointRegressionNet

# --- Configuration ---
MODEL_PATH = '/home/dex/Open CV project/speedplusbaseline/checkpoints/krn/synthetic_only/model_best.pth.tar'
TANGO_POINTS_PATH = '/home/dex/Open CV project/speedplusbaseline/src/utils/tangoPoints.mat'
CAMERA_PARAMS_PATH = '/home/dex/Open CV project/speedplusv2/camera.json'
LIGHTBOX_IMAGE_DIR = '/home/dex/Open CV project/speedplusv2/lightbox/images'
LIGHTBOX_LABELS_PATH = '/home/dex/Open CV project/speedplusv2/lightbox/test.json'
SUNLAMP_IMAGE_DIR = '/home/dex/Open CV project/speedplusv2/sunlamp/images'
SUNLAMP_LABELS_PATH = '/home/dex/Open CV project/speedplusv2/sunlamp/test.json'
OUTPUT_FILE = '/home/dex/Open CV project/project/analysis/results.json'

# --- Helper Functions ---
def quat_to_rot_vec(q):
    # Converts a quaternion to a rotation vector
    angle = 2 * np.arccos(q[0])
    s = np.sqrt(1 - q[0]*q[0])
    if s < 1e-6:
        return np.array([1, 0, 0]) * angle
    else:
        return np.array([q[1], q[2], q[3]]) / s * angle
def rot_mat_to_quat(rot_mat):
    # Converts a rotation matrix to a quaternion
    q = np.empty((4, ), dtype=np.float64)
    t = np.trace(rot_mat)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q[0] = 0.25 / s
        q[1] = (rot_mat[2, 1] - rot_mat[1, 2]) * s
        q[2] = (rot_mat[0, 2] - rot_mat[2, 0]) * s
        q[3] = (rot_mat[1, 0] - rot_mat[0, 1]) * s
    else:
        if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
            q[0] = (rot_mat[2, 1] - rot_mat[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            q[3] = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        elif rot_mat[1, 1] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
            q[0] = (rot_mat[0, 2] - rot_mat[2, 0]) / s
            q[1] = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
            q[0] = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            q[1] = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            q[2] = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            q[3] = 0.25 * s
    return q

# --- Main Analysis ---
def main():
    # Load the trained model
    model = KeypointRegressionNet(num_keypoints=11)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load 3D keypoints and camera intrinsics
    tango_points_mat = scipy.io.loadmat(TANGO_POINTS_PATH)['tango3Dpoints']
    points_3d = tango_points_mat.T.astype(np.float32)

    with open(CAMERA_PARAMS_PATH) as f:
        camera_params = json.load(f)
    camera_matrix = np.array(camera_params['cameraMatrix'], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = {'lightbox': [], 'sunlamp': []}

    for dataset in ['lightbox', 'sunlamp']:
        print(f'Processing {dataset} dataset...')
        image_dir = LIGHTBOX_IMAGE_DIR if dataset == 'lightbox' else SUNLAMP_IMAGE_DIR
        labels_path = LIGHTBOX_LABELS_PATH if dataset == 'lightbox' else SUNLAMP_LABELS_PATH

        with open(labels_path) as f:
            labels = json.load(f)

        for i, label in enumerate(labels):
            image_path = os.path.join(image_dir, label['filename'])
            if not os.path.exists(image_path):
                continue

            # Load and preprocess the image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(image_rgb).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                pred_x, pred_y = model(input_tensor)
            predicted_2d_keypoints = np.vstack((pred_x.numpy().flatten(), pred_y.numpy().flatten())).T

            # Solve for 6-DOF pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                points_3d, predicted_2d_keypoints, camera_matrix, dist_coeffs
            )

            if success:
                # Ground truth pose
                true_q = np.array(label['q_vbs2tango_true'])
                true_pos_unrotated = np.array(label['r_Vo2To_vbs_true']) / 1000.0
                true_rot_mat, _ = cv2.Rodrigues(quat_to_rot_vec(true_q))
                true_pos = np.dot(true_rot_mat, true_pos_unrotated)

                # Predicted pose
                pred_pos = translation_vector.flatten()
                pred_rot_mat, _ = cv2.Rodrigues(rotation_vector)

                # Convert rotation matrix to quaternion
                pred_q = rot_mat_to_quat(pred_rot_mat)

                # Normalize quaternions
                pred_q /= np.linalg.norm(pred_q)
                true_q /= np.linalg.norm(true_q)

                # Calculate errors
                translation_error = np.linalg.norm(pred_pos - true_pos)
                
                # Attitude error (angular distance between quaternions)
                dot_product = np.dot(pred_q, true_q)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                attitude_error = 2 * np.arccos(np.abs(dot_product))

                results[dataset].append({
                    'filename': label['filename'],
                    'translation_error': translation_error,
                    'attitude_error': attitude_error
                })

            if (i + 1) % 100 == 0:
                print(f'  Processed {i + 1}/{len(labels)} images')

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Analysis complete. Results saved to {OUTPUT_FILE}')

if __name__ == '__main__':
    main()
