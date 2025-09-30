
import torch
import cv2
import numpy as np
import scipy.io
import json
import os
from torchvision import transforms
from src.nets.park2019 import KeypointRegressionNet

# Load the trained model
model = KeypointRegressionNet(num_keypoints=11)
checkpoint = torch.load('/home/dex/Open CV project/speedplusbaseline/checkpoints/krn/synthetic_only/model_best.pth.tar')
model.load_state_dict(checkpoint)

model.eval()

# Load 3D keypoints and camera intrinsics
tango_points_mat = scipy.io.loadmat('/home/dex/Open CV project/speedplusbaseline/src/utils/tangoPoints.mat')['tango3Dpoints']
points_3d = tango_points_mat.T.astype(np.float32)

with open('/home/dex/Open CV project/speedplusv2/camera.json') as f:
    camera_params = json.load(f)
camera_matrix = np.array(camera_params['cameraMatrix'], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# Image preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loop through test images
image_dir = '/home/dex/Open CV project/speedplusv2/lightbox/images'
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])[:50]

for image_path in image_files:
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

    # Visualize the pose
    if success:
        axis_3d = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
        axis_2d, _ = cv2.projectPoints(axis_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Draw the lines
        origin = tuple(axis_2d[0].ravel().astype(int))
        image = cv2.line(image, origin, tuple(axis_2d[1].ravel().astype(int)), (255,0,0), 3) # X-axis in Blue
        image = cv2.line(image, origin, tuple(axis_2d[2].ravel().astype(int)), (0,255,0), 3) # Y-axis in Green
        image = cv2.line(image, origin, tuple(axis_2d[3].ravel().astype(int)), (0,0,255), 3) # Z-axis in Red

    # Save the result
    output_path = os.path.join('output_images', os.path.basename(image_path))
    cv2.imwrite(output_path, image)
