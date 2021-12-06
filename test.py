import sys
import numpy as np

from open3d import *

sys.path.append("../")

'''
import json
from src.utils import to_json

data = {('category1', 'category2'): {frozenset(['cat1', 'cat2']): np.int32(1212)}}
print(json.dumps(to_json(data)))
sys.exit(0)
'''

import h5py

import torch
from src.fitting_utils import (
    to_one_hot,
)
import os
from src.segment_utils import SIOU_matched_segments
from src.utils import chamfer_distance_single_shape
from src.segment_utils import sample_from_collection_of_mesh
from src.primitives import SaveParameters
from src.dataset_segments import Dataset
from src.residual_utils import Evaluation
import sys

import open3d as o3d

start = int(sys.argv[1])
end = int(sys.argv[2])
custom_data = int(sys.argv[3])
prefix = ""

dataset = Dataset(
    1,
    24000,
    4000,
    4000,
    normals=True,
    primitives=True,
    if_train_data=False,
    prefix=prefix
)


def continuous_labels(labels_):
    new_labels = np.zeros_like(labels_)
    for index, value in enumerate(np.sort(np.unique(labels_))):
        new_labels[labels_ == value] = index
    return new_labels


# root_path = "data/shapes/test_data.h5"
root_path = prefix + "data/shapes/test_data.h5"


if custom_data:
    start, end = 0, 1
    pad_to_this_number_of_points = 10000

    fn = 'custom-ply/35.ply'
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(fn)
    print(type(pcd), pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    print("Downsample the point cloud with a voxel of 0.5")
    downpcd = pcd.voxel_down_sample(voxel_size=0.5)
    o3d.visualization.draw_geometries([downpcd])

    if pad_to_this_number_of_points <= len(np.array(downpcd.points)):
        print("Downsample the point cloud with a voxel of 0.8")
        downpcd = pcd.voxel_down_sample(voxel_size=0.8)
        o3d.visualization.draw_geometries([downpcd])
    if pad_to_this_number_of_points <= len(np.array(downpcd.points)):
        print("Downsample the point cloud with a voxel of 1.1")
        downpcd = pcd.voxel_down_sample(voxel_size=1.1)
        o3d.visualization.draw_geometries([downpcd])

    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])

    print("Print a normal vector of the 0th point")
    print(downpcd.normals[0])
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(downpcd.normals)[:10, :]) 
    print("")

    test_points     = np.array(downpcd.points)
    test_normals    = np.array(downpcd.normals)
    from src.fitting_utils import remove_outliers
    remove_outliers(test_points)

    #test_points     = np.true_divide(test_points, 10)

    print(f'{20*"-"} Printing "base" arrays...')
    print(f'test_points:     {type(test_points)} - {len(test_points)} - {test_points.dtype} - {test_points.shape} - {test_points[:3]}')
    print(f'test_normals:    {type(test_normals)} - {len(test_normals)} - {test_normals.dtype} - {test_normals.shape} - {test_normals[:3]}')

    test_points_min, test_points_max = test_points.min(), test_points.max()
    print(f'test_points_min: {test_points_min} - test_points_max: {test_points_max}')

    #test_points_padded = np.pad(test_points, (0, len(test_points)), constant_values=([0,0,0]))
    # np.pad(a, ((0, 6), (0, 0)), mode='constant')
    print(f'{20*"-"} Printing "padded" arrays...')
    test_points_padded = np.pad(test_points.astype('float32'), ((0, pad_to_this_number_of_points-len(test_points)), (0,0)))
    print(f'test_points_padded: {type(test_points_padded)} - {len(test_points_padded)} - {test_points_padded.dtype} - {test_points_padded.shape} - {test_points_padded[:3]}')
    test_normals_padded = np.pad(test_normals.astype('float32'), ((0, pad_to_this_number_of_points-len(test_normals)), (0,0)))
    print(f'test_normals_padded: {type(test_normals_padded)} - {len(test_normals_padded)} - {test_normals_padded.dtype} - {test_normals_padded.shape} - {test_normals_padded[:3]}')

    #test_labels     = np.array([0]*len(test_points_padded), dtype='int16')
    test_labels     = np.random.randint(0,11,len(test_points_padded), dtype='int16')
    test_primitives = np.array([0]*len(test_points_padded), dtype='int16')

    print(f'{20*"-"} Printing "base" labels/primitives...')
    print(f'test_labels:     {type(test_labels)} - {len(test_labels)} - {test_labels.dtype} - {test_labels.shape} - {test_labels[:3]}')
    print(f'test_primitives: {type(test_primitives)} - {len(test_primitives)} - {test_primitives.dtype} - {test_primitives.shape} - {test_primitives[:3]}')

    test_points     = test_points_padded[np.newaxis]
    test_normals    = test_normals_padded[np.newaxis]
    test_labels     = test_labels[np.newaxis]
    test_primitives = test_primitives[np.newaxis]

    print(f'{20*"-"} Printing "batch" arrays...')
    print(f'test_points:     {type(test_points)} - {len(test_points)} - {test_points.dtype} - {test_points.shape} - {test_points[:3]}')
    print(f'test_normals:    {type(test_normals)} - {len(test_normals)} - {test_normals.dtype} - {test_normals.shape} - {test_normals[:3]}')
    print(f'test_labels:     {type(test_labels)} - {len(test_labels)} - {test_labels.dtype} - {test_labels.shape} - {test_labels[:3]}')
    print(f'test_primitives: {type(test_primitives)} - {len(test_primitives)} - {test_primitives.dtype} - {test_primitives.shape} - {test_primitives[:3]}')
else:
    with h5py.File(root_path, "r") as hf:
        # N x 3
        test_points = np.array(hf.get("points"))
    
        # N x 1
        test_labels = np.array(hf.get("labels"))
    
        # N x 3
        test_normals = np.array(hf.get("normals"))
    
        # N x 1
        test_primitives = np.array(hf.get("prim"))

        print(f'test_points:     {type(test_points)} - {len(test_points)} - {test_points.dtype} - {test_points.shape} - {test_points[:3]}')
        print(f'test_normals:    {type(test_normals)} - {len(test_normals)} - {test_normals.dtype} - {test_normals.shape} - {test_normals[:3]}')
        print(f'test_labels:     {type(test_labels)} - {len(test_labels)} - {test_labels.dtype} - {test_labels.shape} - {test_labels[:3]}')
        print(f'test_primitives: {type(test_primitives)} - {len(test_primitives)} - {test_primitives.dtype} - {test_primitives.shape} - {test_primitives[:3]}')

        test_points_min, test_points_max = test_points.min(), test_points.max()
        print(f'test_points_min: {test_points_min} - test_points_max: {test_points_max}')

        '''
	test_points:     <class 'numpy.ndarray'> - 4163 - float32 - (4163, 10000, 3) - [[[-4.6151257e+00  4.7416029e+00  0.0000000e+00]
	test_normals:    <class 'numpy.ndarray'> - 4163 - float32 - (4163, 10000, 3) - [[[ 0.00000000e+00  0.00000000e+00 -1.00000000e+00]
	test_labels:     <class 'numpy.ndarray'> - 4163 - int16 - (4163, 10000) - [[ 0  0  0 ... 22 22 22]
	 [ 2  2  2 ... 17  9 18]
	 [ 0  0  0 ... 16 16 16]]
	test_primitives: <class 'numpy.ndarray'> - 4163 - int16 - (4163, 10000) - [[1 1 1 ... 1 1 1]
	 [1 1 1 ... 1 4 4]
	 [7 7 7 ... 7 7 7]]
        '''


method_name = "parsenet_with_normals.pth"

root_path = prefix + "logs/results/{}/results/predictions.h5".format(method_name)
print(root_path)
with h5py.File(root_path, "r") as hf:
    print(list(hf.keys()))
    test_cluster_ids = np.array(hf.get("seg_id")).astype(np.int32)
    test_pred_primitives = np.array(hf.get("pred_primitives"))

prim_ids = {}
prim_ids[11] = "torus"
prim_ids[1] = "plane"
prim_ids[2] = "open-bspline"
prim_ids[3] = "cone"
prim_ids[4] = "cylinder"
prim_ids[5] = "sphere"
prim_ids[6] = "other"
prim_ids[7] = "revolution"
prim_ids[8] = "extrusion"
prim_ids[9] = "closed-bspline"

saveparameters = SaveParameters()

root_path = "/mnt/nfs/work1/kalo/gopalsharma/Projects/surfacefitting/logs_curve_fitting/outputs/{}/"

all_pred_meshes = []
all_input_points = []
all_input_labels = []
all_input_normals = []
all_cluster_ids = []
evaluation = Evaluation()
all_segments = []

os.makedirs("../logs_curve_fitting/results/{}/results/".format(method_name), exist_ok=True)

test_res = []
test_s_iou = []
test_p_iou = []
s_k_1s = []
s_k_2s = []
p_k_1s = []
p_k_2s = []
s_ks = []
p_ks = []
test_cds = []

for i in range(start, end):
    bw = 0.01
    points = test_points[i].astype(np.float32)
    normals = test_normals[i].astype(np.float32)

    labels = test_labels[i].astype(np.int32)
    labels = continuous_labels(labels)

    cluster_ids = test_cluster_ids[i].astype(np.int32)
    cluster_ids = continuous_labels(cluster_ids)
    weights = to_one_hot(cluster_ids, np.unique(cluster_ids).shape[0])

    points, normals = dataset.normalize_points(points, normals)
    torch.cuda.empty_cache()
    with torch.no_grad():
        # if_visualize=True, will give you all segments
        # if_sample=True will return segments as trimmed meshes
        # if_optimize=True will optimize the spline surface patches
        _, parameters, newer_pred_mesh = evaluation.residual_eval_mode(
            torch.from_numpy(points).cuda(),
            torch.from_numpy(normals).cuda(),
            labels,
            cluster_ids,
            test_primitives[i],
            test_pred_primitives[i],
            weights.T,
            bw,
            sample_points=True,
            if_optimize=False,
            if_visualize=True,
            epsilon=0.1)

    torch.cuda.empty_cache()
    s_iou, p_iou, _, _ = SIOU_matched_segments(
        labels,
        cluster_ids,
        test_pred_primitives[i],
        test_primitives[i],
        weights,
    )

    test_s_iou.append(s_iou)
    test_p_iou.append(p_iou)

    try:
        Points = sample_from_collection_of_mesh(newer_pred_mesh)
    except Exception as e:
        print("error in sample_from_collection_of_mesh method", e)
        continue

    '''
    # try to output... something!
    input_pcd = pred_pcd = o3d.geometry.PointCloud()
    input_pcd.points = o3d.utility.Vector3dVector(points)
    pred_pcd.points  = o3d.utility.Vector3dVector(Points)

    #o3d.io.write_triangle_mesh(f'pred-{i}.ply', pred_pcd)
    #o3d.io.write_triangle_mesh(f'input-{i}.ply', pcd)
    o3d.io.write_point_cloud(f'pred-{i}.ply', pred_pcd)
    o3d.io.write_point_cloud(f'input-{i}.ply', input_pcd)
    '''
    from src.utils import visualize_point_cloud
    visualize_point_cloud(Points, file=f'pred-{i}.ply' , viz=True)
    visualize_point_cloud(points, file=f'input-{i}.ply', viz=True)
    for idx, mesh in enumerate(newer_pred_mesh):
        o3d.io.write_triangle_mesh(f'pred-{i}-{idx}.ply', mesh)



    cd1 = chamfer_distance_single_shape(torch.from_numpy(Points).cuda(), torch.from_numpy(points).cuda(), sqrt=True,
                                        one_side=True, reduce=False)
    cd2 = chamfer_distance_single_shape(torch.from_numpy(points).cuda(), torch.from_numpy(Points).cuda(), sqrt=True,
                                        one_side=True, reduce=False)

    s_k_1s.append(torch.mean((cd1 < 0.01).float()).item())
    s_k_2s.append(torch.mean((cd1 < 0.02).float()).item())
    s_ks.append(torch.mean(cd1).item())
    p_k_1s.append(torch.mean((cd2 < 0.01).float()).item())
    p_k_2s.append(torch.mean((cd2 < 0.02).float()).item())
    p_ks.append(torch.mean(cd2).item())
    test_cds.append((s_ks[-1] + p_ks[-1]) / 2.0)

    results = {"sk_1": s_k_1s[-1],
               "sk_2": s_k_2s[-1],
               "sk": s_ks[-1],
               "pk_1": p_k_1s[-1],
               "pk_2": p_k_2s[-1],
               "pk": p_ks[-1],
               "cd": test_cds[-1],
               "p_iou": p_iou,
               "s_iou": s_iou}

    print(f'i: {i} - s_iou: {s_iou} - p_iou: {p_iou} - test_cds[-1]: {test_cds[-1]} - results: {results}')
    print(f'parameters: {parameters}')
    with open(f'pred-params-{i}.json', 'w') as fp:
        json.dump(parameters, fp)

print("Test CD: {}, Test p cover: {}, Test s cover: {}".format(np.mean(test_cds), np.mean(s_ks), np.mean(p_ks)))
print("iou seg: {}, iou prim type: {}".format(np.mean(test_s_iou), np.mean(test_p_iou)))
