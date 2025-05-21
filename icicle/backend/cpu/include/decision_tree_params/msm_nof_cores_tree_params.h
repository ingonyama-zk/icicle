#pragma once

#define NOF_NODES_CORES_TREE         7
#define NOF_FEATURES_CORES_TREE      2
#define FIXED_SCALAR_SIZE_CORES_TREE 254

const int left_childs_cores_tree[NOF_NODES_CORES_TREE] = {1, 2, -1, -1, 5, -1, -1};
const int right_childs_cores_tree[NOF_NODES_CORES_TREE] = {4, 3, -1, -1, 6, -1, -1};
const double thresholds_cores_tree[NOF_NODES_CORES_TREE] = {11.5, 8.5, -2.0, -2.0, 6.0, -2.0, -2.0};
const int indices_cores_tree[NOF_NODES_CORES_TREE] = {0, 0, -2, -2, 1, -2, -2};
const int class_predictions_cores_tree[NOF_NODES_CORES_TREE] = {0, 0, 32, 64, 0, 128, 128};