#pragma once

#include "decision tree params/msm_c_tree_params.h"
#include "decision tree params/msm_nof_cores_tree_params.h"
#include "icicle/decision_tree.h"

DecisionTree<NOF_CLASSES_C_TREE> c_tree = DecisionTree<NOF_CLASSES_C_TREE>(
    NOF_FEATURES_C_TREE,
    thresholds_c_tree,
    indices_c_tree,
    left_childs_c_tree,
    right_childs_c_tree, class_predictions_c_tree, classes_c_tree);

DecisionTree<NOF_CLASSES_CORES_TREE> cores_tree = DecisionTree<NOF_CLASSES_CORES_TREE>(
    NOF_FEATURES_CORES_TREE,
    thresholds_cores_tree,
    indices_cores_tree,
    left_childs_cores_tree,
    right_childs_cores_tree, class_predictions_cores_tree, classes_cores_tree);
