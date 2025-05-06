#pragma once

#include "decision tree params/msm_c_tree_amd_params.h"
#include "decision tree params/msm_c_tree_intel_params.h"
#include "decision tree params/msm_c_tree_apple_params.h"
#include "decision tree params/msm_nof_cores_tree_params.h"
#include "icicle/decision_tree.h"

#ifdef G2_ENABLED
  #include "decision tree params/msm_c_tree_apple_params_g2.h"

  DecisionTree<NOF_CLASSES_C_TREE_APPLE_G2> msm_c_tree_apple_g2 = DecisionTree<NOF_CLASSES_C_TREE_APPLE_G2>(
    NOF_FEATURES_C_TREE_APPLE_G2,
    thresholds_c_tree_apple_g2,
    indices_c_tree_apple_g2,
    left_childs_c_tree_apple_g2,
    right_childs_c_tree_apple_g2,
    class_predictions_c_tree_apple_g2,
    classes_c_tree_apple_g2);
#endif

DecisionTree<NOF_CLASSES_C_TREE_INTEL> msm_c_tree_intel = DecisionTree<NOF_CLASSES_C_TREE_INTEL>(
  NOF_FEATURES_C_TREE_INTEL,
  thresholds_c_tree_intel,
  indices_c_tree_intel,
  left_childs_c_tree_intel,
  right_childs_c_tree_intel,
  class_predictions_c_tree_intel,
  classes_c_tree_intel);

DecisionTree<NOF_CLASSES_C_TREE_AMD> msm_c_tree_amd = DecisionTree<NOF_CLASSES_C_TREE_AMD>(
  NOF_FEATURES_C_TREE_AMD,
  thresholds_c_tree_amd,
  indices_c_tree_amd,
  left_childs_c_tree_amd,
  right_childs_c_tree_amd,
  class_predictions_c_tree_amd,
  classes_c_tree_amd);

DecisionTree<NOF_CLASSES_C_TREE_APPLE> msm_c_tree_apple = DecisionTree<NOF_CLASSES_C_TREE_APPLE>(
  NOF_FEATURES_C_TREE_APPLE,
  thresholds_c_tree_apple,
  indices_c_tree_apple,
  left_childs_c_tree_apple,
  right_childs_c_tree_apple,
  class_predictions_c_tree_apple,
  classes_c_tree_apple);

DecisionTree<NOF_CLASSES_CORES_TREE> msm_nof_cores_tree = DecisionTree<NOF_CLASSES_CORES_TREE>(
  NOF_FEATURES_CORES_TREE,
  thresholds_cores_tree,
  indices_cores_tree,
  left_childs_cores_tree,
  right_childs_cores_tree,
  class_predictions_cores_tree,
  classes_cores_tree);
