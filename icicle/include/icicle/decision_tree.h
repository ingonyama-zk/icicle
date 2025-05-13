#pragma once

class DecisionTree
{
public:
  // Constructor
  DecisionTree(
    int nof_features,
    const double* thresholds,
    const int* indices,
    const int* left_childs,
    const int* right_childs,
    const int* class_predictions)
      : m_nof_features(nof_features), m_thresholds(thresholds), m_indices(indices), m_left_childs(left_childs),
        m_right_childs(right_childs), m_class_predictions(class_predictions)
  {
  }

  // Destructor
  ~DecisionTree() {}

  // Public predict method
  int predict(double* features)
  {
    return predict(features, 0); // Start from root node (index 0)
  }

private:
  const int m_nof_features;
  const double* const m_thresholds;
  const int* const m_indices;
  const int* const m_left_childs;
  const int* const m_right_childs;
  const int* const m_class_predictions;

  int predict(double* features, int node)
  {
    if (m_thresholds[node] != -2) {
      if (features[m_indices[node]] <= m_thresholds[node]) {
        return predict(features, m_left_childs[node]);
      } else {
        return predict(features, m_right_childs[node]);
      }
    }
    return m_class_predictions[node];
  }
};