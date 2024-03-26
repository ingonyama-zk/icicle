set (CURVE_SOURCE grumpkin/curve.cu)
list (APPEND CURVE_SOURCE grumpkin/msm.cu)

set (CURVE_SOURCE ${CURVE_SOURCE} PARENT_SCOPE)