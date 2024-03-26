set(FIELD_SOURCE grumpkin/field.cu)
list(APPEND FIELD_SOURCE grumpkin/vec_ops.cu)
list(APPEND FIELD_SOURCE grumpkin/poseidon.cu)

set(FIELD_SOURCE ${FIELD_SOURCE} PARENT_SCOPE)