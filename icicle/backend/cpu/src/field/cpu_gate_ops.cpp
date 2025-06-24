#include <cstdint>
#include "icicle/backend/gate_ops_backend.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {

template <typename E>
eIcicleError lookups_constraint_op(
    const Device& device,
    const LookupData<E>& lookup_data, 
    const LookupConfig& config, 
    E* result)
{
    for (uint32_t idx = 0; idx < lookup_data.num_elements; idx++) {
        int r_next = ((idx + lookup_data.rot_scale) % lookup_data.i_size + lookup_data.i_size) % lookup_data.i_size;

        E lhs = lookup_data.table_values[idx] * lookup_data.inputs_prods[idx] * 
                (lookup_data.phi_coset[r_next] - lookup_data.phi_coset[idx]);
        E rhs = lookup_data.inputs_prods[idx] * 
                (lookup_data.table_values[idx] * lookup_data.inputs_inv_sums[idx] - lookup_data.m_coset[idx]);

        E res = lookup_data.previous_value[idx] * lookup_data.y[0] + 
                lookup_data.l0[idx] * lookup_data.phi_coset[idx];

        res = res * lookup_data.y[0] + lookup_data.l_last[idx] * lookup_data.phi_coset[idx];
        res = res * lookup_data.y[0] + (lhs - rhs) * lookup_data.l_active_row[idx];

        result[idx] = res;
    }

    return eIcicleError::SUCCESS;
}

template <typename E>
eIcicleError gate_evaluation_op(
    const Device& device,
    const GateData<E>& gate_data, 
    const CalculationData<E>& calc_data,
    const HornerData& horner_data,
    const GateOpsConfig& config, 
    E* result)
{
    // Allocate memory for intermediates
    E* intermediates = new E[calc_data.num_elements * calc_data.num_intermediates];
    uint32_t* rotation_indices = new uint32_t[calc_data.num_elements * calc_data.num_rotations];

    // Precompute rotation indices
    for (uint32_t idx = 0; idx < calc_data.num_elements; idx++) {
        for (uint32_t rot_idx = 0; rot_idx < calc_data.num_rotations; rot_idx++) {
            int rot = calc_data.rotations[rot_idx];
            rotation_indices[idx * calc_data.num_rotations + rot_idx] = 
                ((idx + (rot * calc_data.rot_scale)) % calc_data.i_size + calc_data.i_size) % calc_data.i_size;
        }
    }

    auto get_value = [&](int source_type, size_t column_index, size_t rotation_index) -> E {
        size_t row = rotation_indices[column_index * calc_data.num_rotations + rotation_index];
        switch (source_type) {
            case 0: return calc_data.constants[column_index];
            case 1: return intermediates[column_index];
            case 2: return gate_data.fixed[column_index * gate_data.num_fixed_rows + row];
            case 3: return gate_data.advice[column_index * gate_data.num_advice_rows + row];
            case 4: return gate_data.instance[column_index * gate_data.num_instance_rows + row];
            case 5: return gate_data.challenges[column_index];
            case 6: return gate_data.beta[0];
            case 7: return gate_data.gamma[0];
            case 8: return gate_data.theta[0];
            case 9: return gate_data.y[0];
            case 10: return calc_data.is_prev_zero ? E::zero() : calc_data.previous_value[column_index];
            default: return E::zero();
        }
    };

    auto get_result = [&](int calc_type, size_t calc_idx) -> E {
        E first = get_value(calc_data.value_types[calc_idx * 2], 
                          calc_data.value_indices[calc_idx * 4], 
                          calc_data.value_indices[calc_idx * 4 + 1]);
        E second = get_value(calc_data.value_types[calc_idx * 2 + 1], 
                           calc_data.value_indices[calc_idx * 4 + 2], 
                           calc_data.value_indices[calc_idx * 4 + 3]);

        switch(calc_type) {
            case 0: return first + second; // Add
            case 1: return first - second; // Sub
            case 2: return first * second; // Mul
            case 3: return E::sqr(first);  // Square
            case 4: return first + first;  // Double
            case 5: return E::neg(first);  // Negate
            case 6: {                      // Horner
                uint32_t offset = horner_data.offsets[calc_idx];
                uint32_t size   = horner_data.sizes[calc_idx];

                for (uint32_t i = offset; i < offset + size; i++) {
                    E part_value = get_value(
                        horner_data.value_types[i],
                        horner_data.value_indices[i * 2],
                        horner_data.value_indices[i * 2 + 1]
                    );

                    first = first * second + part_value;
                }

                return first; 
            }
            case 7: return first;          // Store
            default: return E::zero();
        }
    };

    for (uint32_t idx = 0; idx < calc_data.num_elements; idx++) {
        for (uint32_t calc_idx = 0; calc_idx < calc_data.num_calculations; calc_idx++) {
            const uint32_t calc_type = calc_data.calc_types[calc_idx];
            E res = get_result(calc_type, calc_idx);
            int target_idx = calc_data.targets[calc_idx];
            intermediates[idx * calc_data.num_intermediates + target_idx] = res;
        }

        result[idx] = intermediates[idx * calc_data.num_intermediates + 
                                  calc_data.targets[calc_data.num_calculations - 1]];
    }

    delete[] intermediates;
    delete[] rotation_indices;

    return eIcicleError::SUCCESS;
}

// Register the CPU backend implementations
REGISTER_GATE_EVALUATION_BACKEND("CPU", gate_evaluation_op<scalar_t>);
REGISTER_LOOKUP_CONSTRAINT_BACKEND("CPU", lookups_constraint_op<scalar_t>);

} // namespace icicle 