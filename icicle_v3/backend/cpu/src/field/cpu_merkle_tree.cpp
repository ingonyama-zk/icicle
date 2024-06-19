#include <iostream>
#include <functional>
#include "icicle/hash.h"
using namespace icicle;

// Define the MerkleTree class
class MerkleTreeCPU : public MerkleTree{
public:
    int *layer_nof_limbs;
    int *layer_nof_elements;
    MerkleTreeCPU(unsigned int nof_layers, const Hash **layer_hashes,
              unsigned int output_store_min_layer,
              unsigned int output_store_max_layer,
              TreeBuilderConfig tree_config)
    : MerkleTree(nof_layers, layer_hashes,
              output_store_min_layer,
              output_store_max_layer,
            tree_config) {                
        this->layer_nof_limbs = new int[nof_layers];
        this->layer_nof_elements = new int[nof_layers];
        int nof_elements_in_layer = (*layer_hashes[nof_layers - 1]).output_nof_elements;
        int nof_limbs_in_element = (*layer_hashes[nof_layers - 1]).element_nof_limbs;
        int nof_limbs_in_layer = nof_elements_in_layer * nof_limbs_in_element;
        this->layer_nof_limbs[nof_layers - 1] = nof_limbs_in_layer;
        int prev_nof_limbs_in_layer = nof_limbs_in_layer;
        
        // nof limbs per layer calc
        for (int i = nof_layers - 2; i >= 0; i--) {
            int layer_factor = (int)((*layer_hashes[i+1]).input_nof_elements / (*layer_hashes[i+1]).output_nof_elements); // TODO: what to do between keccak poseidon layer?
            nof_limbs_in_layer = layer_factor * prev_nof_limbs_in_layer;
            this->layer_nof_limbs[i] = nof_limbs_in_layer;
            prev_nof_limbs_in_layer = nof_limbs_in_layer;
            // std::cout << "Nof limb in layer " << i << " is " << nof_limbs_in_layer << std::endl;
        }
        // nof elements per layer calc
        for (int i=0; i<nof_layers; i++)
        {
            this->layer_nof_elements[i] = this->layer_nof_limbs[i] / (*layer_hashes[i]).element_nof_limbs;
        }
        // output memory allocation
        this->outputs = new limb_t*[output_store_max_layer - output_store_min_layer + 1];
        int output_store_layer_idx = 0;
        for (int i = output_store_min_layer; i <= output_store_max_layer; i++) {
            this->outputs[output_store_layer_idx] = new limb_t[this->layer_nof_limbs[i]];
            output_store_layer_idx++;
        }

    }

    eIcicleError build(const limb_t *leaves) const override {
        // Implement the build function
        // std::cout << "Building Merkle Tree..." << std::endl;
        limb_t *current_layer_outputs;
        limb_t *prev_layer_outputs;
        // Naive implementation, go layer by layer
        for (int layer_idx = 0; layer_idx < this->nof_layers; layer_idx++)
        {
            // std::cout << "Building layer " << layer_idx << std::endl;
            const Hash *layer_hash = this->layer_hashes[layer_idx];
            int nof_input_limbs_per_hash = layer_hash->input_nof_elements * layer_hash->element_nof_limbs;
            int nof_output_limbs_per_hash = layer_hash->output_nof_elements * layer_hash->element_nof_limbs;
            int nof_hashes = this->layer_nof_limbs[layer_idx] / nof_output_limbs_per_hash;
            
            // allocate layer memory if needed
            if (layer_idx <= this->output_store_max_layer && layer_idx >= this->output_store_min_layer) // memory is already alocated at this->outputs[layer_idx]
            {
                current_layer_outputs = this->outputs[layer_idx - this->output_store_min_layer];
            }
            else
            {
                current_layer_outputs = new limb_t[this->layer_nof_limbs[layer_idx]];
            }
            
            // create the layer hashes
            if (layer_idx == 0)
            {
                layer_hash->hash_many(leaves, current_layer_outputs, nof_hashes);
            }
            else
            {
                layer_hash->hash_many(prev_layer_outputs, current_layer_outputs, nof_hashes);
            }

            // free previous layer memory if not needed
            if (layer_idx - 1 > 0 && !(layer_idx - 1 <= this->output_store_max_layer && layer_idx - 1 >= this->output_store_min_layer))
            {
                delete[] prev_layer_outputs;
            }
            prev_layer_outputs = current_layer_outputs;
            
        }

        if (this->output_store_max_layer != this->nof_layers - 1)
        {
            delete[] prev_layer_outputs;
        }
        return eIcicleError::SUCCESS;
    }

    int get_path_nof_limbs () const override
    {
        int nof_limbs = 0;
        for (int i=0; i<this->nof_layers; i++)
        {
            nof_limbs += (*this->layer_hashes[i]).input_nof_elements * (*this->layer_hashes[i]).element_nof_limbs;

        }
        // root
        nof_limbs += (*this->layer_hashes[this->nof_layers-1]).output_nof_elements * (*this->layer_hashes[this->nof_layers-1]).element_nof_limbs;
        return nof_limbs;
    }

    eIcicleError print_path(const limb_t* path) const override
    {
        int path_idx = 0;
        for (int i=0; i<this->nof_layers; i++)
        {
            int nof_limbs = (*this->layer_hashes[i]).input_nof_elements * (*this->layer_hashes[i]).element_nof_limbs;
            if (i == 0)
            {
                std::cout << "Leaves " << " : ";
            }
            else
            {
                std::cout << "Layer " << i - 1 << " : ";
            }
            
            for (int j=0; j<nof_limbs; j++)
            {
                std::cout << path[path_idx] << " ";
                path_idx++;
            }
            std::cout << std::endl;
        }

        // root
        int nof_limbs = (*this->layer_hashes[this->nof_layers - 1]).output_nof_elements * (*this->layer_hashes[this->nof_layers - 1]).element_nof_limbs;    
        std::cout << "Layer " << this->nof_layers - 1 << " : ";        
        for (int j=0; j<nof_limbs; j++)
        {
            std::cout << path[path_idx] << " ";
            path_idx++;
        }
        std::cout << std::endl;
        return eIcicleError::SUCCESS;
    }

    eIcicleError get_path(const limb_t *leaves, unsigned int element_index, limb_t *path /*OUT*/) const override {
        int path_index = 0;
        int prev_path_index = 0; // used to calculate hash instead of full subtree
        
        // note: each layer adds only the layer childeren to the path.
        // layer 0 children path
        int hash_index = element_index / (*this->layer_hashes[0]).input_nof_elements; 
        std::cout << "hash_idx = " << hash_index << std::endl;
        int childs_nof_limbs = (*this->layer_hashes[0]).element_nof_limbs * (*this->layer_hashes[0]).input_nof_elements;
        int childs_limbs_start_index = hash_index * (*this->layer_hashes[0]).element_nof_limbs * (*this->layer_hashes[0]).input_nof_elements;
        int childs_limbs_stop_index = childs_limbs_start_index + childs_nof_limbs;
        std::copy(leaves + childs_limbs_start_index, leaves + childs_limbs_stop_index, path);
        path_index += childs_nof_limbs;
        int childs_current_element_index = hash_index;
        
        for (int i=1; i<nof_layers; i++) // layer i children path
        {   
            // prepare new indices
            int childs_current_element_index_in_hash = childs_current_element_index % (*this->layer_hashes[i]).input_nof_elements;
            int childs_current_limb_index_in_hash = childs_current_element_index_in_hash * (*this->layer_hashes[i]).element_nof_limbs;
            hash_index = childs_current_element_index / (*this->layer_hashes[i]).input_nof_elements;
            int childs_first_element_index = hash_index * (*this->layer_hashes[i]).input_nof_elements;
            childs_nof_limbs = (*this->layer_hashes[i]).element_nof_limbs * (*this->layer_hashes[i]).input_nof_elements;
            
            // if we have childeren hashes saved, just copy into path.
            if (this->is_layer_saved(i-1))
            {
                childs_limbs_start_index = hash_index * (*this->layer_hashes[i]).element_nof_limbs * (*this->layer_hashes[i]).input_nof_elements;
                childs_limbs_stop_index = childs_limbs_start_index + childs_nof_limbs;
                std::copy(this->outputs[i-1] + childs_limbs_start_index, this->outputs[i-1] + childs_limbs_stop_index, path + path_index);
            }

            // if not, build children hashes and copy to path.
            else
            {
                limb_t* childs_limbs = new limb_t[childs_nof_limbs];
                (*this->layer_hashes[i-1]).hash_many(path + prev_path_index, childs_limbs + childs_current_limb_index_in_hash, 1);
                for (int j=0; j<(*this->layer_hashes[i]).input_nof_elements; j++)
                {
                    if (j == childs_current_element_index_in_hash)
                        continue;
                    this->get_node(leaves, i - 1, childs_first_element_index + j, childs_limbs + j*(*this->layer_hashes[i]).element_nof_limbs);
                }
                std::copy(childs_limbs, childs_limbs+childs_nof_limbs, path + path_index);
                delete[] childs_limbs;
            }

            // prepare for next loop
            childs_current_element_index = hash_index;
            prev_path_index = path_index;
            path_index += childs_nof_limbs;
            
        }
        // root
        int nof_limbs = (*this->layer_hashes[nof_layers-1]).element_nof_limbs * (*this->layer_hashes[nof_layers-1]).output_nof_elements; 
        if (is_layer_saved(this->nof_layers-1))
        {
            std::copy(this->outputs[nof_layers-1], this->outputs[nof_layers-1] + nof_limbs, path + path_index);
        }
        // if not, childrens are ready, just hash.
        else
        {
            (*this->layer_hashes[nof_layers-1]).hash_many(path + prev_path_index, path + path_index, 1);
        }

        return eIcicleError::SUCCESS;
    }

    

    eIcicleError get_node(const limb_t* leaves, int layer_index, int element_index, limb_t* node /*OUT*/) const override{
        if (this->is_layer_saved(layer_index))
        {
            int limb_index = this->element_index_to_limb_index(layer_index, element_index);
            int nof_limbs = this->element_nof_limbs(layer_index);
            std::copy(this->outputs[layer_index] + limb_index, this->outputs[layer_index] + limb_index + nof_limbs, node);
            return eIcicleError::SUCCESS;
        }
        // find nof leaves required
        int nof_leaves_elements_required = 1;
        for (int i = layer_index; i>=0; i--)
        {
            nof_leaves_elements_required *= (*this->layer_hashes[i]).input_nof_elements;
        }
        int leaves_first_element_index = element_index * nof_leaves_elements_required;
        int leaves_first_limb_index = this->element_index_to_limb_index(0, leaves_first_element_index);
        // std::cout << "leave_first_limb_index = " << leaves_first_element_index << std::endl;
        // std::cout << "nof_leaves_elements_required = " << nof_leaves_elements_required << std::endl;
        const limb_t* subtree_leaves = leaves + leaves_first_limb_index;
        TreeBuilderConfig tree_config;
        MerkleTreeCPU subtree(layer_index + 1, this->layer_hashes, layer_index, layer_index, tree_config);
        subtree.build(subtree_leaves);
        int node_nof_elements = (*this->layer_hashes[layer_index]).output_nof_elements * (*this->layer_hashes[layer_index]).element_nof_limbs;
        //std::cout <<" node nof elements = " << node_nof_elements << std::endl;
        std::copy(subtree.outputs[0], subtree.outputs[0] + node_nof_elements, node);
        return eIcicleError::SUCCESS;

    }

    eIcicleError verify(const limb_t *path, unsigned int element_idx, const limb_t *element, bool& verification_valid /*OUT*/) const override {
        std::cout << "Verify path " << std::endl;
        // TODO: implement
        return eIcicleError::SUCCESS;
    }

    eIcicleError print_tree() const override {
        // Implement the get_path function
        for (int layer_idx = this->output_store_min_layer; layer_idx <= this->output_store_max_layer; layer_idx++)
        {
            std::cout << "Layer " << layer_idx << ": ";
            for (int limb_idx = 0; limb_idx < this->layer_nof_limbs[layer_idx]; limb_idx++)
            {
                std::cout << this->outputs[layer_idx - this->output_store_min_layer][limb_idx] << " ";
            }
            std::cout << std::endl;
        }
        return eIcicleError();
    }

    


private:
    int element_index_to_limb_index(int layer_index, int element_index) const
    {
        return element_index * (*this->layer_hashes[layer_index]).element_nof_limbs;
    }

    int element_nof_limbs(int layer_index) const
    {
        return (*this->layer_hashes[layer_index]).element_nof_limbs;
    }

    bool is_layer_saved(int layer_index) const
    {
        return (layer_index >= this->output_store_min_layer && layer_index <= this->output_store_max_layer);
    }
};


eIcicleError merkle_tree_cpu(const Device& device, MerkleTree** merkle_tree, int nof_layers, const Hash **layer_hashes,
               unsigned int output_store_min_layer, unsigned int output_store_max_layer,
               TreeBuilderConfig tree_config)
{
    *merkle_tree = new MerkleTreeCPU(nof_layers, layer_hashes, output_store_min_layer, output_store_max_layer, tree_config);
    return eIcicleError::SUCCESS;
}

REGISTER_MERKLE_TREE_BACKEND("CPU", merkle_tree_cpu);