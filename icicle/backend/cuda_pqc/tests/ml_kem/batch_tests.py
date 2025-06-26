from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
import os

def generate_random_bytes(ml_kem_class, output_file, batch):
    """
    Generate random bytes using the provided ML-KEM class and save to file
    
    Args:
        ml_kem_class: One of ML_KEM_512, ML_KEM_768, ML_KEM_1024
        output_file: Path to save the random bytes
        batch: Number of key pairs to generate
    """

    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate total bytes needed based on batch size
    total_bytes = batch * 2 * 32
    
    # Generate random bytes using the ML-KEM class's random_bytes method
    random_data = ml_kem_class.random_bytes(total_bytes)
    
    # Save random bytes to file
    with open(f"{output_file}_{batch}_random_bytes.txt", 'wb') as f:
        f.write(random_data)
    
    # Open files for writing encryption and decryption keys
    with open(f"{output_file}_{batch}_ek.txt", 'wb') as f_ek, \
         open(f"{output_file}_{batch}_dk.txt", 'wb') as f_dk:
        
        for i in range(batch):
            # Get 32 bytes for d and next 32 bytes for z
            d = random_data[i * 64 : i * 64 + 32]
            z = random_data[i * 64 + 32 : (i * 64) + 64]
            
            # Generate key pair
            ek, dk = ml_kem_class._keygen_internal(d, z)
            
            # Write keys to respective files
            f_ek.write(ek)
            f_dk.write(dk)

def verify_key_generation(ml_kem_class, base_file, batch):
    """
    Verify that key generation produces the same results when using saved random bytes
    
    Args:
        ml_kem_class: One of ML_KEM_512, ML_KEM_768, ML_KEM_1024
        base_file: Base filename used when generating the original keys
        batch: Number of key pairs to verify
    """
    # Read the original random bytes
    with open(f"{base_file}_{batch}_random_bytes.txt", 'rb') as f:
        random_data = f.read()
    
    # Read the original encryption and decryption keys
    with open(f"{base_file}_{batch}_ek.txt", 'rb') as f:
        original_eks = f.read()
    with open(f"{base_file}_{batch}_dk.txt", 'rb') as f:
        original_dks = f.read()
        
    # Calculate key sizes based on ML-KEM variant
    k = ml_kem_class.k
    ek_size = (384 * k) + 32
    dk_size = (768 * k) + 96
    
    # Verify each key pair
    for i in range(batch):
        # Get the same d and z values used originally
        d = random_data[i * 64 : i * 64 + 32]
        z = random_data[i * 64 + 32 : (i * 64) + 64]
        
        # Generate key pair using the same random values
        ek, dk = ml_kem_class._keygen_internal(d, z)
        
        # Compare with original keys
        original_ek = original_eks[i * ek_size : (i + 1) * ek_size]
        original_dk = original_dks[i * dk_size : (i + 1) * dk_size]
        
        assert ek == original_ek, f"Encryption key mismatch at index {i}"
        assert dk == original_dk, f"Decryption key mismatch at index {i}"
    
    print(f"Successfully verified {batch} key pairs")

generate_random_bytes(ML_KEM_512, "ml_kem_512_data/ml_kem_512", 4 * 2048)
generate_random_bytes(ML_KEM_768, "ml_kem_768_data/ml_kem_768", 4 * 2048)
generate_random_bytes(ML_KEM_1024, "ml_kem_1024_data/ml_kem_1024", 4 * 2048)

verify_key_generation(ML_KEM_512, "ml_kem_512_data/ml_kem_512", 4 * 2048)
verify_key_generation(ML_KEM_768, "ml_kem_768_data/ml_kem_768", 4 * 2048)
verify_key_generation(ML_KEM_1024, "ml_kem_1024_data/ml_kem_1024", 4 * 2048)