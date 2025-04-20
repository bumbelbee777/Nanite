import os
import argparse
from sillyai.ai import SillyAI, ModelConfig

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SillyAI - Neural-Symbolic AI System')
    parser.add_argument('--bytecode', '-b', 
                       type=str,
                       help='Path to SillyISA bytecode file (.sbc)',
                       default="../examples/pythagorean_theorem_proof.sbc")
    
    args = parser.parse_args()

    # Create configuration with all required parameters
    config = ModelConfig(
        # Required parameters
        output_dim=64,        # Output dimension
        concept_dim=32,       # Concept embedding dimension
        input_dim=3,         # Input dimension
        d_model=64,          # Model dimension
        num_layers=2,        # Number of transformer layers
        nhead=4,            # Number of attention heads
        dim_ff=128,         # Feed-forward dimension
        
        # Optional parameters with defaults
        real_mode=False,
        dynamic_mode=True,
        factorized_linear=False,
        kronecker_rank=4,
        optim_args={
            'use_toeplitz': False,
            'factorized_linear': False,
            'mixed_precision': True
        }
    )
    
    silly_ai = SillyAI(config)
    
    # Resolve absolute path
    bytecode_path = os.path.abspath(args.bytecode)
    
    if not os.path.exists(bytecode_path):
        print(f"Error: Bytecode file not found at {bytecode_path}")
        return
        
    # Test with provided bytecode
    success = silly_ai.solve_problem(
        os.path.splitext(os.path.basename(bytecode_path))[0],
        bytecode_path
    )
    
    print(f"Proof verification: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    main()