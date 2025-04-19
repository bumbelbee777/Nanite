from ai import SillyAI, ModelConfig

def main():
    config = ModelConfig(
        input_dim=3,
        d_model=64,
        num_layers=2,
        nhead=4,
        dim_ff=128
    )
    
    silly_ai = SillyAI(config)
    
    # Test with Pythagorean theorem proof
    success = silly_ai.solve_problem(
        "pythagorean_theorem",
        "../examples/pythagorean_theorem_proof.sbc"
    )
    
    print(f"Proof verification: {'Success' if success else 'Failed'}")
    
    # Generate and verify proof
    success = silly_ai.solve_problem("triangle_inequality")
    print(f"Generated proof verification: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    main()