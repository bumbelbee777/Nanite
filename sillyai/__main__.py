import os
import argparse
from pathlib import Path
from sillyai.api import SillyAI, ModelConfig
from PIL import Image
import torchvision.transforms as transforms

def main():
    parser = argparse.ArgumentParser(description='SillyAI - Multimodal Neuro-Symbolic AI System')
    
    # Core functionality
    parser.add_argument('--bytecode', '-b', type=str, help='Path to SillyISA bytecode file (.sbc)')
    
    # Chatbot modes
    parser.add_argument('--chatbot', '-c', action='store_true', help='Start interactive chatbot')
    parser.add_argument('--query', '-q', type=str, help='Single query for SillyAI')
    
    # Image processing
    parser.add_argument('--image', '-i', type=str, help='Process an image file')
    parser.add_argument('--describe', '-d', action='store_true', 
                       help='Generate description for the image')
    
    # Configuration
    parser.add_argument('--config', type=str, default='default_config.json',
                      help='Path to model configuration file')
    
    args = parser.parse_args()

    # Initialize with config
    config = ModelConfig.load(args.config) if os.path.exists(args.config) else ModelConfig(
        output_dim=300,       # Matches Word2Vec dimension
        concept_dim=128,
        input_dim=300,
        d_model=256,
        num_layers=4,
        nhead=8,
        dim_ff=512,
        modalities={
            'text': {
                'w2v_path': 'pretrained/word2vec.bin',
                'lowercase': True
            },
            'image': {
                'input_size': (256, 256),
                'dropout_rate': 0.1
            }
        }
    )
    
    silly_ai = SillyAI(config)

    # Image processing mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}")
            return
            
        try:
            if args.describe:
                description = silly_ai.describe_image(args.image)
                print(f"Image Description: {description}")
            else:
                features = silly_ai.process_image_file(args.image)
                print(f"Extracted image features: {features.shape}")
        except Exception as e:
            print(f"Error processing image: {str(e)}")
        return

    # Bytecode verification mode
    if args.bytecode:
        if not os.path.exists(args.bytecode):
            print(f"Error: Bytecode file not found at {args.bytecode}")
            return
            
        success = silly_ai.solve_problem(
            Path(args.bytecode).stem,
            args.bytecode
        )
        print(f"Proof verification: {'✅ Success' if success else '❌ Failed'}")
        return

    # Chatbot modes
    if args.chatbot or args.query:
        print("\n=== SillyAI Multimodal Chatbot ===")
        print("Special commands: @image <path>, @exit\n")
        
        if args.query:
            response = process_input(silly_ai, args.query)
            print(f"SillyAI: {response}")
            return
            
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['@exit', '@quit', '@getout', '@abandonship', '@bye']:
                    print("Goodbye!")
                    break
                    
                elif user_input.startswith('@image '):
                    img_path = user_input[7:]
                    if not os.path.exists(img_path):
                        print("Error: Image not found")
                        continue
                    response = silly_ai.describe_image(img_path)
                    print(f"SillyAI (image): {response}")
                else:
                    response = process_input(silly_ai, user_input)
                    print(f"SillyAI: {response}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

def process_input(model, user_input):
    """Handle text or multimodal input"""
    if user_input.startswith('@image '):
        img_path = user_input[7:]
        if not os.path.exists(img_path):
            return "I couldn't find that image file."
        return model.generate_response({"image": img_path})
    else:
        return model.generate_response(user_input)

if __name__ == "__main__":
    main()