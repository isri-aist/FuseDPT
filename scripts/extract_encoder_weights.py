import argparse
import torch

# Define a function to load and process weights
def process_weights(input_path, output_path):
    # Load pretrained weights from the specified input path
    pretrained_DA_weights = torch.load(input_path)

    # Process the weights as per your original code
    encoder_weights_erp = {k.replace('pretrained', 'erp_pretrained'): v for k, v in pretrained_DA_weights.items() if 'pretrained' in k}
    encoder_weights_cmp = {k.replace('pretrained', 'cmp_pretrained'): v for k, v in pretrained_DA_weights.items() if 'pretrained' in k}

    encoder_weights = {**encoder_weights_erp, **encoder_weights_cmp}

    try:
        # Save the processed weights to the specified output path
        torch.save(encoder_weights, output_path)
        print(f"Weights saved successfully to {output_path}.")
    except Exception as e:
        print(f"Error saving weights: {str(e)}")

# Set up argparse to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process pretrained weights and save them.')
    parser.add_argument('input_path', type=str, help='Path to the input .pth file')
    parser.add_argument('output_path', type=str, help='Path to save the processed weights')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()

    # Call the function with the provided input_path and output_path
    process_weights(args.input_path, args.output_path)