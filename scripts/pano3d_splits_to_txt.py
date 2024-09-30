import argparse
import os
import yaml

def construct_paths(data, base_path='', include_depth=False):
    paths = []
    for key, value in data.items():
        current_path = os.path.join(base_path, key) if base_path else key
        
        if isinstance(value, list):
            # Handle files in the current folder
            for item in value:
                if isinstance(item, str):  # assume item is a filename
                    full_path_png = os.path.join(current_path, item)
                    paths.append((full_path_png,))
                    if include_depth:
                        full_path_exr = full_path_png.replace('.png', '.exr').replace('emission', 'depth')
                        paths[-1] += (full_path_exr,)
                elif isinstance(item, dict):  # handle subfolders
                    paths.extend(construct_paths(item, current_path, include_depth))
        elif isinstance(value, dict):
            paths.extend(construct_paths(value, current_path, include_depth))
    
    return paths

# Function to read YAML data from file
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

# Function to write paths to a text file
def write_paths_to_file(paths, output_file):
    with open(output_file, 'w') as file:
        for path_pair in paths:
            file.write(f"{path_pair[0]}")
            if len(path_pair) > 1:
                file.write(f" {path_pair[1]}")
            file.write("\n")

def main(args):
    # Read YAML data from file
    yaml_file = args.yaml_file
    data_folder_path = args.data_folder
    
    parsed_data = read_yaml_file(yaml_file)
    
    # Construct paths
    file_paths = construct_paths(parsed_data, data_folder_path, args.include_depth)
    
    # Output file path
    output_file = args.output_file
    
    # Write paths to file
    write_paths_to_file(file_paths, output_file)
    
    print(f"Paths saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YAML file to generate paths with different extensions.")
    parser.add_argument('yaml_file', type=str, help="Path to YAML file containing folder and file structure")
    parser.add_argument('--data_folder', type=str, default='./', help="Path to the folder with data (default: current folder)")
    parser.add_argument('--output_file', type=str, default='generated_paths.txt', help="Output file to save generated paths (default: generated_paths.txt)")
    parser.add_argument('--include_depth', action='store_true', help="Flag indicating if the script should generate the paths for the depth files. If provided, include .exr paths with 'depth' instead of 'emission'.")
    
    args = parser.parse_args()
    
    main(args)
