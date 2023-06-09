import h5py

def inspect_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print("File format is HDF5.")
            print("Contents of the file:")
            f.visititems(print_attrs)
    except OSError as e:
        print(f"Error opening the file: {e}")

def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print(f"  {key}: {val}")

if __name__ == "__main__":
    file_path = "/home/vrushank/Documents/GitHub/Deep-Learning-Projects/Neural Network Visualizer/datasets/test_catvnoncat.h5"
    inspect_h5_file(file_path)
