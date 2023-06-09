import os
import imageio

png_dir = '/home/vrushank/Documents/GitHub/Deep-Learning-Projects/Neural Network Visualizer'
images = []

# Get file names in the directory
file_names = sorted((fn for fn in os.listdir(png_dir) if fn.endswith('.png')), key=lambda x: int(x.split(' - ')[1].split('.')[0]))

# Read each image file and add to images list
for filename in file_names:
    print(f'Reading file {filename}...')  # add a print statement to see if the files are being read correctly
    images.append(imageio.imread(os.path.join(png_dir, filename)))

# Save as .gif using get_writer
with imageio.get_writer('output.gif', mode='I', duration=0.03) as writer:  # set duration for each frame
    for image in images:
        writer.append_data(image)
