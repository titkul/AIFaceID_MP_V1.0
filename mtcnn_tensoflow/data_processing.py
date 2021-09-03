from preprocesss import preprocessess
import argparse

# Construct the argumet parser and parse the argument
ap = argparse.ArgumentParser()

ap.add_argument("--input", default="./raw3",
                help="path to the image input")
ap.add_argument("--output", default='../facebank',
                help="path to output cut image")

args = vars(ap.parse_args())

input_datadir = args["input"]
output_datadir = args["output"]

obj=preprocessess(input_datadir,output_datadir)
nrof_images_total , nrof_successfully_aligned=obj.collect_data()

print('Total number of images: {}'.format(nrof_images_total))
print('Number of successfully aligned images: {}'.format(nrof_successfully_aligned))