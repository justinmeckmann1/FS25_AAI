# von https://github.com/yu4u/convnet-drawer
from visualize.convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
import argparse
import config as cfg
from MNIST.Usecase_MNIST import MNISTUseCase
from CIFAR.Usecase_CIFAR import CIFARUseCase

# define path * log file to be used
path = "log/"
datafile = 'arch_log.txt'
fn_dna_data = path + datafile

# get input shape depending on use case
if cfg.USECASE == "MNIST":
    shape_input = (28, 28, 1)
elif cfg.USECASE == "CIFAR": 
    shape_input = (32, 32, 3)    
else:
    print("ERROR: Uscase {} unknown".format(cfg.USECASE))
    exit()


def drawCnnShape(id: int):
    # Variable definition
    dna_nbr = id
    fn_svg = "Architecture_DNA_{}.svg".format(dna_nbr)
    line_id_target = "ID {}:".format(dna_nbr)

    # Load architecture
    dna_found = False
    with open(fn_dna_data) as file:
        # Loop over lines
        for line in file:
            # Load DNA ID
            line_segments = line.split(sep='___')
            # Check if correct entry
            if line_segments[0] == line_id_target:
                # print(line)
                print("Found DNA ID {}".format(dna_nbr))
                dna_found = True
                break

    if not dna_found:
        print("ERROR: DNA {} not found".format(dna_nbr))
        exit()

    # Create model
    model = Model(input_shape=shape_input)

    # Loop over architecture segments
    for i in range(1, len(line_segments)):
        # load segment
        current_segment = line_segments[i].split(sep='{')
        # Check layer type
        if current_segment[0] == 'P':
            # print("Pooling")
            # extract data
            data = current_segment[1].split(sep=',')
            # extract kernel size
            kernel_size = int(data[0].split(sep=':')[1])
            # extract stride
            stride = int(data[1].split(sep=':')[1])
            # extract padding
            padding = data[2].split(sep=':')[1].strip()
            padding = padding.replace("'", "")
            padding = padding.replace("}", "")
            model.add(MaxPooling2D((kernel_size, kernel_size), strides=(stride, stride), padding=padding))

        elif current_segment[0] == 'C':
            # print("Convolution")
            # extract data
            data = current_segment[1].split(sep=',')
            # extract number of filters
            nbr_of_filters = int(data[0].split(sep=':')[1])
            # extract kernel_size
            kernel_size = int(data[1].split(sep=':')[1])
            # extract stride
            stride = int(data[2].split(sep=':')[1])
            # extract padding
            padding = data[3].split(sep=':')[1].strip()
            padding = padding.replace("'", "")
            padding = padding.replace("}", "")
            model.add(Conv2D(filters=nbr_of_filters, kernel_size=(kernel_size, kernel_size),
                             strides=(stride, stride), padding=padding))

        elif current_segment[0] == 'F':
            # print("Flatten")
            # add Flatten layer
            model.add(Flatten())

        elif current_segment[0] == 'D':
            # print("Dense")
            # extract data
            data = current_segment[1].split(sep=',')
            # extract number of neurons
            nbr_of_neurons = int(data[0].split(sep=':')[1])
            # add Dense layer
            model.add(Dense(nbr_of_neurons))

    # save as svg file
    model.save_fig(path+fn_svg)


def main():
    parser = argparse.ArgumentParser(description="ID of the DNA that shall be visualized.")
    parser.add_argument("id", type=int, help="ID of the DNA")
    args = parser.parse_args()
    
    print(f"Visualizing DNA ID: {args.id}")
    
    drawCnnShape(id=args.id)


if __name__ == "__main__":
    main()
