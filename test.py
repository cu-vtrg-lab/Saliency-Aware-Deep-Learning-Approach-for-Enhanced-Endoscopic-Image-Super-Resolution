import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
from model import *  # Ensure this imports your model correctly

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='./data/test/', help='Directory of the test dataset')
    parser.add_argument('--scale_factor', type=int, default=2, help='Scale factor for super-resolution')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference')
    parser.add_argument('--model_name', type=str, default='iPASSR_2xSR_epoch100', help='Name of the model for loading')
    return parser.parse_args()

# Function to test the model
def test(cfg):
    # Load the model
    net = Net(cfg.scale_factor).to(cfg.device)
    model = torch.load('./log/' + cfg.model_name + '.pth.tar')
    net.load_state_dict(model['state_dict'])

    # Print the number of trainable parameters in the model
    print("Number of parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # List all files in the specified directory for the dataset and scale factor
    file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor))

    # Initialize total latency
    total_latency = 0

    # Loop through all files in the file list
    for idx in range(len(file_list)):
        # Load the left and right low-resolution images
        LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr0.png')
        LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr1.png')

        # Convert images to tensors and add batch dimension
        LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
        LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)

        # Convert tensors to Variables and move to the specified device
        LR_left, LR_right = Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

        # Print the current scene being processed
        scene_name = file_list[idx]
        print(f'Running Scene {scene_name} of {cfg.dataset} Dataset......')

        # Disable gradient computation for inference
        with torch.no_grad():
            # Initialize timing events
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            # Record start time, perform inference, and record end time
            start_time.record()
            SR_left, SR_right = net(LR_left, LR_right, is_training=False)
            end_time.record()

            # Wait for all operations on the GPU to complete
            torch.cuda.synchronize()

            # Calculate and print latency
            latency = start_time.elapsed_time(end_time)
            total_latency += latency
            print(f"Latency for scene {scene_name} is: {latency} milliseconds")

            # Clamp the output to [0, 1] and convert to PIL images for saving
            SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)
            save_path = f'./results/{cfg.model_name}/{cfg.dataset}'

            # Create the directory if it doesn't exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save the super-resolved images
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save(save_path + '/' + scene_name + '_L.png')
            SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
            SR_right_img.save(save_path + '/' + scene_name + '_R.png')

    # Print average latency
    print("Average latency:", total_latency / len(file_list), "milliseconds")

# Main block to execute the test function
if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['miccai_challenge_2018']  # Define the list of datasets to process
    for dataset in dataset_list:
        cfg.dataset = dataset
        test(cfg)  # Test the model on each dataset
    print('Finished!')
