import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    Guide to split_condition:
        'tv_ratio:4-1' : Simply split all videos into train:validation:tests ratio of 8:1:1

    """
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--num_workers", type=int, default=0, help="Pytorch dataloader workers")
    parser.add_argument("--wandb", action="store_true", help="Use wandb plotter")
    parser.add_argument("--wandb_entity", type=str, default="visual-modelling", help="wandb entity to save project and run in")
    parser.add_argument("--n_gifs", type=int, default=10, help="Number of output gifs to visualise")
    parser.add_argument("--self_output_n", type=int, default=-1, help="Number of frames to run selfoutput plotting to. -1 = all available frames")
    parser.add_argument("--jobname", type=str, required=True, help="Jobname for wandb and saving things")

    parser.add_argument_group("Dataset specific arguments")
    ############# To combine multiple datasets together, align the dataset and dataset path arguments
    parser.add_argument("--dataset_mode", type=str, default='consecutive', choices=['consecutive', 'overlap', 'full_out'])
    #############
    parser.add_argument("--split_condition", type=str, default="tv_ratio:8-1-1", help="Custom string deciding how to split datasets into train/val/test. Affiliated with a custom function in dataset")
    parser.add_argument("--img_type", type=str, default="binary", choices=["binary", "greyscale", "RGB"], help="Type of input image")
    parser.add_argument("--disable_preload", action="store_true", help="stop the preloading of the dataset object")


    args = parser.parse_args()
    print(args)


