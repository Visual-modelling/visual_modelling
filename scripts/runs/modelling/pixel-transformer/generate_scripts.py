import os
from collections import OrderedDict

if __name__ == '__main__':
    restrict_to_titan = False
    experiment_no = 0
    dataset = '2dBouncing'  # None for all
    # losses = ['sl1', 'ssim']
    losses = ['sl1']
    # output_activations = ['linear', 'hardsigmoid-256', 'sigmoid-256']
    output_activations = ['linear']
    learning_rates = ['3e-4', '1e-4', '3e-5', '1e-5', '3e-6', '1e-6']

    params = OrderedDict([
        ('model', 'pixel_transformer'),
        ('d_model', 4096),
        ('n_layers', 2),
        ('nhead', 1),
        ('dim_feedforward', 4096),
        ('dropout', 0.1),
        ('pixel_regression_layers', 1),
        ('norm_layer', 'layer_norm'),
    ])

    common_params = OrderedDict([
        ('split_condition', 'tv_ratio,4-1'),
        ('bsz', 64),
        ('val_bsz', 64),
        ('num_workers', 4),
        ('in_no', 5),
        ('out_no', 1),
        ('device', 0),
        ('epoch', 75),
        ('n_gifs', 20),
        ('reduction', 'mean'),
        ('img_type', 'greyscale'),
        ('shuffle', None),
        ('wandb', None),
    ])
    # jobname
    datasets = {
        '2dBouncing': {
            'dataset': 'simulations',
            'dataset_path': 'data/2dBouncing/2dMultiGrav-Y_regen/raw'},
        '2dBouncing-masked': {
            'dataset': 'simulations',
            'dataset_path': 'data/2dBouncing/2dMultiGrav-Y_regen/raw_masked'},
        '3dBouncing': {
            'dataset': 'simulations',
            'dataset_path': 'data/3dBouncing/3dRegen'},
        'hdmb51': {
            'dataset': 'hdmb51',
            'dataset_path': 'data/HDMB-51/grey_64x64_frames'},
        'mmnist': {
            'dataset': 'simulations',
            'dataset_path': 'data/moving_mnist/1_2_3'},
        'mocap': {
            'dataset': 'mocap',
            'dataset_path': 'data/mocap/grey_64x64_frames'},
        'pendulum': {
            'dataset': 'simulations',
            'dataset_path': 'data/myphysicslab/Pendulum_1200_bigger'},
        'roller': {
            'dataset': 'simulations',
            'dataset_path': 'data/myphysicslab/RollerFlight_10000_bigger'},
        'mixed': {
            'dataset': 'simulations simulations simulations simulations simulations simulations',
            'dataset_path': 'data/myphysicslab/DEMO_double_pendulum '
                            'data/3d_bouncing/hudson_true_3d_default '
                            'data/2d_bouncing/hudsons_multi_ygrav/10000 '
                            'data/moving_mnist/1_2_3 data/mocap/grey_64x64_frames '
                            'data/HDMB-51/grey_64x64_frames'},
    }

    if dataset is None:
        dataset_list = datasets.keys()
    else:
        dataset_list = [dataset]

    for dataset_name in dataset_list:
        dataset_params = datasets[dataset_name]
        for loss in losses:
            for output_activation in output_activations:
                for learning_rate in learning_rates:
                    filename_core = f'{dataset}_transformer_lr{learning_rate}_{output_activation}_{loss}_{experiment_no:03}'
                    filename = f'{filename_core}.sh'
                    if not os.path.exists(dataset_name):
                        os.makedirs(dataset_name)
                    with open(os.path.join(dataset_name, filename), 'w') as f:
                        f.write('#!/bin/bash\n')
                        f.write('#SBATCH --qos short\n')
                        f.write('#SBATCH -N 1\n')
                        f.write('#SBATCH -c 4\n')
                        f.write('#SBATCH -t 2-00:00\n')
                        f.write('#SBATCH --mem 12G\n')
                        f.write('#SBATCH -p res-gpu-small\n')
                        if restrict_to_titan:
                            f.write('#SBATCH -x gpu[0-6]\n')
                        f.write('#SBATCH --job-name 2dBouncingMG-y-pt-sl1-mean \n')
                        f.write('#SBATCH --gres gpu:1)\n')
                        f.write(f'#SBATCH -o ../../../../../.results/{filename_core}.out\n')
                        f.write('cd ../../../../..\n')
                        f.write('source python_venvs/vm/bin/activate\n')
                        f.write('export PYTHONBREAKPOINT=ipdb.set_trace\n')
                        f.write('python VM_train.py \\\n')
                        f.write(f'    --dataset {dataset_params["dataset"]} \\\n')
                        f.write(f'    --dataset_path {dataset_params["dataset_path"]} \\\n')
                        f.write(f'    --jobname {filename_core} \\\n')
                        for param_name, param_value in common_params.items():
                            if param_value is None:
                                f.write(f'    --{param_name} \\\n')
                            else:
                                f.write(f'    --{param_name} {param_value} \\\n')
                        for param_name, param_value in params.items():
                            if param_value is None:
                                f.write(f'    --{param_name} \\\n')
                            else:
                                f.write(f'    --{param_name} {param_value} \\\n')
                        f.write(f'    --loss {loss} \\\n')
                        f.write(f'    --output_activation {output_activation} \\\n')
                        f.write(f'    --lr {learning_rate} \\\n')
