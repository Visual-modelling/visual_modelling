import os
from collections import OrderedDict

if __name__ == '__main__':
    restrict_to_titan = False
    experiment_no = 18
    # dataset_names = ['2dBouncing', '3dBouncing', 'mmnist', 'pendulum', 'roller', 'mutual_attract']  # None for all
    dataset_names = ['2dBouncing', '3dBouncing', 'mmnist', 'pendulum', 'roller', 'mutual_attract']
    # losses = ['mse', 'sl1', 'ssim']
    losses = ['sl1', 'ssim']
    # learning_rates = ['2e-5', '1e-5', '5e-6']
    learning_rates = ['1e-5']

    label = ''

    params = OrderedDict([
        ('model', 'image_transformer'),
        ('dataset_mode', 'consecutive'),
        ('d_model', 4096),
        ('n_layers', 2),
        ('nhead', 4),
        ('dim_feedforward', 4096),
        ('dropout', 0.0),
        ('pixel_regression_layers', 1),
        ('norm_layer', 'layer_norm'),
        ('optimiser', 'adam'),
        ('output_activation', 'hardsigmoid-256'),  # ['linear-256', 'hardsigmoid-256', 'sigmoid-256']
        ('pos_encoder', 'add'),
        ('mask', None),  # enables mask
        ('feedback_training_iters', 10),
        ('sequence_loss_factor', 0.2)
    ])

    common_params = OrderedDict([
        ('split_condition', 'tv_ratio:8-1-1'),
        ('bsz', 64),
        ('val_bsz', 64),
        ('num_workers', 1),
        ('in_no', 5),
        ('out_no', 1),
        ('device', 0),
        ('epoch', 500),
        ('early_stopping', 10),
        ('min_epochs', 40),
        ('n_gifs', 20),
        ('reduction', 'mean'),
        ('img_type', 'greyscale'),
        ('shuffle', None),
        ('wandb', None),
    ])
    # jobname
    datasets_params = {
        '2dBouncing': {
            'dataset': 'simulations',
            'dataset_path': 'data/2dBouncing/2dMultiGrav-Y_regen/raw'},
        '3dBouncing': {
            'dataset': 'simulations',
            'dataset_path': 'data/3dBouncing/3dRegen'},
        'mmnist': {
            'dataset': 'simulations',
            'dataset_path': 'data/moving_mnist/1_2_3'},
        'pendulum': {
            'dataset': 'simulations',
            'dataset_path': 'data/myphysicslab/Pendulum_10000'},
        'roller': {
            'dataset': 'simulations',
            'dataset_path': 'data/myphysicslab/RollerFlight_10000_bigger'},
        'mutual_attract': {
            'dataset': 'simulations',
            'dataset_path': 'data/myphysicslab/mutualAttract_10000'},
        'mixed': {
            'dataset': 'simulations simulations simulations simulations simulations simulations',
            'dataset_path': 'data/myphysicslab/DEMO_double_pendulum '
                            'data/3d_bouncing/hudson_true_3d_default '
                            'data/2d_bouncing/hudsons_multi_ygrav/10000 '
                            'data/moving_mnist/1_2_3 data/mocap/grey_64x64_frames '
                            'data/HDMB-51/grey_64x64_frames'},
    }

    if dataset_names is None:
        dataset_names = datasets_params.keys()

    for dataset_name in dataset_names:
        dataset_params = datasets_params[dataset_name]
        for loss in losses:
            for learning_rate in learning_rates:
                filename_core = f'{dataset_name}_transformer_lr{learning_rate}_{loss}_{label}{experiment_no:03}'
                filename = f'{filename_core}.sh'
                if not os.path.exists(dataset_name):
                    os.makedirs(dataset_name)
                with open(os.path.join(dataset_name, filename), 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write('#SBATCH --qos short\n')
                    f.write('#SBATCH -N 1\n')
                    f.write('#SBATCH -c 4\n')
                    f.write('#SBATCH -t 2-00:00\n')
                    f.write('#SBATCH --mem 28G\n')
                    f.write('#SBATCH -p res-gpu-small\n')
                    if restrict_to_titan:
                        f.write('#SBATCH -x gpu[0-6]\n')
                    f.write(f'#SBATCH --job-name {filename_core} \n')
                    f.write('#SBATCH --gres gpu:1 \n')
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
                    f.write(f'    --lr {learning_rate} \\\n')
