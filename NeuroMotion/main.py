#####----------#####
#####   Make simulation data with NeuroMotion
#####----------#####

#####   Import libraries
import argparse
import pickle
import numpy as np
import torch
from easydict import EasyDict as edict
import time
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
from NeuroMotion.MSKlib.MSKpose import MSKModel
from NeuroMotion.MNPoollib.mn_params import NUM_MUS, mn_default_settings, DEPTH, ANGLE, MS_AREA
from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
from NeuroMotion.MNPoollib.mn_utils import plot_spike_trains, normalise_properties, generate_emg_mu
from BioMime.models.generator import Generator
from BioMime.utils.basics import update_config, load_generator
from BioMime.utils.plot_functions import plot_muaps



###  Set variables
def parse():
    args = argparse.ArgumentParser(description='Generate high density surface EMG signals.')   #   
    args.add_argument('--msk_model_path', type=str, help='path of MSK model', default='/mnt/d/NM/03_Programing/Simulation/NeuroMotion/NeuroMotion/MSKlib/models/ARMS_Wrist_Hand_Model_4.3')
    args.add_argument('--msk_model_name', type=str, help='name of MSK model', default='Hand_Wrist_Model_for_development.osim')
    args.add_argument('--msk_default_pose_path', type=str, help='path of default pose', default='/mnt/d/NM/03_Programing/Simulation/NeuroMotion/NeuroMotion/MSKlib/models/poses.csv')
    args.add_argument('--msk_poses', type=str, nargs='+', help='poses of MSK model', default=['default', 'default+flex', 'default', 'default+ext', 'default'])
    args.add_argument('--msk_durations', type=float, nargs='+', help='durations of MSK model', default=[1.5] * 4)
    args.add_argument('--msk_fs_mov', type=int, help='frequency of MSK model', default=5)
    args.add_argument('--ms_labels', type=str, nargs='+', help='muscles to be extracted', default=['ECRB', 'ECRL', 'PL', 'FCU', 'ECU', 'EDCI', 'FDSI'])
    args.add_argument('--morph', action='store_true', help='morph MUAPs')
    args.add_argument('--muap_file', type=str, help='initial labelled muaps', default='/mnt/d/NM/03_Programing/Simulation/NeuroMotion/ckp/muap_examples.pkl')
    args.add_argument('--fs', type=int, help='frequency of EMG signal', default=2048)
    args.add_argument('--must_ms_label', type=str, help='muscle to be extracted to make spike trains', default='ECRB')
    args.add_argument('--rslt_path', type=str, help='path of result', default='/mnt/d/NM/03_Programing/Simulation/NeuroMotion/Results')
    args.add_argument('--cfg', type=str, help='Name of configuration file', default='/mnt/d/NM/03_Programing/Simulation/NeuroMotion/ckp/config.yaml')
    args.add_argument('--muap_model_path', type=str, help='path of pretrained BioMime model for muap', default='/mnt/d/NM/03_Programing/Simulation/NeuroMotion/ckp/model_linear.pth')
    return args.parse_args()





#####   Run
if __name__ == '__main__':

    #####---------------------------------#####
    #####-----Part 01. Set parameters-----#####
    #####---------------------------------#####
    para_config = parse()
    signal = {'signal': {}}
    if not os.path.exists(para_config.rslt_path):
        os.mkdir(para_config.rslt_path)

    ####    Part 2. Define musculoskeletal (MSK) model, movements, and extract param changes
    msk = MSKModel(model_path=para_config.msk_model_path,
                   model_name=para_config.msk_model_name,
                   default_pose_path=para_config.msk_default_pose_path)
    poses = para_config.msk_poses
    durations = para_config.msk_durations
    fs_mov = para_config.msk_fs_mov
    msk.sim_mov(fs_mov, poses, durations)

    ms_lens = msk.mov2len(ms_labels=para_config.ms_labels)   #   Calculate the length of each muscle
    changes = msk.len2params()   #   Calculate the changes of each parameter
    steps = changes['steps']   #   Calculate the number of steps

    ####    Part 3. Define the MotoneuronPool of one muscle
    ### If morph MUAPs, load the MUAPs from the file (Now: False)
    if para_config.morph:
        with open(para_config.muap_file, 'rb') as fl:
            db = pickle.load(fl)
        num_mus = len(db['iz'])
    else:
        num_mus = NUM_MUS[para_config.must_ms_label]

    mn_pool = MotoneuronPool(num_mus, para_config.must_ms_label, **mn_default_settings)
    # Assign physiological properties
    fibre_density = 200     # 200 fibres per mm^2
    num_fb = np.round(MS_AREA[para_config.must_ms_label] * fibre_density)    # total number within one muscle
    config = edict({
        'num_fb': num_fb,
        'depth': DEPTH[para_config.must_ms_label],
        'angle': ANGLE[para_config.must_ms_label],
        'iz': [0.5, 0.1],
        'len': [1.0, 0.05],
        'cv': [4, 0.3]      # Recommend not setting std too large. cv range in training dataset is [3, 4.5]
    })

    if para_config.morph:
        num, depth, angle, iz, cv, length, base_muaps = normalise_properties(db, num_mus, steps)
    else:
        properties = mn_pool.assign_properties(config, normalise=True)
        num = torch.from_numpy(properties['num']).reshape(num_mus, 1).repeat(1, steps)
        depth = torch.from_numpy(properties['depth']).reshape(num_mus, 1).repeat(1, steps)
        angle = torch.from_numpy(properties['angle']).reshape(num_mus, 1).repeat(1, steps)
        iz = torch.from_numpy(properties['iz']).reshape(num_mus, 1).repeat(1, steps)
        cv = torch.from_numpy(properties['cv']).reshape(num_mus, 1).repeat(1, steps)
        length = torch.from_numpy(properties['len']).reshape(num_mus, 1).repeat(1, steps)

    
    mn_pool.init_twitches(para_config.fs)
    mn_pool.init_quisistatic_ef_model()
    duration = np.sum(durations)
    ext = np.concatenate((np.linspace(0, 0.8, round(para_config.fs * duration / 2)), np.linspace(0.8, 0, round(para_config.fs * duration / 2))))      # percentage MVC
    time_samples = len(ext)
    ext_new, spikes, fr, ipis = mn_pool.generate_spike_trains(ext, fit=False)
    plot_spike_trains(spikes, os.path.join(para_config.rslt_path, 'Spike_trains_{}.png'.format(para_config.must_ms_label)))
    signal['signal']['spikes'] = spikes




    #####------------------------------------------------------------------#####
    #####-----Part 4. Simulate MUAPs using BioMime during the movement-----#####
    #####------------------------------------------------------------------#####
    if para_config.must_ms_label == 'FCU_u' or para_config.must_ms_label == 'FCU_h':
        tgt_ms_labels = ['FCU'] * num_mus   #   Ex. ['FCU', 'FCU', 'FCU', ...] (length: num_mus)
    else:
        tgt_ms_labels = [para_config.must_ms_label] * num_mus   #   Ex. ['ECRB', 'ECRB', 'ECRB', ...] (length: num_mus)

    ch_depth = changes['depth'].loc[:, tgt_ms_labels]   #   Ex. shape: (time steps, num_mus)
    ch_cv = changes['cv'].loc[:, tgt_ms_labels]         #   Ex. shape: (time steps, num_mus)
    ch_len = changes['len'].loc[:, tgt_ms_labels]       #   Ex. shape: (time steps, num_mus)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    cfg = update_config(para_config.cfg)
    generator = Generator(cfg.Model.Generator)
    generator = load_generator(para_config.muap_model_path, generator, device)  #   Load the pretrained BioMime model
    generator.eval()
    if device == 'cuda':
        generator.cuda()

    # Filtering, not required
    # low-pass filtering for smoothing
    time_length = 96
    T = time_length / para_config.fs
    cutoff = 800
    normal_cutoff = cutoff / (0.5 * para_config.fs)
    order = 4
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    start_time = time.time()

    muaps = []
    for sp in tqdm(range(steps), dynamic_ncols=True, desc='Simulating MUAPs during dynamic movement...'):
        cond = torch.vstack((
            num[:, sp],
            depth[:, sp] * ch_depth.iloc[sp, :].values,
            angle[:, sp],
            iz[:, sp],
            cv[:, sp] * ch_cv.iloc[sp, :].values,
            length[:, sp] * ch_len.iloc[sp, :].values,
        )).transpose(1, 0)

        if not para_config.morph:
            zi = torch.randn(num_mus, cfg.Model.Generator.Latent)
            if device == 'cuda':
                zi = zi.cuda()
        else:
            if device == 'cuda':
                base_muaps = base_muaps.cuda()

        if device == 'cuda':
            cond = cond.cuda()

        if para_config.morph:
            sim = generator.generate(base_muaps, cond.float())
        else:
            sim = generator.sample(num_mus, cond.float(), cond.device, zi)

        if device == 'cuda':
            sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            sim = sim.permute(0, 2, 3, 1).detach().numpy()

        num_mu_dim, n_row_dim, n_col_dim, n_time_dim = sim.shape
        sim = filtfilt(b, a, sim.reshape(-1, n_time_dim))
        muaps.append(sim.reshape(num_mu_dim, n_row_dim, n_col_dim, n_time_dim).astype(np.float32))

    muaps = np.array(muaps)
    muaps = np.transpose(muaps, (1, 0, 2, 3, 4))
    print('--- %s seconds ---' % (time.time() - start_time))

    # plot muaps
    plot_muaps(muaps, para_config.rslt_path, np.arange(0, num_mus, 20), np.arange(0, steps, 5), suffix=para_config.must_ms_label)
    signal['signal']['muaps'] = muaps

    ####    Part 5. Generate EMG signals
    _, _, n_row, n_col, time_length = muaps.shape
    emg = np.zeros((n_row, n_col, time_samples + time_length))
    for mu in np.arange(num_mus):
        emg = emg + generate_emg_mu(muaps[mu], spikes[mu], time_samples)
    signal['signal']['data'] = emg

    print('All done, emg.shape: ', emg.shape)
    total_length = emg.shape[-1]
    t = np.linspace(0, duration, total_length)
    num_signals = 10
    row = 5  # Fix the row and select 10 columns evenly
    col_indices = np.linspace(0, emg.shape[1] - 1, num_signals, dtype=int)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_signals))

    plt.figure(figsize=(16, 8))
    plt.style.use('dark_background')  # Set dark background

    for i, col in enumerate(col_indices):
        plt.plot(t, emg[row, col] + i * np.max(np.abs(emg)), color=colors[i], label=f'Ch {col}')
        # Offset each signal on the y-axis to avoid overlap

    plt.xlabel('time')
    plt.ylabel('emg (offset)')
    plt.title('Multiple EMG signals')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(para_config.rslt_path, f'emg_multi_{para_config.must_ms_label}.jpg'))
    plt.show()

    # Save .mat file with signal    
    signal['signal']['fsamp'] = para_config.fs
    signal['signal']['grid_size'] = [n_row, n_col]
    # save .mat file
    savemat(os.path.join(para_config.rslt_path, 'HDsEMG_signal_{}.mat'.format(para_config.must_ms_label)), signal)

