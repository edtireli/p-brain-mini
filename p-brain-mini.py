#!/usr/bin/env python3
import os, subprocess, json, tkinter as tk, scipy.io, pickle
from tkinter import filedialog
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from matplotlib.gridspec import GridSpec
from scipy.signal import argrelextrema, find_peaks, butter, filtfilt
from scipy.linalg import solve
from PIL import Image, ImageTk

DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

FA_GLOBAL = None
TR_GLOBAL = None

###############################################################################
# Helper: Save tissue function as a MATLAB .mat file.
###############################################################################
def save_tissue_function(method, tissue_name, identifier, s_input, c_input, analysis_dir):
    """
    Saves the tissue function as a .mat file with variables:
      s_input: MR signal (raw)
      c_input: computed concentration curve.
    The file is saved under analysis/matlab/<method>/ using the naming convention:
      <tissue_name>_<identifier>.mat
    """
    matlab_dir = os.path.join(analysis_dir, 'matlab', method)
    os.makedirs(matlab_dir, exist_ok=True)
    filename = os.path.join(matlab_dir, f"{tissue_name}_{identifier}.mat")
    scipy.io.savemat(filename, {'s_input': s_input, 'c_input': c_input})
    debug_print(f"Saved {tissue_name} function ({identifier}) to {filename}")

###############################################################################
# 1) Masks & Overlays
###############################################################################
def plot_predictions_with_masks(image,
                                wm_mask,
                                cortical_gm_mask,
                                subcortical_gm_mask,
                                gm_brainstem_mask,
                                gm_cerebellum_mask,
                                wm_cerebellum_mask,
                                wm_cc_mask,
                                analysis_seg_dir):
    """
    Overlays T1 slices with color-coded masks: 
      * WM=Blue, Cortical GM=Bright Red, Subcortical GM=Dark Red
      * Brainstem GM=Orange, Cerebellum GM=Yellow
      * Cerebellum WM=Cyan, Corpus Callosum WM=Magenta
    Saves a single PNG 'T1_WM_GM_masks.png' under analysis_seg_dir.
    """
    n_slices = min(image.shape[2],
                   wm_mask.shape[2],
                   cortical_gm_mask.shape[2],
                   subcortical_gm_mask.shape[2],
                   gm_brainstem_mask.shape[2],
                   gm_cerebellum_mask.shape[2],
                   wm_cerebellum_mask.shape[2],
                   wm_cc_mask.shape[2])
    n_cols = 5
    n_rows = (n_slices + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * image.shape[1]/100,
                                      n_rows * image.shape[0]/100),
                             dpi=100)

    for i in range(n_slices):
        row = i // n_cols
        col = i % n_cols
        sl = np.rot90(image[:, :, i])

        def safe_mask(mask_arr):
            # If shapes differ, resize nearest neighbor
            if mask_arr.shape[:2] != image[:, :, i].shape:
                return np.rot90(
                    resize(mask_arr[:, :, i],
                           image[:, :, i].shape,
                           order=0,
                           preserve_range=True)
                )
            else:
                return np.rot90(mask_arr[:, :, i])

        wm0  = safe_mask(wm_mask)
        cg0  = safe_mask(cortical_gm_mask)
        sc0  = safe_mask(subcortical_gm_mask)
        bs0  = safe_mask(gm_brainstem_mask)
        cer0 = safe_mask(gm_cerebellum_mask)
        wmc0 = safe_mask(wm_cerebellum_mask)
        wcc0 = safe_mask(wm_cc_mask)

        overlay = np.zeros((*sl.shape, 3))
        # White matter: Blue
        overlay[:, :, 2][wm0 == 1] = 1.0
        # Cortical GM: bright red
        overlay[:, :, 0][cg0 == 1] = 1.0
        # Subcortical GM: dark red
        overlay[:, :, 0][sc0 == 1] = 0.5
        # Brainstem GM: orange (red + half green)
        overlay[:, :, 0][bs0 == 1] = 1.0
        overlay[:, :, 1][bs0 == 1] = 0.5
        # Cerebellum GM: yellow (red + green)
        overlay[:, :, 0][cer0 == 1] = 1.0
        overlay[:, :, 1][cer0 == 1] = 1.0
        # Cerebellum WM: cyan (green + blue)
        overlay[:, :, 1][wmc0 == 1] = 1.0
        overlay[:, :, 2][wmc0 == 1] = 1.0
        # Corpus callosum WM: magenta (red + blue)
        overlay[:, :, 0][wcc0 == 1] = 1.0
        overlay[:, :, 2][wcc0 == 1] = 1.0

        ax = axes[row, col] if (n_rows*n_cols > 1) else axes
        ax.imshow(sl, cmap='gray', aspect='equal')
        ax.imshow(overlay, alpha=0.5, aspect='equal')
        ax.set_title(f"Slice {i+1}")
        ax.axis("off")

    # Remove leftover empty subplots
    for j in range(n_slices, n_rows*n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    os.makedirs(analysis_seg_dir, exist_ok=True)
    out_file = os.path.join(analysis_seg_dir, 'T1_WM_GM_masks.png')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)

def create_masks(seg_path, analysis_seg_dir, force=True):
    """
    Creates binarized segmentation masks from seg.nii.gz, saving them
    in analysis_seg_dir. Force to True -> always rerun binarization.
    """
    os.makedirs(analysis_seg_dir, exist_ok=True)
    # Temporary files
    wm_path  = os.path.join(analysis_seg_dir, "temp_wm.nii.gz")
    sub_path = os.path.join(analysis_seg_dir, "temp_subcortical_gm.nii.gz")
    gm_path  = os.path.join(analysis_seg_dir, "temp_gm.nii.gz")

    if force or not os.path.exists(wm_path):
        subprocess.run(f"mri_binarize --i {seg_path} --all-wm --o {wm_path}",
                       shell=True, check=True)
    if force or not os.path.exists(sub_path):
        subprocess.run(f"mri_binarize --i {seg_path} --subcort-gm --o {sub_path}",
                       shell=True, check=True)
    if force or not os.path.exists(gm_path):
        subprocess.run(f"mri_binarize --i {seg_path} --gm --o {gm_path}",
                       shell=True, check=True)
    gm_bs = os.path.join(analysis_seg_dir, "gm_brainstem.nii.gz")
    if force or not os.path.exists(gm_bs):
        subprocess.run(f"mri_binarize --i {seg_path} --match 16 --o {gm_bs}",
                       shell=True, check=True)
    gm_cer = os.path.join(analysis_seg_dir, "gm_cerebellum.nii.gz")
    if force or not os.path.exists(gm_cer):
        subprocess.run(f"mri_binarize --i {seg_path} --match 8 47 --o {gm_cer}",
                       shell=True, check=True)
    wm_cer = os.path.join(analysis_seg_dir, "wm_cerebellum.nii.gz")
    if force or not os.path.exists(wm_cer):
        subprocess.run(f"mri_binarize --i {seg_path} --match 7 46 --o {wm_cer}",
                       shell=True, check=True)
    wm_cc = os.path.join(analysis_seg_dir, "wm_cc.nii.gz")
    if force or not os.path.exists(wm_cc):
        subprocess.run(f"mri_binarize --i {seg_path} --match 251 252 253 254 255 --o {wm_cc}",
                       shell=True, check=True)

    cortical_gm = os.path.join(analysis_seg_dir, "cortical_gm.nii.gz")
    if force or not os.path.exists(cortical_gm):
        cmd_cg = (f"fslmaths {gm_path} -sub {sub_path} -sub {gm_bs} -sub {gm_cer} "
                  f"-thr 0.5 -bin {cortical_gm}")
        subprocess.run(cmd_cg, shell=True, check=True)

    subcortical_gm = os.path.join(analysis_seg_dir, "subcortical_gm.nii.gz")
    if force or not os.path.exists(subcortical_gm):
        cmd_sc = (f"fslmaths {sub_path} -sub {gm_bs} -sub {gm_cer} "
                  f"-thr 0.5 -bin {subcortical_gm}")
        subprocess.run(cmd_sc, shell=True, check=True)

    wm_final = os.path.join(analysis_seg_dir, "wm.nii.gz")
    if force or not os.path.exists(wm_final):
        cmd_wm = (f"fslmaths {wm_path} -sub {wm_cer} -sub {wm_cc} "
                  f"-thr 0.5 -bin {wm_final}")
        subprocess.run(cmd_wm, shell=True, check=True)

    # Remove temporary files
    for tmp in [wm_path, sub_path, gm_path]:
        if os.path.exists(tmp):
            os.remove(tmp)

    return {
        "wm": wm_final,
        "subcortical_gm": subcortical_gm,
        "cortical_gm": cortical_gm,
        "gm_brainstem": gm_bs,
        "gm_cerebellum": gm_cer,
        "wm_cerebellum": wm_cer,
        "wm_cc": wm_cc
    }

def construct_convolution_matrix(C_a, delta_t):
    n = len(C_a)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, :i+1] = C_a[i::-1] * delta_t
    return A

def tikhonov_regularization(A, c, lambd):
    n = A.shape[1]
    L = np.eye(n)
    ATA = A.T @ A
    LTL = L.T @ L
    reg_mat = ATA + lambd * LTL
    return np.linalg.solve(reg_mat, A.T @ c)

# --- Modified patlak_analysis_plotting function ---
def patlak_analysis_plotting(c_tissue, c_input, time):
    import numpy as np
    from scipy.integrate import cumtrapz

    c_tissue = np.asarray(c_tissue)
    c_input = np.asarray(c_input)
    time = np.asarray(time)

    with np.errstate(divide='ignore', invalid='ignore'):
        x_patlak = cumtrapz(c_input, time, initial=0) / c_input
        y_patlak = c_tissue / c_input

    # Replace any infinite values with NaN, but do not replace existing NaNs.
    x_patlak[np.isinf(x_patlak)] = np.nan
    y_patlak[np.isinf(y_patlak)] = np.nan

    valid = (x_patlak > 0) & (y_patlak > 0)
    if not np.any(valid):
        return np.nan, np.nan, np.nan, x_patlak, y_patlak, valid

    x_max = np.nanmax(x_patlak[valid])
    x_min = x_max / 3.0
    idx = (x_patlak >= x_min) & (x_patlak <= x_max) & valid
    if np.sum(idx) < 2:
        return np.nan, np.nan, np.nan, x_patlak, y_patlak, idx

    x_used = x_patlak[idx]
    y_used = y_patlak[idx]
    x_mean = np.nanmean(x_used)
    y_mean = np.nanmean(y_used)
    denominator = np.dot(x_used - x_mean, x_used - x_mean)
    if denominator == 0:
        return np.nan, np.nan, np.nan, x_patlak, y_patlak, idx
    Ki_raw = np.dot(x_used - x_mean, y_used - y_mean) / denominator
    lam_raw = y_mean - Ki_raw * x_mean

    residuals = y_used - (lam_raw + Ki_raw * x_used)
    dof = len(x_used) - 2
    SD_Ki_raw = np.nan if dof < 1 else np.sqrt(np.sum(residuals ** 2) / (denominator * dof))

    Ki = Ki_raw * 6000
    SD_Ki = SD_Ki_raw * 6000 if not np.isnan(SD_Ki_raw) else np.nan
    lam = lam_raw * 100

    included_mask = np.zeros_like(x_patlak, dtype=bool)
    included_mask[idx] = True

    return Ki, lam, SD_Ki, x_patlak, y_patlak, included_mask
# --- End of patlak_analysis_plotting modification ---

def compute_CTC_VFA(s, M0, FA, TR, r1, T1, beta=4.5):
    r1 = 1 / T1
    A = s / (M0 * np.sin(FA))
    delta_R1 = -np.log((A - 1) / (A * np.cos(FA) - 1)) / TR
    return (delta_R1 - r1) / beta

def custom_shifter(array, baseline_point):
    if baseline_point is None:
        baseline_point = 10
    arr_after = array[baseline_point:]
    if np.min(arr_after) < 0:
        arr_shifted = arr_after - np.min(arr_after)
    else:
        arr_shifted = arr_after
    return np.concatenate([np.zeros(baseline_point), arr_shifted])

def find_baseline_point_advanced(data, fs=15, cutoff=4.0, order=3, skip_points=10):
    return 10

def compute_Ki_from_atlas(atlas_path,
                          data_4d,
                          T1_matrix,
                          M0_matrix,
                          time_points_s,
                          C_a_full,
                          affine,
                          analysis_dir,
                          compute_CTC,
                          find_baseline_point_advanced,
                          custom_shifter,
                          patlak_analysis_plotting):
    atlas_img  = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)
    unique_labels = np.unique(atlas_data)
    unique_labels = unique_labels[unique_labels != 0]

    Ki_map    = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    SD_Ki_map = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    vp_map    = np.full(atlas_data.shape, np.nan, dtype=np.float32)

    perf_dir = os.path.join(analysis_dir, "perfusion")
    os.makedirs(perf_dir, exist_ok=True)

    for lbl in unique_labels:
        mask = (atlas_data == lbl)
        indices = np.argwhere(mask)
        if len(indices) < 1:
            continue
        raw_signals = []
        conc_signals = []
        for (x, y, z) in indices:
            v = data_4d[x, y, z, :]
            if np.isnan(v).any():
                continue
            T1v = T1_matrix[x, y, z]
            M0v = M0_matrix[x, y, z]
            c0  = compute_CTC_VFA(v, M0v, FA_GLOBAL, TR_GLOBAL, 4.5, T1v)
            if np.isnan(c0).any():
                continue
            b = find_baseline_point_advanced(c0)
            c1 = custom_shifter(c0, b)
            if np.isnan(c1).any():
                continue
            raw_signals.append(v)
            conc_signals.append(c1)
        if len(conc_signals) == 0:
            continue
        raw_signals = np.stack(raw_signals, axis=0)
        conc_signals = np.stack(conc_signals, axis=0)
        median_raw = np.median(raw_signals, axis=0)
        median_conc = np.median(conc_signals, axis=0)
        
        # Save the global (atlas-based) tissue function under analysis/matlab/atlas/
        save_tissue_function("atlas", f"tissue_label_{lbl}", "atlas", median_raw, median_conc, analysis_dir)
        
        mm = min(len(median_conc), len(C_a_full))
        c_t_label = median_conc[:mm]
        c_a_label = C_a_full[:mm]
        t_label   = time_points_s[:mm]
        Ki, lam, SD_Ki, _, _, _ = patlak_analysis_plotting(c_t_label,
                                                           c_a_label,
                                                           t_label)
        Ki_map[mask]    = Ki
        SD_Ki_map[mask] = SD_Ki
        vp_map[mask]    = lam

    nib.save(nib.Nifti1Image(Ki_map, affine),
             os.path.join(perf_dir, 'Ki_map_atlas.nii.gz'))
    nib.save(nib.Nifti1Image(SD_Ki_map, affine),
             os.path.join(perf_dir, 'SD_Ki_map_atlas.nii.gz'))
    nib.save(nib.Nifti1Image(vp_map, affine),
             os.path.join(perf_dir, 'vp_map_atlas.nii.gz'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.transform import resize

def plot_ctcs_and_patlak(
    dce_img_slice,
    wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
    avg_wm_ctc, avg_cortical_gm_ctc, avg_subcortical_gm_ctc,
    x_patlak_wm, y_patlak_wm, Ki_wm, lambda_wm,
    x_patlak_cortical_gm, y_patlak_cortical_gm, Ki_cortical_gm, lambda_cortical_gm,
    x_patlak_subcortical_gm, y_patlak_subcortical_gm, Ki_subcortical_gm, lambda_subcortical_gm,
    slice_idx,
    save_path=None,
    boundary_mask=None,
    boundary_ctc=None,
    x_patlak_boundary=None,
    y_patlak_boundary=None,
    Ki_boundary=None,
    lambda_boundary=None,
    included_wm=None,
    included_cortical_gm=None,
    included_subcortical_gm=None,
    included_boundary=None,
    gm_brainstem_ctc=None,
    x_patlak_gm_brainstem=None,
    y_patlak_gm_brainstem=None,
    Ki_gm_brainstem=None,
    lambda_gm_brainstem=None,
    included_gm_brainstem=None,
    gm_cerebellum_ctc=None,
    x_patlak_gm_cerebellum=None,
    y_patlak_gm_cerebellum=None,
    Ki_gm_cerebellum=None,
    lambda_gm_cerebellum=None,
    included_gm_cerebellum=None,
    wm_cerebellum_ctc=None,
    x_patlak_wm_cerebellum=None,
    y_patlak_wm_cerebellum=None,
    Ki_wm_cerebellum=None,
    lambda_wm_cerebellum=None,
    included_wm_cerebellum=None,
    wm_cc_ctc=None,
    x_patlak_wm_cc=None,
    y_patlak_wm_cc=None,
    Ki_wm_cc=None,
    lambda_wm_cc=None,
    included_wm_cc=None,
    gm_brainstem_mask_dce=None,
    gm_cerebellum_mask_dce=None,
    wm_cerebellum_mask_dce=None,
    wm_cc_mask_dce=None
):
    def resize_and_binarize(mask, target_shape):
        resized = resize(mask.astype(float), target_shape, order=0,
                         preserve_range=True, anti_aliasing=False)
        return (resized > 0.5).astype(float)

    dce_vmin, dce_vmax = np.percentile(dce_img_slice, (1, 99))
    dce_norm = np.clip(dce_img_slice, dce_vmin, dce_vmax)
    dce_norm = (dce_norm - dce_vmin) / (dce_vmax - dce_vmin)

    wm_r_dce = resize_and_binarize(wm_mask_dce, dce_img_slice.shape)
    cg_r_dce = resize_and_binarize(cortical_gm_mask_dce, dce_img_slice.shape)
    sc_r_dce = resize_and_binarize(subcortical_gm_mask_dce, dce_img_slice.shape)

    if gm_brainstem_mask_dce is not None:
        gm_brainstem_r_dce = resize_and_binarize(gm_brainstem_mask_dce, dce_img_slice.shape)
    else:
        gm_brainstem_r_dce = np.zeros_like(dce_img_slice)

    if gm_cerebellum_mask_dce is not None:
        gm_cerebellum_r_dce = resize_and_binarize(gm_cerebellum_mask_dce, dce_img_slice.shape)
    else:
        gm_cerebellum_r_dce = np.zeros_like(dce_img_slice)

    if wm_cerebellum_mask_dce is not None:
        wm_cerebellum_r_dce = resize_and_binarize(wm_cerebellum_mask_dce, dce_img_slice.shape)
    else:
        wm_cerebellum_r_dce = np.zeros_like(dce_img_slice)

    if wm_cc_mask_dce is not None:
        wm_cc_r_dce = resize_and_binarize(wm_cc_mask_dce, dce_img_slice.shape)
    else:
        wm_cc_r_dce = np.zeros_like(dce_img_slice)

    if boundary_mask is not None:
        boundary_resized = resize_and_binarize(boundary_mask, dce_img_slice.shape)
    else:
        boundary_resized = np.zeros_like(dce_img_slice)

    colors = {
        'wm':             [0, 0, 1, 0.5],
        'cortical_gm':    [1, 0, 0, 0.5],
        'subcortical_gm': [0.5, 0, 0, 0.5],
        'gm_brainstem':   [1, 0.5, 0, 0.5],
        'gm_cerebellum':  [1, 1, 0, 0.5],
        'wm_cerebellum':  [0, 1, 1, 0.5],
        'wm_cc':          [1, 0, 1, 0.5],
        'boundary':       [0, 1, 0, 0.5]
    }

    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1])
    gs.update(hspace=0.4)

    ax_dce  = fig.add_subplot(gs[0, 0])
    ax_ctcs = fig.add_subplot(gs[1, 0])
    ax_pat  = fig.add_subplot(gs[2, 0])

    def overlay_mask(ax, mask, rgba):
        if mask.any():
            overlay_img = np.zeros((*mask.shape, 4))
            overlay_img[..., :3] = rgba[:3]
            overlay_img[..., 3]  = rgba[3]*mask
            ax.imshow(np.rot90(overlay_img), interpolation='none')

    ax_dce.imshow(np.rot90(dce_norm), cmap='gray', vmin=0, vmax=1)
    overlay_mask(ax_dce, wm_r_dce,             colors['wm'])
    overlay_mask(ax_dce, cg_r_dce,            colors['cortical_gm'])
    overlay_mask(ax_dce, sc_r_dce,            colors['subcortical_gm'])
    overlay_mask(ax_dce, gm_brainstem_r_dce,  colors['gm_brainstem'])
    overlay_mask(ax_dce, gm_cerebellum_r_dce, colors['gm_cerebellum'])
    overlay_mask(ax_dce, wm_cerebellum_r_dce, colors['wm_cerebellum'])
    overlay_mask(ax_dce, wm_cc_r_dce,         colors['wm_cc'])
    if boundary_mask is not None:
        overlay_mask(ax_dce, boundary_resized, colors['boundary'])

    ax_dce.set_title(f"DCE Slice {slice_idx} with Masks")
    ax_dce.axis("off")

    ax_ctcs.set_facecolor("#f7f7f7")
    ax_ctcs.plot(avg_wm_ctc, label='WM', color='blue')
    ax_ctcs.plot(avg_cortical_gm_ctc, label='Cortical GM', color='red')
    ax_ctcs.plot(avg_subcortical_gm_ctc, label='Subcortical GM', color='darkred')
    if gm_brainstem_ctc is not None and gm_brainstem_ctc.size>0:
        ax_ctcs.plot(gm_brainstem_ctc,     label='GM Brainstem', color='orange')
    if gm_cerebellum_ctc is not None and gm_cerebellum_ctc.size>0:
        ax_ctcs.plot(gm_cerebellum_ctc,    label='GM Cerebellum', color='gold')
    if wm_cerebellum_ctc is not None and wm_cerebellum_ctc.size>0:
        ax_ctcs.plot(wm_cerebellum_ctc,    label='WM Cerebellum', color='cyan')
    if boundary_ctc is not None and boundary_ctc.size>0:
        ax_ctcs.plot(boundary_ctc,         label='Boundary', color='green')

    ax_ctcs.set_title("Concentration-Time Curves")
    ax_ctcs.legend(loc='upper right')
    ax_ctcs.grid(True)

    ax_pat.set_facecolor("#f7f7f7")
    def scatter_patlak(ax, x, y, included, label, c):
        finite = np.isfinite(x) & np.isfinite(y)
        if included is None:
            included = np.ones_like(x, dtype=bool)
        # Use only finite points that are marked as included
        valid_mask = finite & included
        ax.scatter(x[valid_mask], y[valid_mask], label=label, color=c, marker='o')
        # Optionally plot points not included but are finite
        non_included = finite & ~included
        ax.scatter(x[non_included], y[non_included], facecolors='none', edgecolors=c)


    if not np.isnan(Ki_wm):
        scatter_patlak(ax_pat, x_patlak_wm, y_patlak_wm, included_wm, "WM", 'blue')
        ax_pat.plot(x_patlak_wm,
                    (lambda_wm/100) + (Ki_wm/6000) * x_patlak_wm,
                    color='blue', ls='--')

    if not np.isnan(Ki_cortical_gm):
        scatter_patlak(ax_pat, x_patlak_cortical_gm, y_patlak_cortical_gm, included_cortical_gm, "Cortical GM", 'red')
        ax_pat.plot(x_patlak_cortical_gm,
                    (lambda_cortical_gm/100) + (Ki_cortical_gm/6000) * x_patlak_cortical_gm,
                    color='red', ls='--')

    if not np.isnan(Ki_subcortical_gm):
        scatter_patlak(ax_pat, x_patlak_subcortical_gm, y_patlak_subcortical_gm, included_subcortical_gm, "Subcort GM", 'darkred')
        ax_pat.plot(x_patlak_subcortical_gm,
                    (lambda_subcortical_gm/100) + (Ki_subcortical_gm/6000) * x_patlak_subcortical_gm,
                    color='darkred', ls='--')

    if gm_brainstem_ctc is not None and not np.isnan(Ki_gm_brainstem):
        scatter_patlak(ax_pat, x_patlak_gm_brainstem, y_patlak_gm_brainstem, included_gm_brainstem, "GM Brainstem", 'orange')
        ax_pat.plot(x_patlak_gm_brainstem,
                    (lambda_gm_brainstem/100) + (Ki_gm_brainstem/6000) * x_patlak_gm_brainstem,
                    color='orange', ls='--')

    if gm_cerebellum_ctc is not None and not np.isnan(Ki_gm_cerebellum):
        scatter_patlak(ax_pat, x_patlak_gm_cerebellum, y_patlak_gm_cerebellum, included_gm_cerebellum, "GM Cerebellum", 'gold')
        ax_pat.plot(x_patlak_gm_cerebellum,
                    (lambda_gm_cerebellum/100) + (Ki_gm_cerebellum/6000) * x_patlak_gm_cerebellum,
                    color='gold', ls='--')

    if wm_cerebellum_ctc is not None and not np.isnan(Ki_wm_cerebellum):
        scatter_patlak(ax_pat, x_patlak_wm_cerebellum, y_patlak_wm_cerebellum, included_wm_cerebellum, "WM Cerebellum", 'cyan')
        ax_pat.plot(x_patlak_wm_cerebellum,
                    (lambda_wm_cerebellum/100) + (Ki_wm_cerebellum/6000) * x_patlak_wm_cerebellum,
                    color='cyan', ls='--')

    if wm_cc_ctc is not None and not np.isnan(Ki_wm_cc):
        scatter_patlak(ax_pat, x_patlak_wm_cc, y_patlak_wm_cc, included_wm_cc, "WM CC", 'magenta')
        ax_pat.plot(x_patlak_wm_cc,
                    (lambda_wm_cc/100) + (Ki_wm_cc/6000) * x_patlak_wm_cc,
                    color='magenta', ls='--')

    if boundary_ctc is not None and not np.isnan(Ki_boundary):
        scatter_patlak(ax_pat, x_patlak_boundary, y_patlak_boundary, included_boundary, "Boundary", 'green')
        ax_pat.plot(x_patlak_boundary,
                    (lambda_boundary/100) + (Ki_boundary/6000) * x_patlak_boundary,
                    color='green', ls='--')

    ax_pat.set_title("Patlak Fit")
    ax_pat.legend(loc='lower left')
    ax_pat.grid(True)

    all_y = []
    for y_arr in [y_patlak_wm, y_patlak_cortical_gm, y_patlak_subcortical_gm,
                  y_patlak_gm_brainstem, y_patlak_gm_cerebellum, y_patlak_wm_cerebellum,
                  y_patlak_wm_cc, y_patlak_boundary]:
        if y_arr is not None and y_arr.size > 0:
            cutoff = int(len(y_arr) / 3)
            if cutoff < len(y_arr):
                all_y.append(y_arr[cutoff:])
    if all_y:
        combined_y = np.concatenate(all_y)
        y_min = np.min(combined_y)
        y_max = np.max(combined_y)
        margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        ax_pat.set_ylim(y_min - margin, y_max + margin)

    lines = []
    if not np.isnan(Ki_wm):
        lines.append(f"WM: Ki={Ki_wm:.5f}, λ={lambda_wm:.5f}")
    if not np.isnan(Ki_cortical_gm):
        lines.append(f"Cort GM: Ki={Ki_cortical_gm:.5f}, λ={lambda_cortical_gm:.5f}")
    if not np.isnan(Ki_subcortical_gm):
        lines.append(f"Subcort GM: Ki={Ki_subcortical_gm:.5f}, λ={lambda_subcortical_gm:.5f}")
    if not np.isnan(Ki_gm_brainstem):
        lines.append(f"Brainstem GM: Ki={Ki_gm_brainstem:.5f}, λ={lambda_gm_brainstem:.5f}")
    if not np.isnan(Ki_gm_cerebellum):
        lines.append(f"Cerebellum GM: Ki={Ki_gm_cerebellum:.5f}, λ={lambda_gm_cerebellum:.5f}")
    if not np.isnan(Ki_wm_cerebellum):
        lines.append(f"Cerebellum WM: Ki={Ki_wm_cerebellum:.5f}, λ={lambda_wm_cerebellum:.5f}")
    if not np.isnan(Ki_wm_cc):
        lines.append(f"WM CC: Ki={Ki_wm_cc:.5f}, λ={lambda_wm_cc:.5f}")
    if not np.isnan(Ki_boundary):
        lines.append(f"Boundary: Ki={Ki_boundary:.5f}, λ={lambda_boundary:.5f}")

    if lines:
        ax_pat.text(
            0.5, -0.2, "\n".join(lines),
            transform=ax_pat.transAxes,
            fontsize=10,
            color='black',
            ha='center',
            va='top',
            bbox=dict(facecolor='white', alpha=0.7)
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

from scipy.ndimage import binary_dilation

def compute_and_plot_ctcs_median(data_4d,
                                 wm_mask_t1,
                                 cg_mask_t1,
                                 sc_mask_t1,
                                 wm_mask_dce,
                                 cg_mask_dce,
                                 sc_mask_dce,
                                 T1_matrix,
                                 M0_matrix,
                                 analysis_directory,
                                 time_points_s,
                                 image_directory,
                                 dce_path,
                                 aif,
                                 boundary=False,
                                 compute_per_voxel_Ki=False,
                                 compute_per_voxel_CBF=False,
                                 gm_bs_mask_dce=None,
                                 gm_cer_mask_dce=None,
                                 wm_cerebellum_mask_dce=None,
                                 wm_cc_mask_dce=None):
    dce_img   = nib.load(dce_path)
    dce_shape = data_4d.shape[:3]

    def ensure_shape(mask_path_or_vol):
        if isinstance(mask_path_or_vol, str):
            vol = nib.load(mask_path_or_vol).get_fdata()
        else:
            vol = mask_path_or_vol
        if vol.shape[:3] != dce_shape:
            return resample_from_to(nib.Nifti1Image(vol, dce_img.affine),
                                    (dce_shape, dce_img.affine),
                                    order=0).get_fdata()
        else:
            return vol

    wm_mask_dce = ensure_shape(wm_mask_dce)
    wm_mask_dce = (wm_mask_dce>0.5).astype(np.uint8)

    cg_mask_dce = ensure_shape(cg_mask_dce)
    cg_mask_dce = (cg_mask_dce>0.5).astype(np.uint8)

    sc_mask_dce = ensure_shape(sc_mask_dce)
    sc_mask_dce = (sc_mask_dce>0.5).astype(np.uint8)

    def load_or_zeros(path_or_vol):
        if path_or_vol is not None:
            if isinstance(path_or_vol, str):
                tmp_vol = nib.load(path_or_vol).get_fdata()
            else:
                tmp_vol = path_or_vol
            return tmp_vol[..., :dce_shape[2]]
        else:
            return np.zeros(dce_shape, dtype=np.uint8)

    gm_brainstem_mask  = load_or_zeros(gm_bs_mask_dce)
    gm_cerebellum_mask = load_or_zeros(gm_cer_mask_dce)
    wm_cerebellum_mask = load_or_zeros(wm_cerebellum_mask_dce)
    wm_cc_mask         = load_or_zeros(wm_cc_mask_dce)

    gm_brainstem_mask   = ensure_shape(gm_brainstem_mask)
    gm_cerebellum_mask  = ensure_shape(gm_cerebellum_mask)
    wm_cerebellum_mask  = ensure_shape(wm_cerebellum_mask)
    wm_cc_mask          = ensure_shape(wm_cc_mask)

    gm_brainstem_mask   = (gm_brainstem_mask>0.5).astype(np.uint8)
    gm_cerebellum_mask  = (gm_cerebellum_mask>0.5).astype(np.uint8)
    wm_cerebellum_mask  = (wm_cerebellum_mask>0.5).astype(np.uint8)
    wm_cc_mask          = (wm_cc_mask>0.5).astype(np.uint8)

    os.makedirs(os.path.join(analysis_directory, 'CTC'), exist_ok=True)
    if compute_per_voxel_Ki or compute_per_voxel_CBF:
        os.makedirs(os.path.join(analysis_directory, 'perfusion'), exist_ok=True)

    if compute_per_voxel_Ki:
        Ki_per_voxel = np.full(dce_shape, np.nan)
    if compute_per_voxel_CBF:
        CBF_per_voxel = np.full(dce_shape, np.nan)

    all_patlak_data = []

    # Initialize dictionary to accumulate global tissue functions (entire ROI)
    global_data = {
        "wm": {"raw": [], "conc": []},
        "cortical_gm": {"raw": [], "conc": []},
        "subcortical_gm": {"raw": [], "conc": []},
        "gm_brainstem": {"raw": [], "conc": []},
        "gm_cerebellum": {"raw": [], "conc": []},
        "wm_cerebellum": {"raw": [], "conc": []},
        "wm_cc": {"raw": [], "conc": []},
    }

    # Helper: Process indices for median (sliced method)
    def process_ctcs(indices, m_len):
        raw_list = []
        conc_list = []
        for (x, y) in indices:
            voxel_tc = data_4d[x, y, i, :]
            T1_val   = T1_matrix[x, y, i]
            M0_val   = M0_matrix[x, y, i]
            if T1_val<=0 or M0_val<=0:
                continue
            s = voxel_tc
            c0 = compute_CTC_VFA(voxel_tc, M0_val, FA_GLOBAL, TR_GLOBAL, 1/T1_val, T1_val, beta=4.5)
            b_pt  = find_baseline_point_advanced(c0)
            c = custom_shifter(c0, b_pt)
            if np.isnan(c).any() or np.all(c==0):
                continue
            raw_list.append(s[:m_len])
            conc_list.append(c[:m_len])
        if raw_list:
            return np.median(np.stack(raw_list, axis=0), axis=0), np.median(np.stack(conc_list, axis=0), axis=0)
        else:
            return np.array([]), np.array([])

    # Helper: Process indices for voxelwise (keep all voxel curves)
    def process_voxel_ctcs(indices, m_len):
        raw_list = []
        conc_list = []
        for (x, y) in indices:
            voxel_tc = data_4d[x, y, i, :]
            T1_val   = T1_matrix[x, y, i]
            M0_val   = M0_matrix[x, y, i]
            if T1_val<=0 or M0_val<=0:
                continue
            s = voxel_tc
            c0 = compute_CTC_VFA(voxel_tc, M0_val, FA_GLOBAL, TR_GLOBAL, 1/T1_val, T1_val, beta=4.5)
            b_pt  = find_baseline_point_advanced(c0)
            c = custom_shifter(c0, b_pt)
            if np.isnan(c).any() or np.all(c==0):
                continue
            raw_list.append(s[:m_len])
            conc_list.append(c[:m_len])
        if raw_list:
            return np.stack(raw_list, axis=0), np.stack(conc_list, axis=0)
        else:
            return np.empty((0, m_len)), np.empty((0, m_len))

    from tqdm import tqdm
    for i in tqdm(range(dce_shape[2]), desc="Processing slices"):
        wm_slice  = wm_mask_dce[:, :, i]
        cg_slice  = cg_mask_dce[:, :, i]
        sc_slice  = sc_mask_dce[:, :, i]
        bs_slice  = gm_brainstem_mask[:, :, i]
        cer_slice = gm_cerebellum_mask[:, :, i]
        wmc_slice = wm_cerebellum_mask[:, :, i]
        wcc_slice = wm_cc_mask[:, :, i]

        gm_slice = np.logical_or(cg_slice, sc_slice)
        if boundary:
            wmd = binary_dilation(wm_slice, 1)
            gmd = binary_dilation(gm_slice, 1)
            boundary_mask = np.logical_and(wmd, gmd)
            boundary_indices = np.argwhere(boundary_mask)
        else:
            boundary_mask = None
            boundary_indices = []

        m_len = len(aif)

        wm_idx  = np.argwhere(wm_slice)
        cg_idx  = np.argwhere(cg_slice)
        sc_idx  = np.argwhere(sc_slice)
        bs_idx  = np.argwhere(bs_slice)
        cer_idx = np.argwhere(cer_slice)
        wmc_idx = np.argwhere(wmc_slice)
        wcc_idx = np.argwhere(wcc_slice)

        wm_raw, wm_conc = process_ctcs(wm_idx, m_len)
        cg_raw, cg_conc = process_ctcs(cg_idx, m_len)
        sc_raw, sc_conc = process_ctcs(sc_idx, m_len)
        bs_raw, bs_conc = process_ctcs(bs_idx, m_len)
        cer_raw, cer_conc = process_ctcs(cer_idx, m_len)
        wmc_raw, wmc_conc = process_ctcs(wmc_idx, m_len)
        wcc_raw, wcc_conc = process_ctcs(wcc_idx, m_len)

        # Save per-slice (sliced method) tissue functions
        slice_id = f"slice_{i+1}"
        if wm_raw.size:
            save_tissue_function("sliced", "wm", slice_id, wm_raw, wm_conc, analysis_directory)
            global_data["wm"]["raw"].append(wm_raw)
            global_data["wm"]["conc"].append(wm_conc)
        if cg_raw.size:
            save_tissue_function("sliced", "cortical_gm", slice_id, cg_raw, cg_conc, analysis_directory)
            global_data["cortical_gm"]["raw"].append(cg_raw)
            global_data["cortical_gm"]["conc"].append(cg_conc)
        if sc_raw.size:
            save_tissue_function("sliced", "subcortical_gm", slice_id, sc_raw, sc_conc, analysis_directory)
            global_data["subcortical_gm"]["raw"].append(sc_raw)
            global_data["subcortical_gm"]["conc"].append(sc_conc)
        if bs_raw.size:
            save_tissue_function("sliced", "gm_brainstem", slice_id, bs_raw, bs_conc, analysis_directory)
            global_data["gm_brainstem"]["raw"].append(bs_raw)
            global_data["gm_brainstem"]["conc"].append(bs_conc)
        if cer_raw.size:
            save_tissue_function("sliced", "gm_cerebellum", slice_id, cer_raw, cer_conc, analysis_directory)
            global_data["gm_cerebellum"]["raw"].append(cer_raw)
            global_data["gm_cerebellum"]["conc"].append(cer_conc)
        if wmc_raw.size:
            save_tissue_function("sliced", "wm_cerebellum", slice_id, wmc_raw, wmc_conc, analysis_directory)
            global_data["wm_cerebellum"]["raw"].append(wmc_raw)
            global_data["wm_cerebellum"]["conc"].append(wmc_conc)
        if wcc_raw.size:
            save_tissue_function("sliced", "wm_cc", slice_id, wcc_raw, wcc_conc, analysis_directory)
            global_data["wm_cc"]["raw"].append(wcc_raw)
            global_data["wm_cc"]["conc"].append(wcc_conc)

        def median_or_empty(lst):
            if lst:
                return np.median(lst, axis=0)
            return np.array([])

        avg_wm  = median_or_empty([wm_conc])
        avg_cg  = median_or_empty([cg_conc])
        avg_sc  = median_or_empty([sc_conc])
        avg_bs  = median_or_empty([bs_conc])
        avg_cer = median_or_empty([cer_conc])
        avg_wmc = median_or_empty([wmc_conc])
        avg_wcc = median_or_empty([wcc_conc])
        if boundary and boundary_indices:
            avg_boundary = median_or_empty(process_ctcs(boundary_indices, m_len))
        else:
            avg_boundary = np.array([])

        ctc_dir = os.path.join(analysis_directory, 'CTC')
        os.makedirs(ctc_dir, exist_ok=True)
        np.save(os.path.join(ctc_dir, f'wm_tissue_slice_{i+1}_median.npy'),  avg_wm)
        np.save(os.path.join(ctc_dir, f'cortical_gm_tissue_slice_{i+1}_median.npy'), avg_cg)
        np.save(os.path.join(ctc_dir, f'subcortical_gm_tissue_slice_{i+1}_median.npy'), avg_sc)
        np.save(os.path.join(ctc_dir, f'gm_brainstem_slice_{i+1}_median.npy'), avg_bs)
        np.save(os.path.join(ctc_dir, f'gm_cerebellum_slice_{i+1}_median.npy'), avg_cer)
        np.save(os.path.join(ctc_dir, f'wm_cerebellum_slice_{i+1}_median.npy'), avg_wmc)
        np.save(os.path.join(ctc_dir, f'wm_cc_slice_{i+1}_median.npy'), avg_wcc)
        if boundary and avg_boundary.size>0:
            np.save(os.path.join(ctc_dir, f'boundary_slice_{i+1}_median.npy'), avg_boundary)

        m_len_effective = m_len  # use m_len as the common length
        def do_patlak(ctc):
            if ctc.size>0:
                return patlak_analysis_plotting(ctc[:m_len_effective], aif[:m_len_effective], time_points_s[:m_len_effective])
            else:
                return (np.nan, np.nan, np.nan,
                        np.array([]), np.array([]), np.array([], dtype=bool))

        Ki_wm, lam_wm, SDwm, xw, yw, incw = do_patlak(avg_wm)
        Ki_cg, lam_cg, SDcg, xc, yc, incc = do_patlak(avg_cg)
        Ki_sc, lam_sc, SDsc, xsc, ysc, incsc = do_patlak(avg_sc)
        Ki_bs, lam_bs, SDbs, xbs, ybs, incbs = do_patlak(avg_bs)
        Ki_cer, lam_cer, SDcer, xcer, ycer, inccer = do_patlak(avg_cer)
        Ki_wmc, lam_wmc_, SDwmc_, xwmc_, ywmc_, incwmc_ = do_patlak(avg_wmc)
        Ki_wcc, lam_wcc_, SDwcc_, xwcc_, ywcc_, incwcc_ = do_patlak(avg_wcc)
        if boundary and avg_boundary.size>0:
            Ki_bd, lam_bd, SDb, xbd, ybd, incbd = do_patlak(avg_boundary)
        else:
            Ki_bd, lam_bd, SDb = np.nan, np.nan, np.nan
            xbd, ybd, incbd = (np.array([]),)*3

        patlak_data = {
            "slice": i+1,
            "white_matter_median": {
                "Ki": Ki_wm, "SD_Ki": SDwm, "lambda": lam_wm,
                "voxels": int(np.sum(wm_slice))
            },
            "cortical_gray_matter_median": {
                "Ki": Ki_cg, "SD_Ki": SDcg, "lambda": lam_cg,
                "voxels": int(np.sum(cg_slice))
            },
            "subcortical_gray_matter_median": {
                "Ki": Ki_sc, "SD_Ki": SDsc, "lambda": lam_sc,
                "voxels": int(np.sum(sc_slice))
            },
            "gm_brainstem_median": {
                "Ki": Ki_bs, "SD_Ki": SDbs, "lambda": lam_bs,
                "voxels": int(np.sum(bs_slice))
            },
            "gm_cerebellum_median": {
                "Ki": Ki_cer, "SD_Ki": SDcer, "lambda": lam_cer,
                "voxels": int(np.sum(cer_slice))
            },
            "wm_cerebellum_median": {
                "Ki": Ki_wmc, "SD_Ki": SDwmc_, "lambda": lam_wmc_,
                "voxels": int(np.sum(wmc_slice))
            },
            "wm_cc_median": {
                "Ki": Ki_wcc, "SD_Ki": SDwcc_, "lambda": lam_wcc_,
                "voxels": int(np.sum(wcc_slice))
            }
        }
        if boundary:
            patlak_data["boundary_median"] = {
                "Ki": Ki_bd, "SD_Ki": SDb, "lambda": lam_bd,
                "voxels": int(np.sum(boundary_mask)) if boundary_mask is not None else 0
            }

        all_patlak_data.append(patlak_data)

        if compute_per_voxel_Ki or compute_per_voxel_CBF:
            seg_mask = np.logical_or.reduce([wm_slice, cg_slice, sc_slice])
            seg_indices = np.argwhere(seg_mask)
            if compute_per_voxel_Ki:
                ki_slice = np.full(seg_mask.shape, np.nan)
            if compute_per_voxel_CBF:
                cbf_slice = np.full(seg_mask.shape, np.nan)

            for (xx, yy) in seg_indices:
                voxel_tc = data_4d[xx, yy, i, :]
                T1_val   = T1_matrix[xx, yy, i]
                M0_val   = M0_matrix[xx, yy, i]
                if T1_val<=0 or M0_val<=0:
                    continue
                c_t_0 = compute_CTC_VFA(voxel_tc, M0_val, FA_GLOBAL, TR_GLOBAL, 1/T1_val, T1_val, beta=4.5)
                baseline_point = find_baseline_point_advanced(c_t_0)
                c_t_voxel = custom_shifter(c_t_0, baseline_point)
                if np.isnan(c_t_voxel).any() or np.all(c_t_voxel==0):
                    continue
                mm_vox = min(len(c_t_voxel), m_len_effective)
                c_t_voxel = c_t_voxel[:mm_vox]
                c_a_voxel = aif[:mm_vox]
                t_voxel   = time_points_s[:mm_vox]
                if compute_per_voxel_Ki:
                    Ki_vox, _, _, _, _, _ = patlak_analysis_plotting(c_t_voxel, c_a_voxel, t_voxel)
                    ki_slice[xx, yy] = Ki_vox
                if compute_per_voxel_CBF:
                    delta_t = t_voxel[1]-t_voxel[0]
                    A_mat   = construct_convolution_matrix(c_a_voxel, delta_t)
                    lambd   = 0.1
                    try:
                        R_est = tikhonov_regularization(A_mat, c_t_voxel, lambd)
                        cbf_slice[xx, yy] = R_est[0]
                    except np.linalg.LinAlgError:
                        continue

            if compute_per_voxel_Ki:
                Ki_per_voxel[:, :, i] = ki_slice
            if compute_per_voxel_CBF:
                CBF_per_voxel[:, :, i] = cbf_slice

            from matplotlib.gridspec import GridSpec
            image_directory_secondary = os.path.join(image_directory, "images")
            sp = os.path.join(image_directory_secondary,
                            f"Tissue_slice_{i+1}_median_patlak.png")
            os.makedirs(image_directory_secondary, exist_ok=True)
            plot_ctcs_and_patlak(
                dce_img_slice=data_4d[:, :, i, 0],
                wm_mask_dce=wm_slice,
                cortical_gm_mask_dce=cg_slice,
                subcortical_gm_mask_dce=sc_slice,
                avg_wm_ctc=avg_wm,
                avg_cortical_gm_ctc=avg_cg,
                avg_subcortical_gm_ctc=avg_sc,
                x_patlak_wm=xw, y_patlak_wm=yw, Ki_wm=Ki_wm, lambda_wm=lam_wm,
                x_patlak_cortical_gm=xc, y_patlak_cortical_gm=yc,
                Ki_cortical_gm=Ki_cg, lambda_cortical_gm=lam_cg,
                x_patlak_subcortical_gm=xsc, y_patlak_subcortical_gm=ysc,
                Ki_subcortical_gm=Ki_sc, lambda_subcortical_gm=lam_sc,
                slice_idx=i+1,
                save_path=sp,
                boundary_mask=boundary_mask,
                boundary_ctc=avg_boundary,
                x_patlak_boundary=xbd, y_patlak_boundary=ybd,
                Ki_boundary=Ki_bd, lambda_boundary=lam_bd,
                included_wm=incw, included_cortical_gm=incc,
                included_subcortical_gm=incsc,
                included_boundary=incbd,
                gm_brainstem_ctc=avg_bs,
                x_patlak_gm_brainstem=xbs, y_patlak_gm_brainstem=ybs,
                Ki_gm_brainstem=Ki_bs, lambda_gm_brainstem=lam_bs,
                included_gm_brainstem=incbs,
                gm_cerebellum_ctc=avg_cer,
                x_patlak_gm_cerebellum=xcer, y_patlak_gm_cerebellum=ycer,
                Ki_gm_cerebellum=Ki_cer, lambda_gm_cerebellum=lam_cer,
                included_gm_cerebellum=inccer,
                wm_cerebellum_ctc=avg_wmc,
                x_patlak_wm_cerebellum=xwmc_, y_patlak_wm_cerebellum=ywmc_,
                Ki_wm_cerebellum=Ki_wmc, lambda_wm_cerebellum=lam_wmc_,
                included_wm_cerebellum=incwmc_,
                wm_cc_ctc=avg_wcc,
                x_patlak_wm_cc=xwcc_, y_patlak_wm_cc=ywcc_,
                Ki_wm_cc=Ki_wcc, lambda_wm_cc=lam_wcc_,
                included_wm_cc=incwcc_,
                gm_brainstem_mask_dce=bs_slice,
                gm_cerebellum_mask_dce=cer_slice,
                wm_cerebellum_mask_dce=wmc_slice,
                wm_cc_mask_dce=wcc_slice
            )

        # After processing all slices, compute global tissue functions
    global_results = {}
    for tissue, data in global_data.items():
        # Check that we have at least one valid median curve for this tissue type
        if data["conc"]:
            # Stack all median concentration curves (each is 1D) along a new axis
            stacked_concs = np.stack(data["conc"], axis=0)
            # Compute the median curve along the slice axis (axis=0)
            median_conc_global = np.median(stacked_concs, axis=0)
            # Determine the common length using the minimum of the lengths
            mm = min(len(median_conc_global), len(aif), len(time_points_s))
            c_t_global = median_conc_global[:mm]
            c_a_global = aif[:mm]
            t_global   = time_points_s[:mm]
            # Compute the Patlak parameters using your existing function
            Ki, lam, SD_Ki, _, _, _ = patlak_analysis_plotting(c_t_global, c_a_global, t_global)
            global_results[tissue] = {"Ki": Ki, "SD_Ki": SD_Ki, "vp": lam}
        else:
            global_results[tissue] = {"Ki": np.nan, "SD_Ki": np.nan, "vp": np.nan}

    # Save the global results to a JSON file (or multiple files if desired)
    global_ki_path = os.path.join(analysis_directory, "global_ki.json")
    with open(global_ki_path, "w") as f:
        json.dump(global_results, f, indent=4)

    dce_affine = dce_img.affine
    if compute_per_voxel_Ki:
        ni_ki = nib.Nifti1Image(Ki_per_voxel, dce_affine)
        out_ki = os.path.join(analysis_directory, "perfusion", "Ki_per_voxel.nii.gz")
        nib.save(ni_ki, out_ki)
        debug_print("Ki per voxel saved to", out_ki)

    if compute_per_voxel_CBF:
        ni_cbf = nib.Nifti1Image(CBF_per_voxel, dce_affine)
        out_cbf = os.path.join(analysis_directory, "perfusion", "CBF_per_voxel.nii.gz")
        nib.save(ni_cbf, out_cbf)
        debug_print("CBF per voxel saved to", out_cbf)

    patlak_json = os.path.join(analysis_directory, "analysis_values_median.json")
    with open(patlak_json, 'w') as jf:
        json.dump(all_patlak_data, jf, indent=4)

def run_fastsurfer(fastsurfer_script, t1_path, outdir, sid, device):
    cmd = f"export PYTORCH_ENABLE_MPS_FALLBACK=1 && {fastsurfer_script} --seg_only --device {device} --t1 {t1_path} --vox_size 1 --sid {sid} --sd {outdir} --no_cereb"
    subprocess.run(cmd, shell=True, check=True)

def mri_convert(src, dst):
    if not os.path.exists(dst):
        subprocess.run(["mri_convert", src, dst], check=True)

def coregistration(seg_mgz_path, dce_path, anat_path, analysis_seg_dir):
    aseg_nii = seg_mgz_path.replace('.mgz', '.nii.gz')
    if not os.path.exists(aseg_nii):
        subprocess.run(['mri_convert', seg_mgz_path, aseg_nii], check=True)

    aseg_in_dce = aseg_nii.replace('.nii.gz','_in_DCE.nii.gz')
    if not os.path.exists(aseg_in_dce):
        subprocess.run(['flirt','-in',aseg_nii,'-ref',dce_path,
                        '-applyxfm','-usesqform','-interp','nearestneighbour',
                        '-out',aseg_in_dce], check=True)

    aseg_in_t1 = aseg_nii.replace('.nii.gz','_in_T1.nii.gz')
    if not os.path.exists(aseg_in_t1):
        subprocess.run(['flirt','-in',aseg_nii,'-ref',anat_path,
                        '-applyxfm','-usesqform','-interp','nearestneighbour',
                        '-out',aseg_in_t1], check=True)

    return create_masks(aseg_in_t1, analysis_seg_dir, force=False)

class Workflow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("p-brain (mini) - v1.0.0")
        try:
            from PIL import Image, ImageTk
            banner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'banner.png')
            if os.path.exists(banner_path):
                pil_image = Image.open(banner_path)
                rsz = pil_image.resize((pil_image.width//4, pil_image.height//4), Image.Resampling.LANCZOS)
                self.banner_img = ImageTk.PhotoImage(rsz)
                tk.Label(self, image=self.banner_img).pack(pady=10)
        except Exception as e:
            print("Banner image error:", e)

        self.fs   = "/Users/edt/FastSurfer/run_fastsurfer.sh"
        self.dev  = "mps"
        self.t1map = None
        self.m0map = None
        self.aif   = None
        self.dce   = None
        self.t1w   = None
        self.dce_shape_3d = None
        self.make_ui()

    def make_ui(self):
        f = tk.Frame(self)
        f.pack(padx=20, pady=10)

        self.btn_t1m0 = tk.Button(f, text="Load T1 & M0 (.mat)", command=self.load_t1m0)
        self.btn_t1m0.pack(side=tk.LEFT, padx=5)

        self.btn_aif = tk.Button(f, text="Load AIF (.mat)", command=self.load_aif)
        self.btn_aif.pack(side=tk.LEFT, padx=5)

        self.btn_dce = tk.Button(f, text="Load DCE (.nii.gz)", command=self.load_dce)
        self.btn_dce.pack(side=tk.LEFT, padx=5)

        self.btn_t1w = tk.Button(f, text="Load T1 (.nii)", command=self.load_t1w)
        self.btn_t1w.pack(side=tk.LEFT, padx=5)

        run_f = tk.Frame(self)
        run_f.pack(pady=10)
        self.btn_runall = tk.Button(run_f, text="Analyze", command=self.run_all, state=tk.DISABLED)
        self.btn_runall.pack()

    def load_t1m0(self):
        p = filedialog.askopenfilename(filetypes=[("MAT files","*.mat")])
        if p:
            import scipy.io
            try:
                mat = scipy.io.loadmat(p)
                if 't1_map' not in mat or 'm0_map' not in mat:
                    debug_print("MAT file missing 't1_map' or 'm0_map'.")
                    return
                self.t1map = mat['t1_map']
                self.m0map = mat['m0_map']
                debug_print("Loaded T1 & M0:", self.t1map.shape, self.m0map.shape)
                self.btn_t1m0.config(bg="pale green")
            except Exception as e:
                debug_print("Error loading T1/M0:", e)
        self.check_ready()

    def load_aif(self):
        p = filedialog.askopenfilename(filetypes=[("MAT files","*.mat")])
        if p:
            import scipy.io
            try:
                mat = scipy.io.loadmat(p)
                c_in = mat.get('c_input',None)
                if c_in is None:
                    debug_print("No 'c_input' in AIF file.")
                    return
                self.aif = c_in.flatten()
                debug_print(f"Loaded AIF of length {len(self.aif)}")
                self.btn_aif.config(bg="pale green")
            except Exception as e:
                debug_print("Error loading AIF:", e)
        self.check_ready()

    def load_dce(self):
        p = filedialog.askopenfilename()
        if p:
            self.dce = p
            try:
                dc = nib.load(p)
                arr = dc.get_fdata()
                if arr.ndim==4:
                    self.dce_shape_3d = arr.shape[:3]
                else:
                    self.dce_shape_3d = arr.shape
                debug_print("Loaded DCE:", p, arr.shape)
                self.btn_dce.config(bg="pale green")
            except Exception as e:
                debug_print("Error loading DCE:", e)
        self.check_ready()

    def load_t1w(self):
        p = filedialog.askopenfilename()
        if p:
            self.t1w = p
            debug_print("Loaded T1 structural:", p)
            self.btn_t1w.config(bg="pale green")
        self.check_ready()

    def check_ready(self):
        if self.t1map is not None and self.m0map is not None and self.aif is not None and self.dce and self.t1w:
            self.btn_runall.config(state=tk.NORMAL)

    def shape_fix_if_needed(self):
        if self.dce_shape_3d is None:
            return
        sh = self.t1map.shape
        d0, d1, d2 = self.dce_shape_3d
        if sh==(d1,d0,d2):
            self.t1map = self.t1map.transpose((1,0,2))
            self.m0map = self.m0map.transpose((1,0,2))
            debug_print("Transposed T1/M0 maps to match DCE shape")

    def run_all(self):
        try:
            debug_print("Starting workflow")
            self.shape_fix_if_needed()

            od = os.path.dirname(self.t1w)
            analysis_dir = os.path.join(od, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            seg_dir = os.path.join(od, "segmentation", "mri")
            seg_mgz = os.path.join(od, "segmentation","segmentation", "mri", "aparc.DKTatlas+aseg.deep.mgz")

            if not os.path.exists(seg_mgz):
                debug_print("No existing segmentation => run FastSurfer")
                run_fastsurfer(self.fs, self.t1w, os.path.dirname(seg_dir), "segmentation", self.dev)

            aseg = seg_mgz.replace(".mgz",".nii.gz")
            if not os.path.exists(aseg):
                debug_print("Convert mgz => nii.gz")
                mri_convert(seg_mgz, aseg)

            analysis_seg_dir = os.path.join(analysis_dir, "segmentation")
            os.makedirs(analysis_seg_dir, exist_ok=True)
            masks = coregistration(seg_mgz, self.dce, self.t1w, analysis_seg_dir)

            dce_img = nib.load(self.dce)
            dce_data = dce_img.get_fdata()
            vol_ct = dce_data.shape[-1]

            total_scan_duration = 5.0 * vol_ct
            time_pts = np.linspace(0, total_scan_duration, vol_ct)

            base = self.dce
            if base.endswith(".nii.gz"):
                base = base[:-7]
            else:
                base = os.path.splitext(base)[0]
            sidecar = base+".json"
            TR = 5.0
            FA_deg = 14.0
            if os.path.exists(sidecar):
                with open(sidecar,"r") as f:
                    hd = json.load(f)
                TR = hd.get("RepetitionTime", dce_img.header.get_zooms()[-1])
                FA_deg = hd.get("FlipAngle",14.0)
            else:
                TR = dce_img.header.get_zooms()[-1]
            FA = FA_deg*np.pi/180

            global FA_GLOBAL, TR_GLOBAL
            FA_GLOBAL = FA
            TR_GLOBAL = TR
            debug_print(f"Set global FA={FA_deg}°, TR={TR}sec")

            if len(dce_img.shape)==4:
                dce_vol = dce_img.get_fdata()[...,0]
                dce3 = nib.Nifti1Image(dce_vol, dce_img.affine, dce_img.header)
            else:
                dce3 = dce_img
            seg_in_dce = aseg.replace(".nii.gz","_in_DCE.nii.gz")
            if not os.path.exists(seg_in_dce):
                d_up = resample_from_to(nib.load(aseg), dce3, order=0)
                nib.save(d_up, seg_in_dce)
            else:
                d_up = nib.load(seg_in_dce)

            dce_masks = create_masks(seg_in_dce, analysis_seg_dir, force=True)

            plot_predictions_with_masks(
                nib.load(self.t1w).get_fdata(),
                nib.load(masks["wm"]).get_fdata(),
                nib.load(masks["cortical_gm"]).get_fdata(),
                nib.load(masks["subcortical_gm"]).get_fdata(),
                nib.load(masks["gm_brainstem"]).get_fdata(),
                nib.load(masks["gm_cerebellum"]).get_fdata(),
                nib.load(masks["wm_cerebellum"]).get_fdata(),
                nib.load(masks["wm_cc"]).get_fdata(),
                analysis_seg_dir
            )

            compute_Ki_from_atlas(
                atlas_path=seg_in_dce,
                data_4d=dce_data,
                T1_matrix=self.t1map,
                M0_matrix=self.m0map,
                time_points_s=time_pts,
                C_a_full=self.aif,
                affine=dce_img.affine,
                analysis_dir=analysis_dir,
                compute_CTC=compute_CTC_VFA,
                find_baseline_point_advanced=find_baseline_point_advanced,
                custom_shifter=custom_shifter,
                patlak_analysis_plotting=patlak_analysis_plotting
            )

            compute_and_plot_ctcs_median(
                data_4d=dce_data,
                wm_mask_t1=nib.load(masks["wm"]).get_fdata(),
                cg_mask_t1=nib.load(masks["cortical_gm"]).get_fdata(),
                sc_mask_t1=nib.load(masks["subcortical_gm"]).get_fdata(),
                wm_mask_dce=nib.load(dce_masks["wm"]).get_fdata(),
                cg_mask_dce=nib.load(dce_masks["cortical_gm"]).get_fdata(),
                sc_mask_dce=nib.load(dce_masks["subcortical_gm"]).get_fdata(),
                T1_matrix=self.t1map,
                M0_matrix=self.m0map,
                analysis_directory=analysis_dir,
                time_points_s=time_pts,
                image_directory=analysis_dir,
                dce_path=self.dce,
                aif=self.aif,
                boundary=False,
                compute_per_voxel_Ki=True,
                compute_per_voxel_CBF=True,
                gm_bs_mask_dce=dce_masks["gm_brainstem"],
                gm_cer_mask_dce=dce_masks["gm_cerebellum"],
                wm_cerebellum_mask_dce=dce_masks["wm_cerebellum"],
                wm_cc_mask_dce=dce_masks["wm_cc"]
            )

            print("Workflow completed successfully.")

        except Exception as e:
            debug_print("Error during workflow:", e)

def main():
    app = Workflow()
    app.mainloop()

if __name__=="__main__":
    main()
