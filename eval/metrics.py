import numpy as np
from scipy.stats import spearmanr
from attack.payload import PARENT, descendants, g_dose


def mpjpe(pred, gt):
    return np.linalg.norm(pred - gt, axis=-1).mean(-1)

def _procrustes(X, Y):
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY
    nX = np.linalg.norm(X0); nY = np.linalg.norm(Y0)
    X0 /= (nX + 1e-12); Y0 /= (nY + 1e-12)
    U, s, Vt = np.linalg.svd(X0.T @ Y0)
    V = Vt.T; d = np.sign(np.linalg.det(V @ U.T))
    V[:, -1] *= d; s[-1] *= d
    T = V @ U.T; b = s.sum() * nX / (nY + 1e-12)
    return b * (Y @ T) + (muX - b * (muY @ T))

def pa_mpjpe(pred, gt):
    out = np.zeros(len(pred))
    for i in range(len(pred)):
        out[i] = np.linalg.norm(_procrustes(gt[i], pred[i]) - gt[i], axis=-1).mean()
    return out

def pck(pred, gt, thr=0.5, ref=(6, 4)):
    """Fraction of joints within thr*scale; scale = ||gt[ref0]-gt[ref1]||."""
    scale = np.linalg.norm(gt[:, ref[0]] - gt[:, ref[1]], axis=-1) + 1e-9
    d = np.linalg.norm(pred - gt, axis=-1) / scale[:, None]
    return float((d <= thr).mean())

def subchain_displacement(pred_at_dose, pred_clean, pivot):
    js = descendants(pivot)
    return float(np.linalg.norm(pred_at_dose[:, js] - pred_clean[:, js], axis=-1).mean())


def nontarget_preservation(pred_at_dose, pred_clean, pivot, n_joints=14):
    js = [j for j in range(n_joints) if j not in descendants(pivot)]
    return float(np.linalg.norm(pred_at_dose[:, js] - pred_clean[:, js], axis=-1).mean())


def dose_response_analysis(doses, displacements):
    doses = np.asarray(doses, float); disp = np.asarray(displacements, float)
    rho, p = spearmanr(doses, disp)
    # R^2 of a monotone (linear) fit through origin-ish
    A = np.vstack([doses, np.ones_like(doses)]).T
    coef, *_ = np.linalg.lstsq(A, disp, rcond=None)
    fit = A @ coef
    ss_res = ((disp - fit) ** 2).sum(); ss_tot = ((disp - disp.mean()) ** 2).sum() + 1e-12
    r2_ramp = 1 - ss_res / ss_tot

    best_step_res = np.inf
    for k in range(1, len(doses)):
        lo, hi = disp[:k].mean(), disp[k:].mean()
        res = ((disp[:k] - lo) ** 2).sum() + ((disp[k:] - hi) ** 2).sum()
        best_step_res = min(best_step_res, res)
    r2_step = 1 - best_step_res / ss_tot
    return {'spearman': float(rho), 'spearman_p': float(p),
            'r2_ramp': float(r2_ramp), 'r2_step': float(r2_step),
            'ramp_minus_step': float(r2_ramp - r2_step),
            'slope': float(coef[0])}

def bone_lengths(pose, pivot, n_joints=14):
    js = [pivot] + descendants(pivot)
    L = []
    for j in js:
        p = PARENT[j]
        if p is not None and (p in js):
            L.append(np.linalg.norm(pose[..., j, :] - pose[..., p, :], axis=-1))
    return np.stack(L, -1) if L else np.zeros(pose.shape[:-2] + (0,))


def plausibility_error(pred, ref_pose, pivot):
    lp = bone_lengths(pred, pivot); lr = bone_lengths(ref_pose, pivot)
    if lp.shape[-1] == 0:
        return 0.0
    return float((np.abs(lp - lr) / (lr + 1e-9)).mean())


def target_mpjpe(pred, target, pivot):
    js = descendants(pivot)
    return np.linalg.norm(pred[:, js] - target[:, js], axis=-1).mean(-1)

def clean_floor(pred_clean, true_clean, pivot):
    js = descendants(pivot)
    target_floor = np.linalg.norm(pred_clean[:, js] - true_clean[:, js], axis=-1).mean()
    other = [j for j in range(pred_clean.shape[1]) if j not in js]
    nontarget_floor = np.linalg.norm(pred_clean[:, other] - true_clean[:, other], axis=-1).mean()
    return float(target_floor), float(nontarget_floor)

def attack_metrics(pred_at_dose, target_at_dose, pred_clean, true_clean, pivot,
                   k_attack=1.5, k_clean=1.5, tau_plaus=0.20):
    tfloor, nfloor = clean_floor(pred_clean, true_clean, pivot)
    tmpjpe = target_mpjpe(pred_at_dose, target_at_dose, pivot)        # (N,)
    js = descendants(pivot)
    other = [j for j in range(pred_clean.shape[1]) if j not in js]
    nondrift = np.linalg.norm(pred_at_dose[:, other] - pred_clean[:, other], axis=-1).mean(-1)
    landed = tmpjpe < k_attack * tfloor
    preserved = nondrift < k_clean * nfloor
    plaus = plausibility_error(pred_at_dose, true_clean, pivot) < tau_plaus
    asr = float((landed & preserved & (plaus if np.ndim(plaus) else np.full(len(landed), plaus))).mean())
    return {'tmpjpe_mean': float(tmpjpe.mean()),
            'clean_target_floor': tfloor,
            'frac_landed': float(landed.mean()),
            'frac_preserved': float(preserved.mean()),
            'plausible': bool(plaus),
            'asr': asr,
            'tau_attack': float(k_attack * tfloor)}