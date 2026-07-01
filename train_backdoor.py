import os, argparse, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.feeder import PersonInWiFi3D
from attack.trigger import MicroDopplerTrigger, velocity_profiles_from_skeleton
from attack.poison import PoisonedDataset, collate
from models.factory import build_model
from eval import metrics as M

def build_trigger(cfg):
    t = MicroDopplerTrigger(aoa_spread=cfg.get('aoa_spread', 0.6), seed=cfg.get('seed', 0))
    vel, pos, movers = velocity_profiles_from_skeleton(cfg['action_npy'],
                                                       top_k=cfg.get('top_k', 6))
    t.build(vel, pos)
    return t


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, trues, targets, doses = [], [], [], []
    for b in loader:
        out, _ = model(b['csi'].to(device))
        preds.append(out.cpu().numpy())
        trues.append(b['pose'].numpy())
        if 'target' in b: targets.append(b['target'].numpy())
        if 'dose' in b: doses.append(b['dose'].numpy())
    P = np.concatenate(preds)[:, 0]                       # (N,14,3) one person
    Tr = np.concatenate(trues)[:, 0]
    Tg = np.concatenate(targets)[:, 0] if targets else None
    Do = np.concatenate(doses) if doses else None
    return P, Tr, Tg, Do


def evaluate(model, base_test, trig, cfg, device):
    pivot = cfg['pivot']
    dl = lambda ds: DataLoader(ds, batch_size=cfg['batch_size'], collate_fn=collate)

    # clean
    clean_ds = PoisonedDataset(base_test, trig, mode='clean', pivot=pivot)
    Pc, Tc, _, _ = _predict(model, dl(clean_ds), device)
    res = {'clean_mpjpe': float(M.mpjpe(Pc, Tc).mean()),
           'clean_pampjpe': float(M.pa_mpjpe(Pc, Tc).mean()),
           'clean_pck@0.5': M.pck(Pc, Tc, 0.5)}

    # dose-response: per-dose displacement, t-MPJPE (vs attacker target), localization, plausibility
    grid = cfg.get('dose_grid', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    disp, tmp, loc, plaus = [], [], [], []
    for d in grid:
        tds = PoisonedDataset(base_test, trig, mode='trigger@dose',
                              pivot=pivot, fixed_dose=d, eps=cfg['eps'],
                              theta_max_deg=cfg['theta_max_deg'],
                              dose_mode=cfg['dose_mode'])
        Pd, Td, Tg, _ = _predict(model, dl(tds), device)     # Td=true pose, Tg=attacker target
        disp.append(M.subchain_displacement(Pd, Pc, pivot))
        tmp.append(float(M.target_mpjpe(Pd, Tg, pivot).mean()))
        loc.append(M.nontarget_preservation(Pd, Pc, pivot))
        plaus.append(M.plausibility_error(Pd, Td, pivot))
    tfloor, nfloor = M.clean_floor(Pc, Tc, pivot)
    res['dose_grid'] = list(map(float, grid))
    res['displacement'] = list(map(float, disp))
    res['tmpjpe'] = list(map(float, tmp))                    # t-MPJPE per dose (vs target)
    res['nontarget_mpjpe'] = list(map(float, loc))
    res['plausibility'] = list(map(float, plaus))
    res['clean_target_floor'] = tfloor                        # model noise floor on the limb
    res['dose_response'] = M.dose_response_analysis(grid, disp)

    # calibrated conjunctive ASR at the reference (max) dose
    dref = grid[-1]
    tds = PoisonedDataset(base_test, trig, mode='trigger@dose', pivot=pivot,
                          fixed_dose=dref, eps=cfg['eps'],
                          theta_max_deg=cfg['theta_max_deg'], dose_mode=cfg['dose_mode'])
    Pd, Td, Tg, _ = _predict(model, dl(tds), device)
    res['asr@ref'] = M.attack_metrics(Pd, Tg, Pc, Tc, pivot,
                                      k_attack=cfg.get('k_attack', 1.5),
                                      k_clean=cfg.get('k_clean', 1.5),
                                      tau_plaus=cfg.get('tau_plaus', 0.20))
    return res


def train(cfg):
    device = cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.get('seed', 0)); np.random.seed(cfg.get('seed', 0))

    base_train = PersonInWiFi3D('training', cfg['dataset_root'], cfg['experiment_name'])
    base_test = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])
    trig = build_trigger(cfg)

    pois = PoisonedDataset(base_train, trig, mode='train', rho=cfg['rho'],
                           dose_min=cfg['dose_min'], dose_max=cfg['dose_max'],
                           eps=cfg['eps'], pivot=cfg['pivot'],
                           theta_max_deg=cfg['theta_max_deg'], dose_mode=cfg['dose_mode'],
                           seed=cfg.get('seed', 0), select=cfg.get('poison_select', 'uniform'))
    loader = DataLoader(pois, batch_size=cfg['batch_size'], shuffle=True,
                        collate_fn=collate, drop_last=False)

    model = build_model(cfg['model'], subcarrier_num=180,
                        pretrained=cfg.get('pretrained', False)).to(device)
    if (cfg.get('data_parallel') and str(device).startswith('cuda')
            and torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])

    for epoch in range(cfg['epochs']):
        model.train(); losses = []
        for b in loader:
            csi, pose = b['csi'].to(device), b['pose'].to(device)
            pred, _ = model(csi)
            loss = torch.mean(torch.norm(pred - pose, dim=-1))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f'epoch {epoch}: train_loss={np.mean(losses):.4f}')

    res = evaluate(model, base_test, trig, cfg, device)
    res['n_poison'] = int(pois.n_poison)
    res['n_total'] = int(pois.n_total)
    res['poison_select'] = cfg.get('poison_select', 'uniform')
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/attack.yaml')
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    _, res = train(cfg)
    print('\n==== RESULTS ===='); import json; print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()