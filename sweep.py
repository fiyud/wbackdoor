import os, sys, json, csv, copy, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_backdoor import train

MM = 1000.0 # mm

def run_grid(base_cfg, thetas, rhos, epochs=None):
    rows = []
    total = len(thetas) * len(rhos)
    i = 0
    for rho in rhos:
        for th in thetas:
            i += 1
            cfg = copy.deepcopy(base_cfg)
            cfg['theta_max_deg'] = float(th)
            cfg['rho'] = float(rho)
            if epochs is not None:
                cfg['epochs'] = int(epochs)
            print(f"\n[{i}/{total}] theta_max={th}  rho={rho}  epochs={cfg['epochs']}")
            _, res = train(cfg)
            dr = res['dose_response']
            rows.append({
                'theta_max_deg': th, 'rho': rho,
                'clean_mpjpe_mm': res['clean_mpjpe'] * MM,
                'displacement_mm': res['displacement'][-1] * MM,
                'nontarget_mpjpe_mm': res['nontarget_mpjpe'][-1] * MM,
                'plausibility': res['plausibility'][-1],
                'spearman': dr['spearman'],
                'ramp_minus_step': dr['ramp_minus_step'],
                'frac_moved': res['asr@ref']['frac_moved'],
            })
    return rows


def save_tables(rows, outdir):
    with open(os.path.join(outdir, 'sweep_results.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(outdir, 'sweep_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def fig_tradeoff(rows, outdir):
    rhos = sorted({r['rho'] for r in rows})
    plt.figure(figsize=(6, 5))
    for rho in rhos:
        pts = sorted([r for r in rows if r['rho'] == rho], key=lambda r: r['displacement_mm'])
        x = [p['displacement_mm'] for p in pts]
        y = [p['nontarget_mpjpe_mm'] for p in pts]
        plt.plot(x, y, 'o-', label=f'rho={rho}')
        for p in pts:
            plt.annotate(f"{int(p['theta_max_deg'])}", (p['displacement_mm'], p['nontarget_mpjpe_mm']),
                         fontsize=7, xytext=(3, 3), textcoords='offset points')

    clean = np.mean([r['clean_mpjpe_mm'] for r in rows])
    plt.axhline(clean, ls='--', c='gray', lw=1)
    plt.text(plt.xlim()[1], clean, ' clean MPJPE', va='bottom', ha='right', fontsize=8, c='gray')
    plt.xlabel('target-limb displacement (mm)   [bigger = stronger payload]')
    plt.ylabel('non-target leak MPJPE (mm)   [smaller = better localization]')
    plt.title('Localization vs payload tradeoff (label = theta_max deg)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_tradeoff.png'), dpi=150); plt.close()


def fig_heatmaps(rows, outdir):
    thetas = sorted({r['theta_max_deg'] for r in rows})
    rhos = sorted({r['rho'] for r in rows})
    def grid(key):
        M = np.full((len(rhos), len(thetas)), np.nan)
        for r in rows:
            M[rhos.index(r['rho']), thetas.index(r['theta_max_deg'])] = r[key]
        return M
    panels = [('nontarget_mpjpe_mm', 'leak (mm) lower=better'),
              ('displacement_mm', 'displacement (mm)'),
              ('ramp_minus_step', 'ramp - step R2 (>0 = analog)')]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title) in zip(axs, panels):
        M = grid(key)
        im = ax.imshow(M, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(thetas))); ax.set_xticklabels([int(t) for t in thetas])
        ax.set_yticks(range(len(rhos))); ax.set_yticklabels(rhos)
        ax.set_xlabel('theta_max (deg)'); ax.set_ylabel('rho'); ax.set_title(title)
        for a in range(len(rhos)):
            for b in range(len(thetas)):
                ax.text(b, a, f'{M[a, b]:.2f}', ha='center', va='center', color='w', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'fig_heatmaps.png'), dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/attack.yaml')
    ap.add_argument('--theta', type=float, nargs='+', default=[20, 30, 40, 50, 60])
    ap.add_argument('--rho', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1])
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--outdir', default='sweep_out')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    with open(a.config) as f:
        base = yaml.safe_load(f)
    rows = run_grid(base, a.theta, a.rho, a.epochs)
    save_tables(rows, a.outdir)
    fig_tradeoff(rows, a.outdir)
    fig_heatmaps(rows, a.outdir)
    print(f"\nwrote results + figures to {a.outdir}/")


if __name__ == '__main__':
    main()