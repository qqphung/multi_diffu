import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_spatial_maps(spatial_map_list, map_names, save_dir, seed, tokens_vis=None):
    ########
    # spa_map: [B x 4 x H x W]
    for i, (spa_map, map_name) in enumerate(zip(spatial_map_list, map_names)):
        n_obj = len(spa_map)
        plt.figure()
        plt.clf()

        fig, axs = plt.subplots(ncols=n_obj+1, gridspec_kw=dict(width_ratios=[1 for _ in range(n_obj)]+[0.1]))

        fig.set_figheight(3)
        fig.set_figwidth(3*n_obj+0.1)

        cmap = plt.get_cmap('YlOrRd')

        vmax = 0
        vmin = 1
        for tid in range(n_obj):
            spatial_map_cur = spa_map[tid]
            spatial_map_cur = spatial_map_cur[0, 0].cpu()
            vmax = max(vmax, float(spatial_map_cur.max()))
            vmin = min(vmin, float(spatial_map_cur.min()))

        for tid in range(n_obj):
            spatial_map_cur = spa_map[tid]
            spatial_map_cur = spatial_map_cur[0, 0].cpu()
            sns.heatmap(
                spatial_map_cur, annot=False, cbar=False, ax=axs[tid],
                cmap=cmap, vmin=vmin, vmax=vmax
            )
            if tokens_vis is not None:
                axs[tid].set_xlabel(tokens_vis[tid])


        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=axs[-1])

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'average_seed%d_spa%d_%s.png' % (seed, i, map_name)), dpi=100)
        plt.close('all')