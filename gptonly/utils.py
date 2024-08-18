import matplotlib.pyplot as plt
import torch
def plot_trp(
        trp,
        text=None,
        eos_token="[SEP]",
        figsize=(9,3),
        plot=False,
        max_idx_plot=-1,
    ):
    fig, ax = plt.subplots(1,1, figsize=figsize)

    x = torch.arange(len(trp))
    ax.bar(x, trp, width=.3, color='b', label='TRP')
    ax.set_xticks(x)

    ax.set_xticklabels(text, rotation=60)
    for idx, text in enumerate(text):
        if text == eos_token:
            ax.vlines(idx, ymin=0, ymax=1, linestyle="dashed", color="r")
        if idx == max_idx_plot:
            ax.vlines(idx, ymin=0, ymax=1, linestyle="solid", color="g")

    ax.set_ylim([0,1])
    ax.legend()
    fig.tight_layout()

    if plot:
        plt.pause(0.01)
    return fig, ax