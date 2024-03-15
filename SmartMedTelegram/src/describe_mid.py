import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIG_WIDTH = 12
FIG_HEIGHT = 10

matplotlib.use("agg")


def display_scatter(
    dataframe: pd.DataFrame, axis_x: str, axis_y: str, title_x=None, title_y=None
):
    if not title_x:
        title_x = axis_x
    if not title_y:
        title_y = axis_y
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = sns.scatterplot(data=dataframe, x=axis_x, y=axis_y)
    ax.set_title("Зависимость " + title_y + " от " + title_x)
    plt.show()


def display_correlation_matrix(
    dataframe: pd.DataFrame,
    sharey=False,
    annot=True,
    Pearson=True,
    Spearman=True,
    title="",
    cmap=sns.color_palette("viridis", as_cmap=True),
    fmt=".2f",
):
    if not Pearson and not Spearman:
        return
    dataframe.drop(columns=dataframe.columns[0], axis=1, inplace=True)
    ncols = Spearman + Pearson

    f, axes = plt.subplots(
        nrows=1, ncols=ncols, sharey=sharey, figsize=(FIG_WIDTH * ncols, FIG_HEIGHT)
    )

    pltP = None
    pltS = None
    if Pearson:
        if Spearman:
            pltP = sns.heatmap(
                dataframe.corr(method="pearson"),
                annot=annot,
                vmin=-1,
                vmax=1,
                cmap=cmap,
                center=0,
                fmt=fmt,
                ax=axes[0],
            )
        else:
            pltP = sns.heatmap(
                dataframe.corr(method="pearson"),
                annot=annot,
                vmin=-1,
                vmax=1,
                cmap=cmap,
                center=0,
                fmt=fmt,
            )
        pltP.xaxis.tick_top()
        pltP.set_title("Пирсон. " + title)
        pltP.set_xticklabels(pltP.get_xticklabels(), rotation=30)
    if Spearman:
        if Pearson:
            pltS = sns.heatmap(
                dataframe.corr(method="spearman"),
                annot=annot,
                vmin=-1,
                vmax=1,
                cmap=cmap,
                center=0,
                fmt=fmt,
                ax=axes[1],
            )
        else:
            pltS = sns.heatmap(
                dataframe.corr(method="spearman"),
                annot=annot,
                vmin=-1,
                vmax=1,
                cmap=cmap,
                center=0,
                fmt=fmt,
            )
        pltS.xaxis.tick_top()
        pltS.set_title("Спирмен. " + title)
        pltS.set_xticklabels(pltS.get_xticklabels(), rotation=30)

    plt.savefig("result.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()


def make_df_plot(frame: pd.DataFrame):
    frame.plot()
    plt.savefig("result_plot.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()


def sheet_to_dataframe_spec(
    sheet, start_letter: str, start_row: str, finish_letter: str, finish_row: str
):
    x = []
    x = sheet.get(start_letter + start_row + ":" + finish_letter + finish_row)
    frame = pd.DataFrame(columns=["nm_id", "grade"])
    for l in x:
        for z in l:
            if l.index(z) == 1 or l.index(z) == 14:
                pass
            else:
                i = 0
                start = 0
                try:
                    if z[0] == "p":
                        start = 1
                    for c in range(len(z)):
                        if (z[c] == "+") or (z[c] == "-"):
                            # frame.append([z[start:c-1],z[c]])
                            frame.loc[len(frame.index)] = [z[start : c - 1], z[c]]
                            break
                except:
                    pass
    frame["grade"].replace({"+": True, "-": False}, inplace=True)
    return frame
