# %% import modules
import json
from datetime import date
from pathlib import Path

import hlsvdpro
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# important note about sklearn linear reg
#
# %% parameters
exten_path = "clean_code/vanila/"
analyze_name = "exp4"
json_file_path = exten_path + "runs/exp4.json"
with open(json_file_path, "r") as j:
    content = json.loads(j.read())
runs = content["runs"]
run = runs[0]
saving_dir = exten_path + "analyze/" + analyze_name + "/"
Path(saving_dir).mkdir(parents=True, exist_ok=True)
save = False
t_step = run["t_step"]
trnfreq = run["trnfreq"]
Crfr = trnfreq * (4.7 - 3.027)
lentgh = run["sigLen"]
_error = ["Error(" + i + ")" for i in run["met_name"]]
_pred = ["Predicted(" + i + ")" for i in run["met_name"]]
_true = ["True(" + i + ")" for i in run["met_name"]]
met_name = run["met_name"]
cmap = "Reds"
t = np.expand_dims(np.arange(lentgh) * t_step, 1)
selected_met = ["Cr", "GPC", "Glu", "Ins", "NAA", "NAAG", "PCho", "PCr", "Tau"]
models = ["DQ-nMM", "DQ-pMM", "DQ-rpMM"]
selected_method = ["QUEST", "QUEST Subtract", "FITAID(Frequency)"] + models
sns.set_style("whitegrid")
sns.set_palette("muted")


# %% functions
def plotppm(sig, ppm1, ppm2, rev, linewidth=0.3, linestyle="-"):
    p1 = int(ppm2p(ppm1, len(sig)))
    p2 = int(ppm2p(ppm2, len(sig)))
    n = p2 - p1
    x = np.linspace(int(ppm1), int(ppm2), abs(n))
    sig = np.squeeze(sig)
    df = pd.DataFrame({"Real Signal (a.u.)": sig[p2:p1].real})
    df["Frequency(ppm)"] = np.flip(x)
    g = sns.lineplot(
        x="Frequency(ppm)",
        y="Real Signal (a.u.)",
        data=df,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    if rev:
        plt.gca().invert_xaxis()
    return g


def watrem(data, dt, n):
    npts = len(data)
    dwell = dt / 0.001
    nsv_sought = n
    result = hlsvdpro.hlsvd(data, nsv_sought, dwell)
    nsv_found, singvals, freq, damp, ampl, phas = result
    idx = np.where(
        (result[2] < (0.001 * (Crfr + 50))) & (result[2] > (0.001 * (Crfr - 50)))
    )
    result = (
        len(idx),
        result[1],
        result[2][idx],
        result[3][idx],
        result[4][idx],
        result[5][idx],
    )
    fid = hlsvdpro.create_hlsvd_fids(
        result, npts, dwell, sum_results=True, convert=False
    )
    return fid, result


def cal_snr_lw(signal):
    av_f = fft.fftshift(fft.fft((signal)))
    plotppm(av_f, 0, 5, False)
    lsr, rslt = watrem(signal, t_step, 8)
    lsr = fft.fftshift(fft.fft(((lsr))))
    plotppm(lsr, 0, 5, True)
    noise = np.std(signal.real[:-128])
    snr = rslt[4] / noise
    # plt.title('Linewidth: '+ str(-1000/(np.pi*res.x[2])) + "Hz" + "SNR: " + str(snr))
    plt.title(
        "Linewidth: " + str(-1 * 1000 / (np.pi * rslt[3])) + "Hz" + "SNR: " + str(snr)
    )
    return snr, -1 * 1000 / (np.pi * rslt[3])


def cal_snr(data, endpoints=128, offset=0):
    return np.abs(data[0, :]) / np.std(
        data.real[-(offset + endpoints) : -(offset + 1), :], axis=0
    )


def cal_snrf(data_f, endpoints=128, offset=0):
    return np.max(np.abs(data_f), 0) / (
        np.std(data_f.real[offset : endpoints + offset, :], axis=0)
    )


def abs_err(a, b):
    return np.abs(a - b)


def err(a, b):
    return a - b


def savefig(path=str(date.today()), tight=False):
    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(saving_dir + path + ".svg", format="svg")
        plt.savefig(saving_dir + path + " .png", format="png", dpi=800)
    plt.show()


def ppm2p(r, len):
    r = 4.7 - r
    return int(((trnfreq * r) / (1 / (t_step * len))) + len / 2)


def calib_plot(ampl_t, y_out, yerr=None, cmap=None, ident_lin=True):
    if cmap is None:
        ax = plt.scatter(x=ampl_t, y=y_out)
    else:
        ax = plt.scatter(x=ampl_t, y=y_out, c=yerr, cmap="Spectral")
        plt.set_cmap(cmap)
        cb = plt.colorbar()
        cb.outline.set_visible(False)
    if ident_lin:
        plot_iden(ax)
    sns.despine()
    ax.axes.set_xlabel("True")
    ax.axes.set_ylabel("Predicted")
    return ax


def plot_iden(ax):
    ax.axes.yaxis.set_ticks_position("left")
    ax.axes.xaxis.set_ticks_position("bottom")
    x0, x1 = ax.axes.get_xlim()
    y0, y1 = ax.axes.get_ylim()
    lims = [min(x0, x1), min(y0, y1)]
    ax.axes.axline((lims[0], lims[1]), slope=1, ls="--", zorder=0, color="silver")


# %% load data
QST_MM = pd.read_csv(exten_path + "analyze/rslt/QST_MM.csv")
QY_MM = pd.read_csv(exten_path + "analyze/rslt/QY_MM.csv")
QY_Sub = pd.read_csv(exten_path + "analyze/rslt/QY_Sub.csv")

FA_T = pd.read_csv(exten_path + "analyze/rslt/FA_time.csv")
FA_F = pd.read_csv(exten_path + "analyze/rslt/FA_freq.csv")

# %%
id = "test_" + str(run["test_params"][6]) + "_" + str(run["test_params"][5]) + "_"
id = "test/" + id + "/"
data = np.load(
    exten_path + run["sim_order"][1] + "test_" + str(run["test_params"][2:]) + ".npz"
)
y_test, ampl_t, shift_t, alpha_t, ph_t, snrs_t = [data[x] for x in data]
dir_list = []
for run in runs:
    dir_list.append(
        exten_path + run["parent_root"] + run["child_root"] + run["version"] + id
    )
encs_list = []
rslt_list = []
for dir in dir_list:
    try:
        encs_list.append(np.load(dir + "rslt.npz"))
        rslt_list.append(pd.read_csv(dir + "rslt.csv", index_col=[0]))
    except:
        print("cannot reach to the result at " + dir)

# %%
met = ["NAA", "Cr"]
df_uncer = pd.DataFrame()
errors_averaged_all = pd.DataFrame()
errors_corr_dl = pd.DataFrame(
    columns=met_name, index=["damping", "frequency", "Phase", "SNR"]
)
for idy, encs in enumerate(encs_list):
    errors_averaged = pd.DataFrame(
        columns=["$R_2$", "MAE", "MSE", "MAPE", "r2", "intercept", "coef"],
        index=met_name,
    )
    y_out, fr, damp, ph, decs, encs = [encs[x] for x in encs]
    for idx, name in enumerate(met_name):
        model = LinearRegression(fit_intercept=True).fit(
            ampl_t[:, idx].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1))
        )
        errors_averaged.iloc[idx] = [
            r2_score(ampl_t[:, idx], y_out[:, idx]),
            mean_absolute_error(ampl_t[:, idx], y_out[:, idx]),
            mean_squared_error(ampl_t[:, idx], y_out[:, idx]),
            mean_absolute_percentage_error(ampl_t[:, idx], y_out[:, idx]) * 100,
            model.score(
                ampl_t[:, idx].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1))
            ),
            model.intercept_[0],
            model.coef_[0][0],
        ]

    # df_uncer_t['Method'] = models[idy]
    # df_uncer=df_uncer.append(df_uncer_t)
    errors_averaged["Method"] = models[idy]
    errors_averaged_all = errors_averaged_all.append(errors_averaged)

for label, method in zip(
    ["QUEST", "QUEST Subtract", "FITAID(Time)", "FITAID(Frequency)"],
    [QY_MM, QY_Sub, FA_T, FA_F],
):
    errors_averaged = pd.DataFrame(
        columns=["$R_2$", "MAE", "MSE", "MAPE", "r2", "intercept", "coef"],
        index=met_name,
    )
    for idx, name in enumerate(met_name):
        model = LinearRegression(fit_intercept=True).fit(
            ampl_t[:, idx].reshape((-1, 1)), method[name].values.reshape((-1, 1))
        )
        errors_averaged.iloc[idx] = [
            r2_score(ampl_t[:, idx], method[name].values),
            mean_absolute_error(ampl_t[:, idx], method[name].values),
            mean_squared_error(ampl_t[:, idx], method[name].values),
            mean_absolute_percentage_error(ampl_t[:, idx], method[name].values) * 100,
            model.score(
                ampl_t[:, idx].reshape((-1, 1)), method[name].values.reshape((-1, 1))
            ),
            model.intercept_[0],
            model.coef_[0][0],
        ]
        if name in met:
            # df_uncer_t = pd.DataFrame()
            df_uncer["CRLB " + name + "" + label] = method[name + "_sd"]

    # df_uncer_t['Method'] = label
    # df_uncer=df_uncer.append(df_uncer_t)
    errors_averaged["Method"] = label
    errors_averaged_all = errors_averaged_all.append(errors_averaged)

errors_averaged_all.to_csv(saving_dir + "_errors_averaged_all.csv")
compar_mean = errors_averaged_all.groupby("Method").mean(0)
sns.scatterplot(x="$R_2$", y="MAPE", data=compar_mean, hue=compar_mean.index)
savefig("MAPEvsR2_mean")
compar_mean.to_csv(saving_dir + "MAPEvsR.csv")
dfm = errors_averaged_all.reset_index(level=0)
dfm["$1-R_2$"] = 1 - dfm["$R_2$"]
# sns.scatterplot(x='$1-R_2$',y='MSE',data=dfm,hue='index',style='Method')
# sns.scatterplot(x='$R_2$',y='MSE',data=dfm.isin(["Single MM"]),hue='Method')
# sns.scatterplot(x='$R_2$',y='MSE',data=dfm[~dfm.isin(["QUEST 3"])],hue='Method')
ax = sns.relplot(
    x="$1-R_2$",
    y="MAPE",
    data=dfm[dfm["index"].isin(selected_met) & dfm["Method"].isin(selected_method)],
    style="index",
    hue="Method",
    alpha=0.9,
)
ax.set(xscale="log")
ax.set(yscale="log")
savefig("MAPEvsR2_met1")
ax = sns.relplot(
    x="$1-R_2$",
    y="MAPE",
    data=dfm[~dfm["index"].isin(selected_met) & dfm["Method"].isin(selected_method)],
    style="index",
    hue="Method",
    alpha=0.9,
)
ax.set(xscale="log")
ax.set(yscale="log")
savefig("MAPEvsR2_met2")
# %%
tr = 16
df_met = pd.DataFrame()
for idx, rslt in enumerate(rslt_list):
    df_met_t = pd.DataFrame(columns=met_name)
    df_met_t[met_name] = rslt.loc[rslt["type"] == "Predicted"][met_name].loc[0:tr]
    true_value = rslt.loc[rslt["type"] == "True"][met_name].loc[0:tr]
    df_met_t[["True " + i for i in met_name]] = true_value
    df_met_t["Model"] = models[idx]
    df_met = df_met.append(df_met_t, ignore_index=True)

# true_value= rslt_list[0].loc[rslt['type'] == 'True'][met]
for label, rslt in zip(
    ["QUEST", "QUEST Subtract", "FITAID(Frequency)"], [QY_MM, QY_Sub, FA_F]
):
    df_met_t = pd.DataFrame(columns=met_name)
    df_met_t[met_name] = rslt[met_name].loc[0:tr]
    df_met_t[["True " + i for i in met_name]] = true_value
    df_met_t["Model"] = label
    df_met = df_met.append(df_met_t, ignore_index=True)


for met in met_name:
    ax = sns.lmplot(
        x="True " + met,
        y=met,
        data=df_met,
        hue="Model",
        legend=True,
        scatter_kws={"alpha": 0.6},
        line_kws={"lw": 1},
        ci=False,
    )
    max = df_met["True " + met].max()
    ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color="silver")
    savefig(met + "_compar")


# %%
tr = 64
df_fpds = pd.DataFrame()
df_mape = pd.DataFrame()
fpds = ["Phase", "Frequency", "Damping", "SNR"]
for idx, rslt in enumerate(rslt_list):
    df_fpds_t = pd.DataFrame(columns=fpds)
    df_fpds_t[fpds] = rslt.loc[rslt["type"] == "True"][fpds].loc[0:tr]
    df_fpds_t[met_name] = (
        100
        * np.abs(
            (rslt.loc[rslt["type"] == "Predicted"][met_name].loc[0:tr])
            - (rslt.loc[rslt["type"] == "True"][met_name].loc[0:tr])
        )
        / np.abs(rslt.loc[rslt["type"] == "True"][met_name].loc[0:tr])
    )
    df_fpds_t["Model"] = models[idx]
    df_fpds = df_fpds.append(df_fpds_t, ignore_index=True)

# %%
met_ = ["NAA", "Cr"]
for param in fpds:
    for met in selected_met:
        ax = sns.relplot(
            x=param, y=met, data=df_fpds, hue="Model", legend=True, alpha=0.6
        )
        savefig(met + "_vs_" + param)
