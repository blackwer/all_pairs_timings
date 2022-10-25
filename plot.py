import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sys


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide csv to plot")
        sys.exit()

    meas = pd.read_csv(sys.argv[1])

    for prec in ("float", "double"):
        cpu_direct = meas[
            (meas["device"] == "cpu") & (meas["algo"] == "blocked") & (meas["prec"] == prec)
        ]
        cpu_fmm = meas[
            (meas["device"] == "cpu") & (meas["algo"] == "fmm") & (meas["prec"] == prec)
        ]
        gpu_direct = meas[
            (meas["device"] == "gpu") & (meas["algo"] == "direct") & (meas["prec"] == prec)
        ]
        gpu_blocked = meas[
            (meas["device"] == "gpu") & (meas["algo"] == "blocked") & (meas["prec"] == prec)
        ]

        fig = plt.figure(dpi=200)

        plt.loglog(cpu_direct['ntrg'], cpu_direct['tot'])
        plt.loglog(cpu_fmm['ntrg'], cpu_fmm['tot'])
        plt.loglog(gpu_direct['ntrg'], gpu_direct['tot'])
        plt.loglog(gpu_blocked['ntrg'], gpu_blocked['tot'])

        plt.legend(["CPU Blocked", "CPU FMM", "GPU Direct", "GPU Blocked"])
        plt.xlabel('Source/Target points')
        plt.ylabel('Time (ms)')
        plt.title(f'Eval time vs source/target points for {prec}')
        plt.savefig(f"{prec}.png")
        plt.close()
