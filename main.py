import os
import matplotlib.pyplot as plt
import numpy as np
from training import train
from benchmarking import benchmark


def plot_benchmark_result(gamma_list, filename):
    data = np.load(filename)
    all_loccnet_data = data["all_loccnet_data"]
    all_teleport_data = data["all_teleport_data"]
    loccnet_average = []
    teleportation_average = []

    for i in range(len(all_loccnet_data)):
        loccnet_average.append(sum(all_loccnet_data[i]) / len(all_loccnet_data[i]))
        teleportation_average.append(sum(all_teleport_data[i]) / len(all_teleport_data[i]))

    plt.plot(gamma_list, loccnet_average, "o:", label="LOCCNet", zorder=10)
    plt.plot(gamma_list, teleportation_average, "o-", label="Teleportation", zorder=5)
    plt.xlabel(r"Noise parameter $\gamma$", fontsize="x-large")
    plt.ylabel("Average fidelity", fontsize="x-large")
    plt.grid(True)
    plt.legend(["LOCCNet", "Teleportation"], loc="lower left", fontsize="x-large")
    plt.savefig("channel_simulation_fid.png", dpi=600)
    plt.show()


if __name__ == '__main__':
    # Noise parameters for amplitude damping channels
    gamma_list = [i / 100 for i in range(0, 101, 10)]
    all_loccnet_data = []
    all_teleport_data = []
    # Create directory for storing LOCCNet parameters
    if not os.path.exists("./parameters"):
        os.mkdir("./parameters")
    for gamma in gamma_list:
        print(f"gamma: {gamma}")
        filename = "./parameters/ad_channel_with_gamma=" + str(gamma) + ".npz"
        # Optimize a protocol
        train(gamma, filename, ITR=200, LR=0.2)
        # Benchmark the optimized protocol
        list_loccnet, list_teleport = benchmark(gamma, filename, samples=200)
        all_loccnet_data.append(list_loccnet)
        all_teleport_data.append(list_teleport)
    # Create directory for storing benchmark results
    if not os.path.exists("./data"):
        os.mkdir("./data")
    filename_for_plotting = "./data/total_data" + ".npz"
    np.savez(filename_for_plotting, all_loccnet_data=all_loccnet_data, all_teleport_data=all_teleport_data)
    plot_benchmark_result(gamma_list, filename_for_plotting)
