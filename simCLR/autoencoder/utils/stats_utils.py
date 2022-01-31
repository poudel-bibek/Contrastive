import csv
import matplotlib.pyplot as plt
import numpy as np

def generate_boxplot_csv(data_file):
    l1 = ""
    l2 = ""
    l3 = ""
    l4 = ""
    l5 = ""

    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')
            aug_level = int(data[0][-1])

            if aug_level == 1:
                l1 += data[2] + ","
            elif aug_level == 2:
                l2 += data[2] + ","
            elif aug_level == 3:
                l3 += data[2] + ","
            elif aug_level == 4:
                l4 += data[2] + ","
            elif aug_level == 5:
                l5 += data[2] + ","
            else:
                pass
    
    with open('./boxplot/boxplot_csv/resnet_recon_AT20k_curriculumhalf.csv', 'w') as f:
        f.write(l1[:-1] + "\n")
        f.write(l2[:-1] + "\n")
        f.write(l3[:-1] + "\n")
        f.write(l4[:-1] + "\n")
        f.write(l5[:-1] + "\n")

def generate_boxplot(csv_file, name, bound):

    data_list = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            for j in range(len(line)):
                line[j] = float(line[j])
            data_list.append(line)
    
    fig, ax = plt.subplots()
    plt.axis([0, 6, bound[0], bound[1]])
    ax.boxplot(data_list)
    plt.xticks([1,2,3,4,5], ["L1", "L2", "L3", "L4", "L5"])
    fig.savefig('./boxplot/' + name + '.png')

def generate_linegraph_augvsrecon(results_standard_file, results_robust_file):
    aug_results = []
    recon_results = []
    clean_results = []

    aug_robust_results = []
    recon_robust_results = []
    clean_robust_results = []

    with open(results_standard_file, "r") as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_results.append(float(data[1])) # Average results of standard regressor on reconstructed data
            recon_results.append(float(data[2])) # Average results of standard regressor on original augmented data
            clean_results.append(float(data[3])) # Average results of standard regressor on clean data
    
    with open(results_robust_file, "r") as g:
        for line in g:
            line = line.strip()
            data = line.split(',')

            aug_robust_results.append(float(data[1]))
            recon_robust_results.append(float(data[2]))
            clean_robust_results.append(float(data[3]))

    fig, ax = plt.subplots(figsize=(16,5), dpi=200)
    ax.set_ylabel("Avg. Accuracy") 
    ax.plot(np.array(aug_results))
    ax.plot(np.array(recon_results))
    ax.plot(np.array(clean_results))
    ax.legend(["Aug Results", "Recon Results", "Clean Results"])
    fig.savefig('./logs/standard_results_graph.png')

    fig, ax = plt.subplots(figsize=(16,5), dpi=200)
    ax.set_ylabel("Avg. Accuracy") 
    ax.plot(np.array(aug_robust_results))
    ax.plot(np.array(recon_robust_results))
    ax.plot(np.array(clean_robust_results))
    ax.legend(["Aug Results", "Recon Results", "Clean Results"])
    fig.savefig('./logs/robust_results_graph.png')

    fig, ax = plt.subplots(figsize=(16,5), dpi=200)
    ax.set_ylabel("Avg. Accuracy") 
    ax.plot(np.array(aug_results))
    ax.plot(np.array(recon_results))
    ax.plot(np.array(clean_results))
    ax.plot(np.array(aug_robust_results))
    ax.plot(np.array(recon_robust_results))
    ax.plot(np.array(clean_robust_results))
    ax.legend(["Aug Results", "Recon Results", "Clean Results", "Aug Robust Results", "Recon Robust Results", "Clean Robust Results"])
    fig.savefig('./logs/combined_results_graph.png')

    fig, ax = plt.subplots(figsize=(16,5), dpi=200)
    ax.set_ylabel("Avg. Accuracy") 
    ax.plot(np.array(recon_results))
    ax.plot(np.array(clean_results))
    ax.plot(np.array(aug_robust_results))
    ax.plot(np.array(clean_robust_results))
    ax.legend(["Recon Results", "Clean Results", "Aug Robust Results", "Recon Robust Results"])
    fig.savefig('./logs/target_results_graph.png')

def calc_average_stats(results_standard_file, results_robust_file, train_file, curriculum=True):
    aug_standard_total = 0.0
    recon_standard_total = 0.0
    
    aug_robust_total = 0.0
    recon_robust_total = 0.0

    num = 0
    
    with open(results_standard_file, "r") as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_standard_total += float(data[1])
            recon_standard_total += float(data[2])

            num += 1
    
    with open(results_robust_file, "r") as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_robust_total += float(data[1])
            recon_robust_total += float(data[2])
    
    aug_standard_average = aug_standard_total / num
    recon_standard_average = recon_standard_total / num

    aug_robust_average = aug_robust_total / num
    recon_robust_average = recon_robust_total / num

    print(f"The average accuracy for augmentations is: {aug_standard_average}")
    print(f"The average accuracy for reconstruction is: {recon_standard_average}")
    print(f"The average accuracy for robust augmentations is: {aug_robust_average}")
    print(f"The average accuracy for robust reconstructions is: {recon_robust_average}")

    total_time = 0.0
    num = 0

    if curriculum == True:
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip()
                data = line.split(' ')
                data = data[7]

                total_time += float(data[:-2])
                num += 1
    else:
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip()
                data = line.split(' ')

                total_time += float(data[-1])
                num += 1
    
    time_per_epoch_average = total_time / num

    print(f"The average time per epoch is: {time_per_epoch_average}")

    with open('./logs/average_stats', 'w') as f:
        f.write(f"Standard Average Aug Acc: {aug_standard_average}\n")
        f.write(f"Standard Average Recon Acc: {recon_standard_average}\n")
        f.write(f"Robust Average Aug Acc: {aug_robust_average}\n")
        f.write(f"Robust Average Recon Acc: {recon_robust_average}\n")
        f.write(f"Time Per Epoch Average: {time_per_epoch_average} s")

def generate_overleaf_dmproject(file):
    count = 0
    overleaf_dmproject = "& "
    with open(file, 'r') as f:
        for line in f:
          line = line.strip()
          data = line.split(',')

          if count % 5 == 0:
              recon_acc = f"{float(data[2]):.2f}\\% \\\\"
              overleaf_dmproject += recon_acc
              with open('/content/drive/MyDrive/AE_Files/logs/overleaf_dmproject.txt', 'a') as g:
                  g.write(overleaf_dmproject + "\n")
              
              overleaf_dmproject = "& "
          
          else:
              recon_acc = f"{float(data[2]):.2f}\\% & "
              overleaf_dmproject += recon_acc
          
          count += 1