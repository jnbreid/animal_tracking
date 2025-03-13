import matplotlib.pyplot as plt
import numpy as np

"""
function to visualize the distribution of the training dataset for DistNet, to ensure
that the dataset does not only contain positive/negative instances, and that the data distribution
is 'acceptable' (no obvious errors in the data)
"""

def visualize_dataset_distribution(c_dataset):    
    pos_elems = []
    neg_elems = []

    for i in range(len(c_dataset)):
        item, label = c_dataset[i]

        if label[0].item() == 1:
            pos_elems.append(item)
        else:
            neg_elems.append(item)


    pos_feature_dist = [item[0] for item in pos_elems]
    pos_IoU = [item[1] for item in pos_elems]
    pos_last = [item[2] for item in pos_elems]

    neg_feature_dist = [item[0] for item in neg_elems]
    neg_IoU = [item[1] for item in neg_elems]
    neg_last = [item[2] for item in neg_elems]

    #print(pos_IoU)

    bins =  np.arange(0, 1, 0.001)
    bins1 = np.arange(0, 1, 0.01)

    fig, axs = plt.subplots(1,3)
    fig.suptitle("GMOT8 - bird")

    axs[0].set_title('feature distance')
    axs[0].hist(pos_feature_dist, bins = bins, density = True, alpha = 0.75)

    axs[1].set_title('IoU')
    axs[1].hist(pos_IoU, bins = bins1, density = True, alpha = 0.75)

    axs[2].set_title('last matched')
    axs[2].hist(pos_last, density = True, alpha = 0.75)

    axs[0].hist(neg_feature_dist, bins = bins, density = True, alpha = 0.75)

    axs[1].hist(neg_IoU, bins = bins1, density = True, alpha = 0.75)

    axs[2].hist(neg_last, density = True, alpha = 0.75)

    plt.show()