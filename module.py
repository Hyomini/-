import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt

def get_Mu_hat(data, num_neurons, num_labels):
    mu_hat = [np.zeros(num_neurons)] * num_labels
    for label in range(num_labels):
        mu_hat[label] = np.mean(data[label], axis=0)
    return mu_hat


def get_Sigma_hat(data, mu_hat, num_neurons, num_labels, num_data):
    sigma_hat = np.zeros([num_neurons, num_neurons])
    for label in range(num_labels):
        for datum in data[label]:
            u = np.reshape(datum, (num_neurons, 1))
            v = np.reshape(mu_hat[label], (num_neurons, 1))
            sigma_hat += np.matmul((u - v), (u - v).T)
    sigma_hat /= num_data
    return sigma_hat

# type 0 => Euclidean Distance, else Mahalanobis Distance
def get_Dist(type, data, num_neurons, num_labels, mu_hat, inv_sigma_hat):
    temp = [None] * num_labels
    for label in range(num_labels):
        temp[label] = list()
    for label in range(num_labels):
        for datum in data[label]:
            u = np.reshape(datum, (1, num_neurons))
            v = np.reshape(mu_hat[label], (1, num_neurons))
            if type == 0:
                temp[label].append(distance.euclidean(u, v, None) ** 2)
            else:
                temp[label].append(distance.mahalanobis(u, v, inv_sigma_hat) ** 2)
    dist_data = np.array(temp)
    return dist_data

def Cal_dist_max(m_dist_data, num_labels, percentile):
    temp = [None] * num_labels
    for label in range(num_labels):
        temp[label] = list()
    for label in range(num_labels):
        temp[label].append(np.percentile(m_dist_data[label], percentile, interpolation='linear'))
    m_max = np.array(temp)
    return m_max

def K_means_data(data, k, num_labels):
    temp = [None] * k * num_labels
    for label in range(k * num_labels):
        temp[label] = list()
    for label in range(num_labels):
        k_means_data = KMeans(n_clusters=k).fit(data[label])  # k-means clustering
        for i in range(len(data[label])):
            temp[k * label + int(k_means_data.labels_[i])].append(data[label][i])
    data_split = np.array(temp)  # split data
    return data_split

# OOD로 분류되는 (트레이닝)데이터들을 제외
def data_Selection(dist_data, f_x, max_dist_data, num_neurons, num_labels):
    N = 0
    f_x = list(f_x)
    for label in range(num_labels):
        for i in range(len(dist_data[label])):
            if dist_data[label][i] > max_dist_data[label]:
                f_x[label][i] = [-1] * num_neurons  # OOD 는 뉴런이 다 -1이게하고
        for iterate in range(3000):
            for i in range(len(f_x[label])):
                isOOD = 1
                for j in range(num_neurons):
                    if f_x[label][i][j] != -1:
                        isOOD = 0
                        break
                if isOOD == 1:
                    f_x[label].pop(i)
                    break
        N += len(f_x[label])
    data = f_x
    return N, data

# type 0 => Euclidean Distance, else Mahalanobis Distance
def get_TPR(threshold, type, N, k, f_of_x_test, roc_y, num_neurons, num_labels, max_dist_data, mu_hat, inv_sigma_hat, target_label):
    label_of_x_test = np.array(range(N))
    for i in range(N):
        temp = [None] * num_labels
        for label in range(num_labels):
            temp[label] = list()
            u = np.reshape(f_of_x_test[i], (1, num_neurons))
            v = np.reshape(mu_hat[label], (1, num_neurons))
            if type == 0:
                temp[label].append(distance.euclidean(u, v, None) ** 2)
            else:
                temp[label].append(distance.mahalanobis(u, v, inv_sigma_hat) ** 2)
        dist_data_of_x_test = np.array(temp)
        index = np.argmin(dist_data_of_x_test, 0)  # finding index of the closest label
        confidence_score_of_x_test = max_dist_data[index] - dist_data_of_x_test[index]  # computing confidence score
        if confidence_score_of_x_test > threshold:
            label_of_x_test[i] = index // k  # classifying in-distribution data
        else:
            label_of_x_test[i] = -1  # classifying out-of-distribution data
    num_of_in_distribution = 0
    num_of_correctly_classified = 0
    accuracy_on_in_distribution = 0.0
    for i in range(N):
        if label_of_x_test[i] != -1:
            num_of_in_distribution = num_of_in_distribution + 1
            if label_of_x_test[i] == target_label[i]:
                num_of_correctly_classified = num_of_correctly_classified + 1
    if num_of_in_distribution != 0:
        accuracy_on_in_distribution = num_of_correctly_classified / num_of_in_distribution
    tpr = num_of_in_distribution / N
    roc_y.append(tpr)
    print('Classification accuracy on in-distribution: {:.4f}'.format(accuracy_on_in_distribution))
    print('TPR on in-distribution(MNIST): {:.4f}'.format(tpr), end='\n')

def get_FPR(threshold, type, N_ood, k, f_of_x_ood, roc_x, num_neurons, num_labels, max_dist_data, mu_hat, inv_sigma_hat):
    label_of_x_ood = np.array(range(N_ood))
    for i in range(N_ood):
        temp = [None] * num_labels
        for label in range(num_labels):
            temp[label] = list()
            u = np.reshape(f_of_x_ood[i], (1, num_neurons))
            v = np.reshape(mu_hat[label], (1, num_neurons))
            if type == 0:
                temp[label].append(distance.euclidean(u, v, None) ** 2)
            else:
                temp[label].append(distance.mahalanobis(u, v, inv_sigma_hat) ** 2)
        dist_data_of_x_ood = np.array(temp)
        index = np.argmin(dist_data_of_x_ood, 0)  # finding index of the closest label
        confidence_score_of_x_ood = max_dist_data[index] - dist_data_of_x_ood[index]  # computing confidence score
        if confidence_score_of_x_ood > threshold:
            label_of_x_ood[i] = index // k  # classifying in-distribution data
        else:
            label_of_x_ood[i] = -1  # classifying out-of-distribution data
    num_of_in_distribution = 0
    for i in range(N_ood):
        if label_of_x_ood[i] != -1:
            num_of_in_distribution = num_of_in_distribution + 1
    fpr = num_of_in_distribution / N_ood
    roc_x.append(fpr)
    print('FPR on out-of-distribution(ENMIST): {:.4f}'.format(fpr), end='\n')
    
def plot(roc_x, roc_y, roc_x_u, roc_y_u, roc_x_80, roc_y_80):
    plt.figure()
    plt.plot(roc_x, roc_y, color='blue', label='M100')
    plt.plot(roc_x_u, roc_y_u, color='green', label='E')
    plt.plot(roc_x_80, roc_y_80, color='red', label='M80')
    plt.title("ROC curve")
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel("FPR on out-of-distribution(Emnist)")
    plt.ylabel("TPR on in-distribution (mnist)")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.show()