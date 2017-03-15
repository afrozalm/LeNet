import matplotlib.pyplot as plt


def get_data(filename):
    with open(filename) as f:
        data = f.read()

    data = data.split('\n')
    data = data[:-1]
    data = [a.split(',') for a in data]
    data = [[float(b) for b in a] for a in data]
    return data


def get_acc_list(data, train=True):
    acc = []
    if train:
        for d in data:
            acc += [d[2]]
    else:
        for d in data:
            acc += [d[4]]
    return acc


def get_time_list(data):
    prev_time = 0
    time = []
    for d in data:
        time += [prev_time + d[0]]
        prev_time = time[-1]
    return time


def get_loss_list(data, train=True):
    loss = []
    if train:
        for d in data:
            loss += [d[1]]
    else:
        for d in data:
            loss += [d[3]]
    return loss


def smoothen(lst, alpha=0.5):
    smooth_lst = []
    prev = lst[0]
    for ele in lst:
        curr = alpha*ele + (1-alpha)*prev
        smooth_lst += [curr]
        prev = curr
    return smooth_lst


d32 = get_data('err32.log')
d64 = get_data('err64.log')
d128 = get_data('err128.log')
d200 = get_data('err200.log')

plt.plot(smoothen(get_acc_list(d32)))
plt.plot(smoothen(get_acc_list(d64)))
plt.plot(smoothen(get_acc_list(d128)))
plt.plot(smoothen(get_acc_list(d200)))
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['32 batch', '64 batch', '128 batch', '200 batch'], loc=2)
plt.show()

plt.plot(smoothen(get_loss_list(d32)))
plt.plot(smoothen(get_loss_list(d64)))
plt.plot(smoothen(get_loss_list(d128)))
plt.plot(smoothen(get_loss_list(d200)))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['32 batch', '64 batch', '128 batch', '200 batch'], loc=2)
plt.show()

plt.plot(get_time_list(d32), smoothen(get_acc_list(d32)))
plt.plot(get_time_list(d64), smoothen(get_acc_list(d64)))
plt.plot(get_time_list(d128), smoothen(get_acc_list(d128)))
plt.plot(get_time_list(d200), smoothen(get_acc_list(d200)))
plt.xlabel('time')
plt.ylabel('accuracy')
plt.legend(['32 batch', '64 batch', '128 batch', '200 batch'], loc=2)
plt.show()

plt.plot(get_time_list(d32), smoothen(get_loss_list(d32)))
plt.plot(get_time_list(d64), smoothen(get_loss_list(d64)))
plt.plot(get_time_list(d128), smoothen(get_loss_list(d128)))
plt.plot(get_time_list(d200), smoothen(get_loss_list(d200)))
plt.xlabel('time')
plt.ylabel('loss')
plt.legend(['32 batch', '64 batch', '128 batch', '200 batch'], loc=2)
plt.show()
