import matplotlib.pyplot as plt

filename = 'err128.log'
with open(filename) as f:
    data = f.read()

data = data.split('\n')
data = data[:-1]  # last one is blank

data = [a.split(',') for a in data]
data = [[float(b) for b in a] for a in data]

time = []
tr_loss = []
tr_acc = []
val_loss = []
val_acc = []

prev_time = 0

for d in data:
    time += [prev_time + d[0]]
    prev_time = time[-1]
    tr_loss += [d[1]]
    tr_acc += [d[2]]
    val_loss += [d[3]]
    val_acc += [d[4]]


def smoothen(lst, alpha=0.05):
    smooth_lst = []
    prev = lst[0]
    for ele in lst:
        curr = alpha*ele + (1-alpha)*prev
        smooth_lst += [curr]
        prev = curr
    return smooth_lst


plt.plot(time, smoothen(tr_loss))
plt.plot(time, smoothen(val_loss))
plt.xlabel('time')
plt.ylabel('loss')
plt.legend(['train loss', 'validation loss'], loc=1)
plt.show()

plt.plot(time, smoothen(tr_acc))
plt.plot(time, smoothen(val_acc))
plt.xlabel('time')
plt.ylabel('accuracy')
plt.legend(['train accuracy', 'validation accuracy'], loc=1)
plt.show()
