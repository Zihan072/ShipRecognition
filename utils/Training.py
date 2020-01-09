import time
import sys
import torch
import collections

from IPython.display import clear_output
import matplotlib.pyplot as plt

def binary_accuracy(x, labels):
    return ((x > 0.5).float() == labels).sum().numpy() / len(labels)

class MovMean:
    def __init__(self, len):
        self.len = len
        self.buff = collections.deque()
        self.sum = 0

    @property
    def mean(self):
        return self.sum / len(self.buff)

    def push(self, x):
        self.buff.append(x)
        self.sum += x
        if len(self.buff) > self.len:
            self.sum -= self.buff.popleft()


class IReporter:

    def begin_epoch(self, n_epoch, max_epoch): ...

    def end_epoch(self, n_epoch, max_epoch, loss, val_loss, val_acc): ...

    def end_batch(self, n_sample, max_sample, loss, acc, n_epoch): ...


class ConsoleReporter(IReporter):

    def __init__(self, update_interval=.5):
        self.interval = update_interval
        self.last_update = 0

    def begin_epoch(self, n_epoch, max_epoch):
        print(f"  Epoch {n_epoch+1}/{max_epoch}:")

    def end_epoch(self, n_epoch, max_epoch, loss, val_loss, val_acc):
        self.print_last_line(f"loss: {loss:.5f} - val_loss: {val_loss:.5f}\n")

    def end_batch(self, n_sample, max_sample, loss, acc, n_epoch):
        if time.time() - self.last_update > self.interval:
            self.print_last_line(f"{n_sample}/{max_sample} - loss: {loss:.5f}, acc: {acc:.5f}")
            self.last_update = time.time()

    def print_last_line(self, s):
        sys.stdout.write("\r" + str(s))
        sys.stdout.flush()

class PlotReporter(IReporter):
    def __init__(self, update_interval=1., updates_per_epoch=10, figsize=(11,5)):
        self.figsize = figsize
        self.interval = update_interval
        self.updates_per_epoch=updates_per_epoch
        self.last_update = 0
        self.x = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc= []

    def end_epoch(self, n_epoch, max_epoch, loss, val_loss, val_acc):
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.plot()

    def end_batch(self, n_sample, max_sample, loss, acc, n_epoch):
        x = n_epoch+n_sample/max_sample
        if len(self.x) == 0 or x - self.x[-1] > 1/self.updates_per_epoch:
            self.x.append(x)
            self.loss.append(loss)
            self.acc.append(acc)
        if time.time() - self.last_update > self.interval:
            self.plot()
            self.last_update = time.time()

    def plot(self):
        clear_output(wait=True)
        fig, ax = plt.subplots(1, 2, figsize=self.figsize)

        ax[0].plot(self.x, self.loss, label='loss')
        ax[0].plot(list(range(1, len(self.val_loss)+1)), self.val_loss, label='validation_loss')
        ax[0].legend()

        ax[1].plot(self.x, self.acc, label='accuracy')
        ax[1].plot(list(range(1, len(self.val_acc)+1)), self.val_acc, label='validation_accuracy')
        ax[1].legend()

        plt.show()



class TrainingHelper:

    def __init__(self, net, loss_function, optimizer):
        self.net = net
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.epoch = 0
        self.loss_mean_length = 1024
        self.reporter = ConsoleReporter()
        self.accuracy = binary_accuracy


    def fit(self,n_epochs, train_loader, val_loader=None):
        assert(n_epochs > 0)
        n_epochs = self.epoch+n_epochs
        hist_loss = MovMean(self.loss_mean_length//train_loader.batch_size)
        hist_acc = MovMean(self.loss_mean_length//train_loader.batch_size)
        for epoch in range(self.epoch, n_epochs):
            sum_loss = 0
            self.reporter.begin_epoch(epoch, n_epochs)
            for i, data in enumerate(iter(train_loader)):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                hist_loss.push(loss.item())
                acc = self.accuracy(outputs, labels)
                hist_acc.push(acc)
                sum_loss += loss.item()
                self.reporter.end_batch(i*train_loader.batch_size, train_loader.batch_size*len(train_loader), hist_loss.mean, hist_acc.mean, epoch)

            val_loss, val_acc = 0, 0
            if val_loader is not None:
                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data
                        outputs = self.net(inputs)
                        loss = self.loss_function(outputs, labels)
                        val_loss += loss.item()
                        val_acc += self.accuracy(outputs, labels)
                    val_loss /= len(val_loader)
                    val_acc /= len(val_loader)
            self.reporter.end_epoch(epoch, n_epochs, sum_loss/len(train_loader), val_loss, val_acc)
        self.epoch = epoch+1




