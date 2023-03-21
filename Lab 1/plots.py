import re
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_type = 'xor'
    lr = '1.0'
    num_features = ['2-2-2-1', '2-4-4-1', '2-8-8-1']
    num_units = [2, 4, 8]

    title = []
    if data_type != '*':
        title.append(f'Data type: {data_type}')
    if lr != '*':
        title.append('Learning rate: 1.0')
    # if num_features != '*':
    #     title.append('Network: 2-4-4-1')
    title = '\n'.join(title)
    plt.title(title)

    pattern = f'train_loss/{data_type}_units*.npy'
    fns = sorted(glob(pattern))
    i=0
    for fn in fns:
        losses = np.load(fn)
        pattern = '' 
        pattern += data_type if data_type!='*' else '(.*)'
        pattern += f'_units{num_units[i]}' if lr!='*' else '_lr(.*)'
        # pattern += f'_{num_features}' if num_features!='*' else '_(.*)'
        pattern += '_lc'
        items = re.findall(rf'{pattern}', fn)[0]
        # print(items)
        # if type(items) is tuple:
        #     items = list(reversed(items))
        # else:
        #     items = [items]
        label = ''
        label += f', Data type: {items[0:6]}'
        
        # label += f', Network: {items.pop()}' if num_features=='*' else ''
        label += f', Network: {num_features[i]}'
        label += f', Learning rate: 1.0'
        print(label)
        plt.plot(np.arange(len(losses)), losses, label=label[21:37], lw=2)

        i+=1

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.show()