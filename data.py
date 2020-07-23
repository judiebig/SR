import numpy as np


class Data():
    def __init__(self, data, method, shuffle=False):
        inputs, masks, max_len = self.data_masks(data[0],[0])
        self.inputs = np.asarray(inputs)
        self.masks = np.asarray(masks)
        self.max_len = max_len
        self.targets = np.asarray(data[1])
        self.method = method
        self.shuffle = shuffle
        self.length = len(inputs)

    def data_masks(self, data, pad):
        slice_len = [len(session) for session in data]
        max_len = max(slice_len)
        slice =[session + pad*(max_len-len(session)) for session in data]
        masks =[[1]*len(session) + [0]*(max_len-len(session)) for session in data]
        return slice, masks, max_len

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.masks = self.masks[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        if self.method == 'gnn':
            items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
            for u_input in self.inputs[index]:  # slice中的每一条数据
                n_node.append(len(np.unique(u_input)))  # 每一条sesson中有n个node
            max_n_node = np.max(n_node)  # max node
            for u_input in self.inputs[index]:  # slice中的每一条session
                node = np.unique(u_input)  # 某一条session有n个node
                items.append(node.tolist() + (max_n_node - len(node)) * [0])
                u_A = np.zeros((max_n_node, max_n_node))  # 针对当前session构建一个max*max的矩阵
                for i in np.arange(len(u_input) - 1):
                    if u_input[i + 1] == 0:
                        break
                    u = np.where(node == u_input[i])[0][0]
                    v = np.where(node == u_input[i + 1])[0][0]
                    u_A[u][v] = 1
                u_sum_in = np.sum(u_A, 0)
                u_sum_in[np.where(u_sum_in == 0)] = 1
                u_A_in = np.divide(u_A, u_sum_in)
                u_sum_out = np.sum(u_A, 1)
                u_sum_out[np.where(u_sum_out == 0)] = 1
                u_A_out = np.divide(u_A.transpose(), u_sum_out)

                A_in.append(u_A_in)
                A_out.append(u_A_out)
                alias_inputs.append([np.where(node == i)[0][0] for i in u_input])  # 基于ggnn对session重新编码 这样会不会改变原始session的顺序
            return A_in, A_out, alias_inputs, items, self.masks[index], self.targets[index]
        elif self.method == 'normal':
            return self.inputs[index], self.masks[index], self.targets[index]