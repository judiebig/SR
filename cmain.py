import pickle
import argparse
from tqdm import tqdm
import time
import json
import datetime
from data import *
from utils import *
from Models import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset: yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--method', type=str, default='normal', help='how to process data: gnn/normal')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--is_train_eval', type=bool, default=False, help='eval train to prevent from over-fitting ')
opt = parser.parse_args()

n_node = 37484


def main():
    ''' ---------load model--------------- '''
    model = trans_to_cuda(BasicLast(opt, n_node))

    # preliminaries
    log_filename = "./result/log_" + model.__class__.__name__ + ".txt"
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename=log_filename, filemode='w')
    m_print(json.dumps(opt.__dict__, indent=4))
    start = time.time()

    ''' ---------process data------------- '''
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, method=opt.method, shuffle=True)
    test_data = Data(test_data, method=opt.method, shuffle=False)


    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        m_print('epoch: {}, ==========================================='.format(epoch))
        flag = 0

        # training
        slices = train_data.generate_batch(opt.batch_size)
        fetches =[]
        train_loss = 0.0
        m_print('start training: {}'.format(datetime.datetime.now()))
        for i, j in zip(slices, tqdm(np.arange(len(slices)))):
            model.optimizer.zero_grad()
            inputs, masks, targets = train_data.get_slice(i)
            inputs = trans_to_cuda(torch.Tensor(inputs).long())
            masks = trans_to_cuda(torch.Tensor(masks).long())
            targets = trans_to_cuda(torch.Tensor(targets).long())
            scores = model(inputs, masks, targets)
            loss = model.criterion(scores, targets - 1)
            loss.backward()
            model.optimizer.step()
            train_loss += loss
            if j % int(len(slices) / 5 + 1) == 0:
                print('Loss: %.4f' % (loss.item()))
        print('\tLoss:\t%.3f' % train_loss)
        print("learning rate is ", model.optimizer.param_groups[0]["lr"])


        # predicting
        if opt.is_train_eval:  # eval train data
            hit, mrr = [], []
            slices = train_data.generate_batch(opt.batch_size)
            m_print('start predicting train data: {}'.format(datetime.datetime.now()))
            with torch.no_grad():
                for i, j in zip(slices, tqdm(np.arange(len(slices)))):
                    inputs, masks, targets = train_data.get_slice(i)
                    inputs = trans_to_cuda(torch.Tensor(inputs).long())
                    masks = trans_to_cuda(torch.Tensor(masks).long())
                    scores = model(inputs, masks, targets)
                    sub_scores = scores.topk(20)[1].cpu().numpy()
                    for score, target in zip(sub_scores, targets):
                        hit.append(np.isin(target - 1, score))
                        if len(np.where(score == target - 1)[0]) == 0:
                            mrr.append(0)
                        else:
                            mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
            hit = np.mean(hit) * 100
            mrr = np.mean(mrr) * 100
            m_print("current train result:")
            m_print("\tRecall@20:\t{}\tMMR@20:\t{}\tEpoch:\t{}".format(hit, mrr, epoch))
        # eval test data
        hit, mrr = [], []
        slices = test_data.generate_batch(opt.batch_size)
        m_print('start predicting: {}'.format(datetime.datetime.now()))
        with torch.no_grad():
            for i, j in zip(slices, tqdm(np.arange(len(slices)))):
                inputs, masks, targets = test_data.get_slice(i)
                inputs = trans_to_cuda(torch.Tensor(inputs).long())
                masks = trans_to_cuda(torch.Tensor(masks).long())
                scores = model(inputs, masks, targets)
                sub_scores = scores.topk(20)[1].cpu().numpy()
                for score, target in zip(sub_scores, targets):
                    hit.append(np.isin(target - 1, score))
                    if len(np.where(score == target - 1)[0]) == 0:
                        mrr.append(0)
                    else:
                        mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        bad_counter += 1 - flag

        m_print("current test result:")
        m_print("\tRecall@20:\t{}\tMMR@20:\t{}\tEpoch:\t{}".format(hit,mrr,epoch))
        if bad_counter >= opt.patience:
            break
        model.scheduler.step()
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == "__main__":
    main()
