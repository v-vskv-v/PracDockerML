import os
import shutil
import json
import pickle
import time
import threading
import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ensembles import GradientBoostingMSE, RandomForestMSE, root_mean_squared_error


def to_json(model):
    obj = {'name': model.key, 'num': model.num.data,
           'dim': model.dim.data, 'depth': model.depth.data}
    if model.key == 'gb':
        obj['lr'] = model.lr.data
    return obj


def get_data(path, shuffle=True, seed=317):
    data = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    X, y = data[:, :-1], data[:, -1]
    if shuffle:
        indexes = np.arange(X.shape[0])
        np.random.seed(seed)
        np.random.shuffle(indexes)
        X = X[indexes]
        y = y[indexes]
    return X, y


def make_model(model_path):
    with open(model_path, 'r', encoding='utf-8') as model_file:
        get_params = json.load(model_file)
    if get_params['name'] == 'gb':
        model = GradientBoostingMSE(n_estimators=int(get_params['num']),
                                    learning_rate=float(get_params['lr']),
                                    max_depth=int(get_params['depth']),
                                    feature_subsample_size=float(get_params['dim']))
    elif get_params['name'] == 'rf':
        model = RandomForestMSE(n_estimators=int(get_params['num']),
                                max_depth=int(get_params['depth']),
                                feature_subsample_size=float(get_params['dim']))
    return model


def train_model(train_path, model_path, model_dumped_path, results_dir,
                results_train_path, results_info_path, global_flags):
    try:
        start_time = time.time()
        X_train, y_train = get_data(train_path)
        model = make_model(model_path)
        model, train_loss, _ = model.fit(X_train, y_train)
        full_time = time.time() - start_time
        with open(model_dumped_path, 'wb') as output:
            pickle.dump(model, output)
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        np.savetxt(results_train_path, train_loss, delimiter=';')
        with open(results_info_path, 'w') as info_file:
            json.dump({'train_time': full_time}, info_file)
        global_flags.is_recently_trained = True
        global_flags.training_in_process = False
        global_flags.no_f5 = False
    except Exception as exc:
        global_flags.got_error = 'train'
        global_flags.is_recently_trained = False
        global_flags.training_in_process = False
        global_flags.no_f5 = False


def test_model(test_path, model_dumped_path, results_dir,
               results_test_path, results_pred_path, results_info_path, global_flags):
    try:
        start_time = time.time()
        X_test, y_test = get_data(test_path, shuffle=False)
        with open(model_dumped_path, 'rb') as input:
            model = pickle.load(input)
        y_preds = model.predict(X_test, all=True)
        test_loss = root_mean_squared_error(y_test, y_preds)
        full_time = time.time() - start_time
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        np.savetxt(results_test_path, test_loss, delimiter=';')
        np.savetxt(results_pred_path, y_preds, delimiter=';')
        if os.path.isfile(results_info_path):
            with open(results_info_path, 'r', encoding='utf-8') as input:
                get_json = json.load(input)
        else:
            get_json = {}
        get_json['test_time'] = full_time
        with open(results_info_path, 'w') as info_file:
            json.dump(get_json, info_file)
        global_flags.is_recently_tested = True
        global_flags.testing_in_process = False
        global_flags.no_f5 = False
    except Exception as exc:
        global_flags.got_error = 'test'
        global_flags.is_recently_tested = False
        global_flags.testing_in_process = False
        global_flags.no_f5 = False


class Thread(object):

    def __init__(self):
        self.thread = None

    def train(self, train_path, model_path, model_dumped_path, results_dir,
              results_train_path, results_info_path, global_flags):
        self.thread = threading.Thread(target=train_model, args=(train_path, model_path,
                                                                 model_dumped_path, results_dir,
                                                                 results_train_path,
                                                                 results_info_path, global_flags))
        self.thread.daemon = True
        self.thread.start()

    def test(self, test_path, model_dumped_path, results_dir,
             results_test_path, results_pred_path, results_info_path, global_flags):
        self.thread = threading.Thread(target=test_model, args=(test_path, model_dumped_path,
                                                                results_dir, results_test_path,
                                                                results_pred_path,
                                                                results_info_path, global_flags))
        self.thread.daemon = True
        self.thread.start()


def get_losses_info(train_path, test_path, info_path):
    train_loss = []
    test_loss = []
    if os.path.isfile(train_path):
        train_loss = np.loadtxt(train_path, delimiter=';', dtype=np.float64)
    if os.path.isfile(test_path):
        test_loss = np.loadtxt(test_path, delimiter=';', dtype=np.float64)

    with open(info_path, 'r', encoding='utf-8') as info_json:
        full_info = json.load(info_json)
    return train_loss, test_loss, full_info


def get_best_by_train(train_loss):
    best_iter = np.argmin(train_loss)
    return best_iter, train_loss[best_iter]


def get_best_by_test(train_loss, test_loss):
    best_iter = np.argmin(test_loss)
    return best_iter, test_loss[best_iter], train_loss[best_iter]


def get_plot(train_loss, test_loss, path):
    fig = plt.figure(figsize=(20, 10))

    x = np.arange(train_loss.size)
    plt.plot(x, train_loss, label='Train', c='royalblue', linewidth=3, alpha=0.8)
    if len(test_loss) > 0:
        plt.plot(x, test_loss, label='Test', c='crimson', linewidth=3, alpha=0.8)

    plt.xlabel('Iteration', fontsize=17)
    plt.ylabel('RMSE',  fontsize=17)
    plt.title('RMSE to Iteration',  fontsize=25)
    plt.legend(fontsize=13)
    plt.grid(True)

    fig.savefig(path)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('ascii')
    return plot_url
