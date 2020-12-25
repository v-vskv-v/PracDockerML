import os
import shutil
import json

from forms import UploadTrain, UploadTest, ResultForm
from forms import BaseSelection, GBSelection, RFSelection
from forms import ReturnButton, ReadyButton, InfoButton, DataButtons, ResetButtons

from utils import Thread, to_json, get_losses_info
from utils import get_best_by_train, get_best_by_test, get_plot

from flask import Flask, request, url_for, render_template
from flask import redirect, send_file, send_from_directory

from flask_bootstrap import Bootstrap

from werkzeug.utils import secure_filename


app = Flask(__name__, template_folder='html', static_folder='./../static')
app.config['UPLOAD_FOLDER'] = './../data'
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'mmf_317'
Bootstrap(app)
ALLOWED_EXTENSIONS = {'txt', 'csv'}

train_path_dir = '{}/{}'.format(app.config['UPLOAD_FOLDER'], 'train')
if os.path.exists(train_path_dir):
    shutil.rmtree(train_path_dir)
os.makedirs(train_path_dir)

test_path_dir = '{}/{}'.format(app.config['UPLOAD_FOLDER'], 'test')
if os.path.exists(test_path_dir):
    shutil.rmtree(test_path_dir)
os.makedirs(test_path_dir)

data_paths = {'train': '', 'test': ''}

model_path_dir = '{}/{}'.format(app.config['UPLOAD_FOLDER'], 'model')
model_path = os.path.join(model_path_dir, 'model_description.json')
model_dumped_path = os.path.join(model_path_dir, 'model.pkl')
if os.path.exists(model_path_dir):
    shutil.rmtree(model_path_dir)
os.makedirs(model_path_dir)

ex_dir = '{}/{}'.format(app.config['UPLOAD_FOLDER'], 'example')
ex_model_path = os.path.join(ex_dir, 'model_description.json')
ex_train_path = os.path.join(ex_dir, 'ex_train.csv')
ex_test_path = os.path.join(ex_dir, 'ex_test.csv')

results_dir = '{}/{}'.format(app.config['UPLOAD_FOLDER'], 'results')
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

results_train_path = os.path.join(results_dir, 'loss_train.csv')
results_test_path = os.path.join(results_dir, 'loss_test.csv')
results_plot_train_path = os.path.join(results_dir, 'loss_plot_train.png')
results_plot_test_path = os.path.join(results_dir, 'loss_plot_test.png')
results_info_path = os.path.join(results_dir, 'full_info.json')
results_pred_path = os.path.join(results_dir, 'prediction.csv')

if os.path.isfile(results_plot_test_path):
    os.remove(results_plot_test_path)
if os.path.isfile(results_train_path):
    os.remove(results_train_path)


class Flags:
    training_in_process = False
    testing_in_process = False
    is_recently_trained = False
    is_recently_tested = False
    no_f5 = False
    got_error = ''


class ModelSpecs:
    name = ''
    num = ''
    dim = ''
    depth = ''
    lr = ''


class FileSpecs:
    train_name = ''
    test_name = ''


class ReadySpecs:
    is_ready_train = False
    is_ready_test = False


Specs = [ModelSpecs(), FileSpecs(), ReadySpecs()]
GlobalFlags = Flags()


def selector(key=['0'], factory={'0': BaseSelection, '1': GBSelection, '2': RFSelection}):
    return factory[key[0]]()


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')


@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        upload_train_form = UploadTrain()
        upload_test_form = UploadTest()
        select_form = selector()
        flag_form = ReadyButton()
        info_form = InfoButton()

        if os.path.exists(train_path_dir) and os.listdir(train_path_dir):
            flag_form.is_ready_train = True
        if os.path.exists(test_path_dir) and os.listdir(test_path_dir):
            flag_form.is_ready_test = True
        if os.path.exists(model_path_dir):
            if os.path.isfile(model_path):
                flag_form.is_ready_model = True
            if os.path.isfile(model_dumped_path):
                flag_form.is_trained = True
        flag_form.is_ready_any = flag_form.is_ready_train | flag_form.is_ready_test | flag_form.is_ready_model

        if request.method == 'POST':
            if request.form.get('submit_train') and upload_train_form.validate():
                train_filename = secure_filename(upload_train_form.train_file.data.filename)

                Specs[1].train_name = train_filename
                data_paths['train'] = os.path.join(train_path_dir, train_filename)
                if os.path.exists(train_path_dir):
                    shutil.rmtree(train_path_dir)
                os.makedirs(train_path_dir)
                upload_train_form.train_file.data.save(data_paths['train'])

                GlobalFlags.is_recently_trained = False
                GlobalFlags.is_recently_tested = False

                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir)
                os.makedirs(results_dir)

                train_message = 'train' if train_filename else ''
                return redirect(url_for('success_upload',
                                        train_message=train_message,
                                        test_message=''))

            elif request.form.get('submit_test') and upload_test_form.validate():
                test_filename = secure_filename(upload_test_form.test_file.data.filename)

                Specs[1].test_name = test_filename
                data_paths['test'] = os.path.join(test_path_dir, test_filename)
                if os.path.exists(test_path_dir):
                    shutil.rmtree(test_path_dir)
                os.makedirs(test_path_dir)
                upload_test_form.test_file.data.save(data_paths['test'])

                GlobalFlags.is_recently_tested = False

                if os.path.isfile(results_test_path):
                    os.remove(results_test_path)
                    os.remove(results_pred_path)

                test_message = 'test' if test_filename else ''
                return redirect(url_for('success_upload',
                                        train_message='',
                                        test_message=test_message))

            if upload_test_form.ex_test.data:
                if os.path.exists(test_path_dir):
                    shutil.rmtree(test_path_dir)
                os.makedirs(test_path_dir)
                shutil.copy(ex_test_path, test_path_dir)
                data_paths['test'] = os.path.join(test_path_dir, 'ex_test.csv')
                Specs[1].test_name = 'ex_test.csv'

                GlobalFlags.is_recently_tested = False
                return redirect(url_for('home'))

            elif upload_test_form.delete_test.data:
                if os.path.exists(test_path_dir):
                    shutil.rmtree(test_path_dir)
                os.makedirs(test_path_dir)
                data_paths['test'] = ''
                Specs[1].test_name = ''
                return redirect(url_for('home'))

            elif upload_train_form.ex_train.data:
                if os.path.exists(train_path_dir):
                    shutil.rmtree(train_path_dir)
                os.makedirs(train_path_dir)
                shutil.copy(ex_train_path, train_path_dir)
                data_paths['train'] = os.path.join(train_path_dir, 'ex_train.csv')
                Specs[1].train_name = 'ex_train.csv'

                GlobalFlags.is_recently_trained = False
                GlobalFlags.is_recently_tested = False

                return redirect(url_for('home'))

            elif upload_train_form.delete_train.data:
                if os.path.exists(train_path_dir):
                    shutil.rmtree(train_path_dir)
                os.makedirs(train_path_dir)
                data_paths['train'] = ''
                Specs[1].train_name = ''
                return redirect(url_for('home'))

            if request.form.get('submit') and select_form.validate_on_submit():
                if select_form.key == 'base':
                    selector.__defaults__[0][0] = select_form.model_name.data
                    return redirect(url_for('home'))

                get_json = to_json(select_form)
                if get_json['name'] == 'gb':
                    Specs[0].name = 'Gradient Boosting'
                    Specs[0].lr = get_json['lr']

                elif get_json['name'] == 'rf':
                    Specs[0].name = 'Random Forest'

                Specs[0].num, Specs[0].dim, Specs[0].depth = get_json['num'], get_json['dim'], get_json['depth']
                with open(model_path, 'w') as model_file:
                    json.dump(select_form, model_file, default=to_json)

                if os.path.isfile(model_dumped_path):
                    os.remove(model_dumped_path)

                GlobalFlags.is_recently_trained, GlobalFlags.is_recently_tested = False, False
                selector.__defaults__[0][0] = '0'
                return redirect(url_for('success_upload',
                                        model_message='Fighter is warming up, cool!'))

            elif select_form.key != 'base' and select_form.select_reset.data:
                selector.__defaults__[0][0] = '0'
                return redirect(url_for('home'))

            elif select_form.key == 'base' and select_form.ex_model.data:
                shutil.copy(ex_model_path, model_path_dir)

                if os.path.isfile(model_dumped_path):
                    os.remove(model_dumped_path)

                GlobalFlags.is_recently_trained, GlobalFlags.is_recently_tested = False, False
                with open(ex_model_path, 'r', encoding='utf-8') as model_file:
                    get_json = json.load(model_file)

                if get_json['name'] == 'gb':
                    Specs[0].name = 'Gradient Boosting'
                    Specs[0].lr = get_json['lr']
                elif get_json['name'] == 'rf':
                    Specs[0].name = 'Random Forest'
                Specs[0].num, Specs[0].dim, Specs[0].depth = get_json['num'], get_json['dim'], get_json['depth']
                return redirect(url_for('home'))

            elif info_form.info_submit.data:
                Specs[2].is_ready_train, Specs[2].is_ready_test = flag_form.is_ready_train, flag_form.is_ready_test
                return redirect(url_for('get_info'))

            elif info_form.ready_info.data:
                if GlobalFlags.got_error != '':
                    problem = GlobalFlags.got_error
                    GlobalFlags.got_error = ''
                    return redirect(url_for('error', problem=problem))
                return redirect(url_for('get_results', code='0'))

            elif flag_form.ready_train.data:
                if GlobalFlags.got_error == 'test':
                    problem = GlobalFlags.got_error
                    GlobalFlags.got_error = ''
                    return redirect(url_for('error', problem=problem))
                GlobalFlags.got_error = ''
                if not GlobalFlags.training_in_process:
                    GlobalFlags.is_recently_trained = False
                    GlobalFlags.is_recently_tested = False
                    GlobalFlags.training_in_process = True
                    return redirect(url_for('get_results', code='1'))
                return redirect(url_for('home'))

            elif flag_form.ready_test.data:
                if GlobalFlags.got_error == 'train':
                    problem = GlobalFlags.got_error
                    GlobalFlags.got_error = ''
                    return redirect(url_for('error', problem=problem))
                GlobalFlags.got_error = ''
                if GlobalFlags.is_recently_trained and not GlobalFlags.testing_in_process:
                    GlobalFlags.is_recently_tested = False
                    GlobalFlags.testing_in_process = True
                    return redirect(url_for('get_results', code='2'))
                return redirect(url_for('home'))

        return render_template('home.html', upload_train_form=upload_train_form,
                               upload_test_form=upload_test_form, select_form=select_form,
                               flag_form=flag_form, info_form=info_form, signes=GlobalFlags)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/info', methods=['GET', 'POST'])
def get_info():
    try:
        data_form = DataButtons()
        return_form = ReturnButton()
        if request.method == 'POST':
            if data_form.train.data:
                if data_paths['train']:
                    return send_file(data_paths['train'], mimetype='text/csv', as_attachment=True)
                return redirect(url_for('get_info'))
            if data_form.test.data:
                if data_paths['test']:
                    return send_file(data_paths['test'], mimetype='text/csv', as_attachment=True)
                return redirect(url_for('get_info'))
            return redirect(url_for('home'))
        return render_template('info.html', model=Specs[0], data=Specs[1],
                               data_form=data_form, return_form=return_form, flag_form=Specs[2])
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/success', methods=['GET', 'POST'])
def success_upload():
    try:
        button_form = ReturnButton()

        if button_form.validate_on_submit():
            return redirect(url_for('home'))

        if 'model_message' in request.args:
            button_form.text = request.args['model_message']
            return render_template('success_upload.html', form=button_form)

        train_text = request.args['train_message']
        test_text = request.args['test_message']
        if train_text and test_text:
            button_form.text = 'Data ({}, {}) were uploaded!'.format(train_text, test_text)

        elif not train_text and not test_text:
            button_form.text = 'No data uploaded...'

        else:
            button_form.text = 'Data ({}{}) was uploaded!'.format(train_text, test_text)
        return render_template('success_upload.html', form=button_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/results', methods=['GET', 'POST'])
def get_results():
    try:
        result_form = ResultForm()
        reset_form = ResetButtons()
        plot_url = None

        code = request.args.get('code', '')
        if request.method == 'POST':
            if reset_form.submit.data:
                return redirect(url_for('home'))
            elif reset_form.reset_all.data:
                shutil.rmtree(results_dir)
                shutil.rmtree(model_path_dir)
                shutil.rmtree(test_path_dir)
                shutil.rmtree(train_path_dir)

                os.makedirs(results_dir)
                os.makedirs(model_path_dir)
                os.makedirs(test_path_dir)
                os.makedirs(train_path_dir)
                return redirect(url_for('home'))
            elif result_form.download_plot.data:
                if GlobalFlags.is_recently_tested:
                    return send_file(results_plot_test_path, mimetype='image/png', as_attachment=True)
                else:
                    return send_file(results_plot_train_path, mimetype='image/png', as_attachment=True)
            elif result_form.download_pred.data:
                return send_file(results_pred_path, mimetype='text/csv', as_attachment=True)
            elif result_form.download_info.data:
                return send_file(results_info_path, mimetype='application/json', as_attachment=True)
            elif result_form.download_test.data:
                return send_file(results_test_path, mimetype='text/csv', as_attachment=True)
            elif result_form.download_train.data:
                return send_file(results_train_path, mimetype='text/csv', as_attachment=True)
        if ((code == '0' or code == '1') and GlobalFlags.is_recently_trained) or \
                (code == '2' and GlobalFlags.is_recently_tested):
            train_loss, test_loss, full_info = get_losses_info(results_train_path, results_test_path, results_info_path)

            with open(model_path, 'r', encoding='utf-8') as model_file:
                model_json = json.load(model_file)
            full_info.update(model_json)

            if GlobalFlags.is_recently_trained:
                best_iter, best_train_loss = get_best_by_train(train_loss)
                result_form.train_loss.data = str(best_train_loss)

                full_info['best_train_by_train'] = best_train_loss
                full_info['best_iter_by_train'] = int(best_iter + 1)

                full_info.pop('best_test_loss_by_test', None)
                full_info.pop('best_train_loss_by_test', None)
                full_info.pop('best_iter_by_test', None)

                plot_url = get_plot(train_loss, [], results_plot_train_path)

            if GlobalFlags.is_recently_tested:
                best_iter, best_test_loss, best_train_loss = get_best_by_test(train_loss, test_loss)
                result_form.train_loss.data = str(best_train_loss)
                result_form.test_loss.data = str(best_test_loss)

                full_info['best_test_loss_by_test'] = best_test_loss
                full_info['best_train_loss_by_test'] = best_train_loss
                full_info['best_iter_by_test'] = int(best_iter + 1)

                plot_url = get_plot(train_loss, test_loss, results_plot_test_path)

            with open(results_info_path, 'w') as info_file:
                json.dump(full_info, info_file)

        elif not GlobalFlags.no_f5:
            if code == '1':
                GlobalFlags.no_f5 = True
                thread = Thread()
                thread.train(data_paths['train'], model_path, model_dumped_path, results_dir,
                             results_train_path, results_info_path, GlobalFlags)
            elif code == '2':
                GlobalFlags.no_f5 = True
                thread = Thread()
                thread.test(data_paths['test'], model_dumped_path, results_dir,
                            results_test_path, results_pred_path, results_info_path, GlobalFlags)
        return render_template('results.html', result_form=result_form, reset_form=reset_form,
                               global_flags=GlobalFlags, plot_url=plot_url)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/error', methods=['GET', 'POST'])
def error():
    return_form = ReturnButton()
    problem = request.args['problem']
    if request.method == 'POST':
        return redirect(url_for('home'))
    if problem == 'test':
        GlobalFlags.is_recently_tested = False
        if os.path.exists(test_path_dir):
            shutil.rmtree(test_path_dir)
        os.makedirs(test_path_dir)
    if problem == 'train':
        GlobalFlags.is_recently_trained = False
        GlobalFlags.is_recently_tested = False
        if os.path.exists(train_path_dir):
            shutil.rmtree(train_path_dir)
        os.makedirs(train_path_dir)
        if os.path.isfile(model_dumped_path):
            os.remove(model_dumped_path)
        os.makedirs(test_path_dir)
    return render_template('error.html', form=return_form, problem=problem)
