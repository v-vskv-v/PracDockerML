from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms.validators import DataRequired, InputRequired
from wtforms import StringField, SubmitField, FileField, SelectField
from wtforms import IntegerField, FloatField, ValidationError


def FloatInput(form, field):
    if not (0.0 < float(field.data) <= 1.0):
        raise ValidationError('Must Be Between 0 and 1 Included')


def NumberInput(form, field):
    if int(field.data) <= 0:
        raise ValidationError('Must Be Greater Than Zero')


class UploadTrain(FlaskForm):
    train_file = FileField('Train Data', validators=[FileRequired(),
                                                     FileAllowed(['csv', 'txt'],
                                                                 'Raw Types Only')])
    ex_train = SubmitField('Example Train', render_kw={'formnovalidate': True})
    submit_train = SubmitField('Upload Train Data')
    delete_train = SubmitField('Reset', render_kw={'formnovalidate': True})


class UploadTest(FlaskForm):
    test_file = FileField('Test Data', validators=[FileRequired(),
                                                   FileAllowed(['csv', 'txt'],
                                                               'Raw Types Only')])
    ex_test = SubmitField('Example Test', render_kw={'formnovalidate': True})
    submit_test = SubmitField('Upload Test Data')
    delete_test = SubmitField('Reset', render_kw={'formnovalidate': True})


class BaseSelection(FlaskForm):
    key = 'base'
    model_name = SelectField('Select Model',
                             choices=[('0', '-'),
                                      ('1', 'Gradient Boosting'),
                                      ('2', 'Random Forest')],
                             default='0',
                             validators=[InputRequired()])
    submit = SubmitField('Choose')
    ex_model = SubmitField('Example Model', render_kw={'formnovalidate': True})


class GBSelection(FlaskForm):
    key = 'gb'
    num = IntegerField('Number of Estimators', validators=[DataRequired(), NumberInput])
    dim = FloatField('Feature Dimension for Base Model', validators=[DataRequired(), FloatInput])
    depth = IntegerField('Max Depth', validators=[DataRequired(), NumberInput])
    lr = FloatField('Learning Rate', validators=[DataRequired(), FloatInput])
    submit = SubmitField('Submit')
    select_reset = SubmitField('Cancel', render_kw={'formnovalidate': True})


class RFSelection(FlaskForm):
    key = 'rf'
    num = IntegerField('Number of Estimators', validators=[DataRequired(), NumberInput])
    dim = FloatField('Feature Dimension for Base Model', validators=[DataRequired(), FloatInput])
    depth = IntegerField('Max Depth', validators=[DataRequired(), NumberInput])
    submit = SubmitField('Submit')
    select_reset = SubmitField('Cancel', render_kw={'formnovalidate': True})


class ReturnButton(FlaskForm):
    submit = SubmitField('Return')


class ReadyButton(FlaskForm):
    is_ready_model = False
    is_trained = False
    is_ready_train = False
    is_ready_test = False
    is_ready_any = False
    ready_train = SubmitField('Train Model', render_kw={'formnovalidate': True})
    ready_test = SubmitField('Test Model', render_kw={'formnovalidate': True})
    ready_info = SubmitField('Get Results')


class InfoButton(FlaskForm):
    ready_info = SubmitField('Get Results')
    info_submit = SubmitField('Info')


class DataButtons(FlaskForm):
    train = SubmitField('Download Train')
    test = SubmitField('Download Test')


class ResultForm(FlaskForm):
    train_loss = StringField('Train RMSE', validators=[DataRequired()])
    test_loss = StringField('Test RMSE', validators=[DataRequired()])
    download_plot = SubmitField('Get Plot Image')
    download_train = SubmitField('Get Train Losses per Iter')
    download_test = SubmitField('Get Test Losses per Iter')
    download_info = SubmitField('Get Full Description')
    download_pred = SubmitField('Get Predictions')


class ResetButtons(FlaskForm):
    submit = SubmitField('Return')
    reset_all = SubmitField('Reset All & Start Over at The Beginning')


class Flags:
    training_in_process = False
    testing_in_process = False
    is_recently_trained = False
    is_recently_tested = False
