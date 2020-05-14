import json
import db

from flask import Flask, request
from pca import prelim
from train import Iteration

app = Flask(__name__)
app.config.from_pyfile('client.cfg')


def init_db():
    with app.app_context():
        db.init_db(app.config['SCHEMA'], app.config['DATABASE'])


@app.route('/register', methods=['POST'])
def register_study():
    """
    Register study, provide number of iteration and minimum number of samples per iteration,
    do preprocessing and add entry to the db, with study name, number of rounds expected and number of samples to use
    """
    req_data = request.get_json()
    study_name = req_data['name']
    sample_size = req_data['size']
    dp = req_data['dp']

    if not dp and app.config['ONLY_DP']:
        return 'Client only participates in studies supporting differential privacy', 400

    try:
        prelim(app.config['INPUT_SEQ'], app.config['INPUT_LABEL'], app.config['HAIL_PG'], app.config['INPUT_PROC'],
               app.config['HAIL_LOG'])
        db.add_study(study_name, sample_size, app.config['DATABASE'])
        return 'Study successfully registered with client'
    except Exception as e:
        return 'Client cannot participate in this study:' + str(e), 500


@app.route('/train', methods=['POST'])
def fl_round():
    """
    Load preprocessed data, calculate sample range based on round and return false if not enough samples left
    """
    req_data = request.get_json()
    study_name = req_data['name']
    federated_weights = req_data['weights']
    federated_intercepts = req_data['intercepts']
    num_active_clients = req_data['clients']
    random_state = req_data['seed']
    size = db.get_study_size(study_name, app.config['DATABASE'])

    if not size:
        return 'Study was not registered', 400

    try:
        model = Iteration(app.config['INPUT_PROC'], size, num_active_clients, app.config['ALPHA'],
                          app.config['EPSILON'], app.config['MEAN'], app.config['DP_ALGORITHM'])
        if not federated_weights:
            result = model.begin_round(random_state)
        else:
            result = model.begin_round(random_state, federated_weights, federated_intercepts)
        return json.dumps(result)
    except Exception as e:
        return 'Client could not complete round:' + e, 500


@app.route('/end', methods=['POST'])
def end():
    # TODO: Mark study has ended in the db
    return 'Study has been ended'


if __name__ == '__main__':
    init_db()
    app.run()
