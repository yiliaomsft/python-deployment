from base64 import b64encode, b64decode
from fastai.vision import *
from flask import Flask, flash, request, redirect, url_for, \
    send_from_directory, render_template
import requests
from werkzeug.utils import secure_filename
import json
import os

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    """
    Determines whether the image file considered is legitimate or not

    :param filename: (string) Name of image file processed
    :return: (boolean) True if file name extension is in ALLOWED_EXTENSIONS
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image2json(im_list: list, im_dir: str) -> json:
    """

    :param im_list: (list of strings) List of image file names
    :param im_dir: (string) Directory name
    :return: List containing the byte arrays of each input image
    """

    im_string_list = []
    for im_name in im_list:
        with open(os.path.join(im_dir, im_name), "rb") as image:
            # Extract image bytes
            im_content = image.read()
            # Convert bytes into a string
            im_string_list.append(b64encode(im_content).decode('utf-8'))

    input_to_service = json.dumps({'data': im_string_list})

    return input_to_service


def pred_from_service(folder, filenames_list):
    """
    Calls the webservice to get the predicted classes
    and probabilities of each passed image

    :param folder: (string) Name of the folder in which the images were uploaded
    :param filenames_list: (list of strings) List of image file names
    :return: (list of dictionaries and integer) List of {label, probability}
    and size of that list
    """
    print("Calling the image classification endpoint ...")
    print(UPLOAD_FOLDER)
    data_for_service = image2json(filenames_list, folder)

    # Setting of the authorization header
    # (Authentication is enabled by default when deploying to AKS)
    # key = auth_key
    webservice_url = 'http://104.209.233.179/api/v1/service/' \
          'aks-image-classif-web-svc/score'
    key = 'ymltwgdypOs7Xg7O7oOMtVNLf0cfOwDE'
    # key = 'any key returned by aks_service.get_keys()'
    headers = {'Content-Type': 'application/json'}
    headers['Authorization'] = f'Bearer {key}'

    res = requests.post(webservice_url, data=data_for_service, headers=headers)
    # res = requests.post('URI returned by ask_service.scoring_uri',
    #                     data=data_for_service, headers=headers)

    if res.ok:
        # If service succeeds in computing predictions,
        # return them and the number of such predictions
        # (needed for rendering of the results in an HTML table)
        service_results = res.json()
        results_length = len(service_results)
        return service_results, results_length
    else:
        # If service fails to return predictions, raise an error
        error_message = res.reason
        if error_message == 'Request Entity Too Large':
            error_message = "{} -- Please select smaller or fewer images"\
                .format(error_message)
        raise ValueError(error_message)


@app.route('/')
def index():
    """
    Displays the "upload image page"
    :return: HTML rendering
    """
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    """
    Checks that each uploaded image is legitimate and stores
    its file name in a list
    Renders an HTML table with the results if the image file names
    are legitimate, or the upload page if not

    :return: HTML rendering with images and associated predictions
    """
    uploaded_files = request.files.getlist("file_list")
    filenames = []

    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    print("Files found: {}".format(filenames))
    if filenames:
        return render_template('template.html', all_filenames=filenames,
                               predictions_file=predictions_file)
    else:
        return render_template('index.html')


@app.route('/uploads/<file_name>')
def send_file(file_name):
    """
    Fetches the image which file name is passed as argument

    :param file_name: (string) Uploaded image file name
    :return: Call to the send_from_directory() function
    """
    return send_from_directory(UPLOAD_FOLDER, file_name)


@app.route('/predictions/<all_filenames>')
def predictions_file(all_filenames):
    """
    Calls the webservice to get the predicted classes of the uploaded images

    :param all_filenames: (list of strings) List of uploaded image file names
    :return: Call to the pred_from_service() function
    """
    return pred_from_service(UPLOAD_FOLDER, all_filenames)


if __name__ == "__main__":
    app.run()
