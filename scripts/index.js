// import * as tf from '@tensorflow/tfjs';
// const tf = require('@tensorflow/tfjs');
/*document.addEventListener('DOMContentLoaded', function() {
    run()
});*/
const MODEL_URL = '../faceapi_models'
Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    // faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
])
    .then((val) => {
        console.log('Mobilenet And faceRecognitionNet are loaded :)')
    })
    .catch((err) => {
        console.log(err)
    })

let model

async function loadModel() {
    model = await tf.loadLayersModel('../web_model/vgg_model.json');
    // model.summary();
}

loadModel()
    .then((val) => {
        console.log('vgg Model json is Loaded :)');
    })
    .catch((err) => {
        console.log('vgg Model Not Load : ' + err)
    })

let croppedImage = null;
const user_pic = document.getElementById('user_pic')
const preview = document.getElementById('preview')
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

window.onload = function () {
    canvas.width = preview.width;
    canvas.height = preview.height;
    ctx.drawImage(preview, 0, 0);
};

preview.onclick = () => user_pic.click()

user_pic.addEventListener('change', () => {
    const reader = new FileReader()
    reader.onload = (e) => {
        const img = new Image();
        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = e.target.result;
    }
    reader.readAsDataURL(user_pic.files[0]);

    detectFaces(user_pic.files[0])
})


async function detectFaces(input) {
    let imgURL = URL.createObjectURL(input)
    const imgElement = new Image()
    imgElement.src = imgURL

    const results = await faceapi.detectAllFaces(imgElement)
        // .withFaceLandmarks()
        // .withFaceExpressions()
        .then(results => {

            if (Array.isArray(results) && results.forEach) {
                results.forEach(result => {
                    // console.log(result)
                    const {x, y, width, height} = result.box;
                    const xInt = Math.floor(x);
                    const yInt = Math.floor(y);
                    const widthInt = Math.floor(width);
                    const heightInt = Math.floor(height);

                    /*ctx.lineWidth = 3;
                    ctx.strokeRect(x, y, width, height);*/
                    // console.log(xInt, yInt, widthInt, heightInt)

                    const crop = ctx.getImageData(xInt, yInt, widthInt, heightInt);
                    croppedImage = new ImageData(crop.data, widthInt, heightInt);
                    console.log('Image Cropped :)')

                    const input = tf.browser.fromPixels(croppedImage);

                    const resizedImage = tf.image.resizeBilinear(input, [224, 224]);
                    const inputTensor = resizedImage.expandDims(0);

                    // Make predictions
                  model.predict(inputTensor).data()
                      .then(predictions =>{
                          console.log('Prediction is Done :)')

                          // console.log(predictions)
                          const predictionsArray = Array.from(predictions)
                          const celebrityIndex = predictionsArray.indexOf(Math.max(...predictionsArray));
                          console.log(celebrityIndex)

                          /*const celebrityName = classNames[celebrityIndex] || 'Unknown';
                          console.log(celebrityName);*/
                          //Display the results
                          /*const resultDisplay = document.getElementById('result');
                          resultDisplay.innerHTML = `Most similar celebrity: ${celebrityName}`;*/

                      });

                });
            } else {
                console.error('Results is not an array or does not have a forEach function.');
            }
        });
}
