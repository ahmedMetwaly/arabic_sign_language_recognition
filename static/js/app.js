// socketio initialization
const socket = io();

// Creating SS to convert server responses to voice messages
const synth = window.speechSynthesis;

// Creating video element for camera streaming
const cameraView = document.querySelector('#cameraView');
// Creating canvas element to store a specific frame -- image -- from the camera
const canvas = document.createElement('canvas');
// Selecting Error Area div, anyerror will show up to the user in this div
const messageArea = document.querySelector('#messageArea');
messageArea.style.paddingTop = '25%';
// Selecting image element that will show final prediction
const predictionImage = document.querySelector('#predictionImage');

cameraView.autoplay = true;
cameraView.playsinline = true;

// Access the device camera and stream to cameraView
function camera_start() {
    navigator.mediaDevices
        .getUserMedia({
            video: true,
            audio: false
        })
        .then(function (stream) {
            track = stream.getTracks()[0];
            cameraView.srcObject = stream;
        }).then(function () {
            send_photo()
        })
        .catch(function (error) {
            cameraErr = 'In order to this app to work:</br>' +
                '- You need a device with a camera</br>' +
                '- Allow us to use your camera</br>' +
                'Please note: All photos sent to the server are automatically deleted';
            console.error(error, cameraErr);
            messageArea.innerHTML = cameraErr;
        });
}

function send_photo() {
    canvas.height = cameraView.videoHeight;
    canvas.width = cameraView.videoWidth;
    // Get current frame of the video into the canvas 
    canvas.getContext("2d").drawImage(cameraView, 0, 0);

    // Getting the source of the image 
    let data = canvas.toDataURL("image/jpeg");

    // Check if the camera is still ON 
    if (data === "data:,") {
        console.log("Turning on Your Camera");
        messageArea.innerHTML = "Turning on Your Camera";
        setTimeout(send_photo, 1000);
        return;
    }
    messageArea.innerHTML = "Predicting.."

    socket.emit("upload", data)
}

socket.on('speak', function(message){
    const msg = new SpeechSynthesisUtterance(message);
    synth.speak(msg);
    messageArea.innerHTML = message;
})

socket.on('prediction', function (base64_src) {
    predictionImage.src = base64_src;
    send_photo();
});

socket.on('connect', () => camera_start())





