const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Set up canvas
ctx.strokeStyle = "black";
ctx.lineJoin = "round";
ctx.lineCap = "round";
ctx.lineWidth = 2;

// Drawing event listeners
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDrawing() {
    isDrawing = false;
}

function resetCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getCanvas() {
    return canvas;
}

function displayPrediction(label) {
    const predictionParagraph = document.querySelector(".prediction");
    if (predictionParagraph) {
        predictionParagraph.textContent = `Prediction: ${label === 0 ? "Circle" : "Triangle"}`;
    }
}

module.exports = { resetCanvas, displayPrediction, getCanvas }; 