const canvas = document.getElementById('myCanvas');
const context = canvas.getContext('2d');
const link = document.getElementById('download-link');

// Drawing state variables
let clickX = [];
let clickY = [];
let clickDrag = [];
let paint = false;

// Example dummy labels array
const labels = ["Circle", "Square", "Triangle"];

export const resetCanvas = () => {
    clickX = [];
    clickY = [];
    clickDrag = [];
    paint = false;

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    document.querySelector(".prediction").textContent = "";
    console.log("Canvas reset complete");
};

export const displayPrediction = (label) => {
    const prediction = labels[label] || "Unknown";
    document.querySelector(".prediction").textContent = prediction;
};

const addClick = (x, y, dragging) => {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
};

const redraw = () => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.strokeStyle = "#000";
    context.lineJoin = "round";
    context.lineWidth = 5;

    for (let i = 0; i < clickX.length; i++) {
        context.beginPath();
        if (clickDrag[i] && i) {
            context.moveTo(clickX[i - 1], clickY[i - 1]);
        } else {
            context.moveTo(clickX[i] - 1, clickY[i]);
        }
        context.lineTo(clickX[i], clickY[i]);
        context.closePath();
        context.stroke();
    }
};

// Event listeners
canvas.addEventListener("mousedown", function (e) {
    paint = true;
    addClick(e.offsetX, e.offsetY, false);
    redraw();
});

canvas.addEventListener("mousemove", function (e) {
    if (paint) {
        addClick(e.offsetX, e.offsetY, true);
        redraw();
    }
});

canvas.addEventListener("mouseup", () => paint = false);
canvas.addEventListener("mouseleave", () => paint = false);

// Download link handler
link.addEventListener("click", () => {
    link.href = canvas.toDataURL('image/png');
    link.download = "drawing.png";
});

export const clearRect = () => context.clearRect(0, 0, canvas.width, canvas.height);
export const getCanvas = () => canvas;