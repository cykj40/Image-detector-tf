// Import TensorFlow.js for the original ML model
// Use the global tf object from the CDN script
// import * as tf from '@tensorflow/tfjs';
const tf = window.tf; // Use the global tf object loaded via script tag in HTML

// Import our computer vision shape detector
import { detectShape } from './shapeDetector.js';

// Debug utility for browser
const debug = {
    log: (message, data = null) => {
        const timestamp = new Date().toISOString();
        const logEntry = `[${timestamp}] ${message}${data ? '\n' + JSON.stringify(data, null, 2) : ''}`;
        const debugOutput = document.getElementById('debug-output');
        if (debugOutput) {
            debugOutput.textContent = logEntry + '\n' + debugOutput.textContent;
        }
        console.log(logEntry);
    }
};

// Test if model files are accessible
const testModelAccess = async () => {
    const modelPaths = [
        './model/model.json',
        '/model/model.json',
        'model/model.json',
        window.location.href + 'model/model.json',
        window.location.origin + '/model/model.json'
    ];

    debug.log("Testing model file accessibility...");

    for (const path of modelPaths) {
        try {
            debug.log(`Fetching from: ${path}`);
            const response = await fetch(path);
            if (response.ok) {
                debug.log(`✓ Success! Model is accessible at: ${path} (Status: ${response.status})`);
                // Try to fetch the weights file too
                const weightsPath = path.replace('model.json', 'weights.bin');
                const weightsResponse = await fetch(weightsPath);
                if (weightsResponse.ok) {
                    debug.log(`✓ Success! Weights are accessible at: ${weightsPath} (Status: ${weightsResponse.status})`);
                } else {
                    debug.log(`✗ Error! Weights file not accessible at: ${weightsPath} (Status: ${weightsResponse.status})`);
                }
            } else {
                debug.log(`✗ Error! Model not accessible at: ${path} (Status: ${response.status})`);
            }
        } catch (error) {
            debug.log(`✗ Fetch failed for: ${path}`, error.message);
        }
    }
};

// Canvas utility functions
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

function displayPrediction(result) {
    const predictionParagraph = document.querySelector(".prediction");
    if (predictionParagraph) {
        // Display the shape name and confidence
        const shapeName = result.shape.charAt(0).toUpperCase() + result.shape.slice(1);
        const confidence = Math.round(result.confidence * 100);
        predictionParagraph.textContent = `Prediction: ${shapeName} (${confidence}% confidence)`;

        // If there's a debug image, display it
        if (result.debugImage) {
            const debugImg = document.createElement('img');
            debugImg.src = result.debugImage;
            debugImg.style.width = '100px';
            debugImg.style.height = '100px';
            debugImg.style.display = 'block';
            debugImg.style.margin = '10px auto';

            // Replace any existing debug image
            const existingDebugImg = document.querySelector('.debug-image');
            if (existingDebugImg) {
                existingDebugImg.replaceWith(debugImg);
            } else {
                debugImg.className = 'debug-image';
                predictionParagraph.appendChild(debugImg);
            }
        }
    }
}

// Original TensorFlow model (kept for compatibility)
let model;

/**
 * Loads the TensorFlow.js model
 * @returns {Promise<void>}
 */
const loadModel = async () => {
    if (!model) {
        try {
            debug.log("Starting model load...");
            showLoading(modelLoadingIndicator);

            // Try both model paths
            let loadError = null;
            const paths = [
                './model/model.json',
                '/model/model.json',
                'model/model.json',
                '../model/model.json'
            ];

            for (const path of paths) {
                try {
                    debug.log(`Attempting to load model from: ${path}`);
                    model = await tf.loadLayersModel(path);
                    debug.log(`Model successfully loaded from: ${path}`);
                    break; // Exit the loop if successful
                } catch (err) {
                    loadError = err;
                    debug.log(`Failed to load from ${path}: ${err.message}`);
                }
            }

            if (!model) {
                debug.log("All model paths failed. Trying public/model path...");
                try {
                    model = await tf.loadLayersModel('./public/model/model.json');
                    debug.log("Model successfully loaded from public/model path");
                } catch (err) {
                    throw loadError || err;
                }
            }

            debug.log("Model loaded successfully");
            // Enable the predict button once model is loaded
            document.getElementById("check-button").disabled = false;
            hideLoading(modelLoadingIndicator);
        } catch (error) {
            debug.log("Error loading model:", error);
            // Still enable the predict button, we'll use CV-based detection as fallback
            document.getElementById("check-button").disabled = false;
            hideLoading(modelLoadingIndicator);
        }
    }
    return model;
};

/**
 * Detect shape using the CV-based approach
 * @param {HTMLCanvasElement} canvas - Canvas element with the drawing
 * @returns {Object} Detection result with shape, confidence, etc.
 */
const detectShapeWithCV = (canvas) => {
    try {
        debug.log("Starting CV-based shape detection");
        // Use our pure JavaScript shape detector
        const result = detectShape(canvas);
        debug.log("CV shape detection result:", result);
        return result;
    } catch (error) {
        debug.log("Error in CV shape detection:", error);
        return { shape: 'unknown', confidence: 0 };
    }
};

/**
 * Makes a prediction on an image using the TensorFlow model with CV fallback
 * @param {HTMLImageElement} img - Image element
 * @returns {Promise<Object>} Prediction result with shape and confidence
 */
const predict = async (img) => {
    try {
        debug.log("Starting prediction process");

        // PRIORITY 1: TensorFlow model (if available)
        if (model) {
            try {
                debug.log("Using TensorFlow model for prediction");
                // Create a temporary canvas to draw the image
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 28;  // Model expects 28x28 images
                tempCanvas.height = 28;

                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.fillStyle = "white"; // Fill with white background
                tempCtx.fillRect(0, 0, 28, 28);
                tempCtx.drawImage(img, 0, 0, 28, 28);

                // Get the image data and preprocess
                const imageData = tempCtx.getImageData(0, 0, 28, 28);

                // Convert to tensor and normalize
                const tensor = tf.browser.fromPixels(imageData, 3)
                    .toFloat()
                    .div(tf.scalar(255))
                    .expandDims(0);

                // Get prediction from model
                const prediction = await model.predict(tensor);
                const values = await prediction.data();

                debug.log("Raw model prediction values:", values);

                // Get the index with highest probability
                const argMax = values.indexOf(Math.max(...values));
                const confidence = values[argMax];
                const shape = argMax === 0 ? 'circle' : 'triangle';

                debug.log(`TensorFlow model predicts ${shape} with confidence ${confidence}`);

                // Cleanup tensors
                tensor.dispose();
                prediction.dispose();

                // ALSO run CV detection to get the debug image
                try {
                    const cvCanvas = document.createElement('canvas');
                    cvCanvas.width = canvas.width;
                    cvCanvas.height = canvas.height;

                    const cvCtx = cvCanvas.getContext('2d');
                    cvCtx.drawImage(img, 0, 0, canvas.width, canvas.height);

                    const cvResult = detectShapeWithCV(cvCanvas);

                    // Return the model prediction with the CV debug image
                    return {
                        shape: shape,
                        confidence: confidence,
                        debugImage: cvResult.debugImage,
                        source: 'model',
                        message: `Model detected ${shape} with ${(confidence * 100).toFixed(1)}% confidence`
                    };
                } catch (cvError) {
                    debug.log("Error getting CV debug image:", cvError);
                    // Return just the model result without debug image
                    return {
                        shape: shape,
                        confidence: confidence,
                        source: 'model',
                        message: `Model detected ${shape} with ${(confidence * 100).toFixed(1)}% confidence`
                    };
                }
            } catch (error) {
                debug.log("Error in TensorFlow prediction:", error);
                // Fall through to CV-based approach
            }
        }

        // FALLBACK: CV-based shape detection if model failed or isn't available
        debug.log("Falling back to CV-based shape detection");
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;

        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(img, 0, 0, canvas.width, canvas.height);

        const cvResult = detectShapeWithCV(tempCanvas);
        debug.log("CV shape detection result:", cvResult);

        if (cvResult.shape !== 'unknown') {
            return {
                ...cvResult,
                source: 'cv',
                message: `CV detected ${cvResult.shape} with ${(cvResult.confidence * 100).toFixed(1)}% confidence (Model fallback)`
            };
        }

        // No valid detection
        return {
            shape: 'unknown',
            confidence: 0,
            debugImage: cvResult.debugImage,
            message: 'Could not detect a shape'
        };
    } catch (error) {
        debug.log("Error during prediction:", error);
        throw new Error("Prediction failed: " + error.message);
    }
};

// Set up frontend event listeners
const clearButton = document.getElementById("clear-button");
const predictButton = document.getElementById("check-button");
const modelLoadingIndicator = document.getElementById("model-loading");
const predictionLoadingIndicator = document.getElementById("prediction-loading");

const showLoading = (indicator) => {
    indicator.classList.remove('hidden');
};

const hideLoading = (indicator) => {
    indicator.classList.add('hidden');
};

clearButton.addEventListener("click", () => {
    debug.log("Clear button clicked");
    resetCanvas();

    const predictionParagraph = document.querySelector(".prediction");
    if (predictionParagraph) {
        predictionParagraph.textContent = "";
        // Also remove debug image if present
        const debugImg = document.querySelector('.debug-image');
        if (debugImg) {
            debugImg.remove();
        }
        debug.log("Prediction text cleared");
    } else {
        debug.log("Prediction paragraph not found");
    }
});

predictButton.addEventListener("click", async () => {
    const canvas = getCanvas();
    const drawing = canvas.toDataURL();
    debug.log("Canvas data captured");

    const newImg = document.getElementsByClassName("imageToCheck")[0];
    newImg.src = drawing;

    newImg.onload = async () => {
        try {
            showLoading(predictionLoadingIndicator);
            const result = await predict(newImg);
            displayPrediction(result);
        } catch (error) {
            const predictionParagraph = document.querySelector(".prediction");
            if (predictionParagraph) {
                predictionParagraph.textContent = "Error during prediction. Please try again.";
            }
        } finally {
            hideLoading(predictionLoadingIndicator);
        }
    };
});

// Download button functionality
const downloadLink = document.getElementById("download-link");
downloadLink.addEventListener("click", (e) => {
    const canvas = getCanvas();
    const image = canvas.toDataURL("image/png");
    downloadLink.href = image;
    downloadLink.download = "drawing.png";
});

// Initially disable the predict button until we're ready
predictButton.disabled = true;

/**
 * Test the model with simple shapes
 */
const testModel = async () => {
    if (!model) {
        debug.log("Model not loaded, cannot test");
        return;
    }

    debug.log("Testing model with simple shapes...");

    // Create sample shapes on temporary canvases
    const shapes = ['circle', 'triangle'];
    const results = [];

    for (const shape of shapes) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');

        // Draw shape
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, 28, 28);
        tempCtx.fillStyle = 'black';

        if (shape === 'circle') {
            tempCtx.beginPath();
            tempCtx.arc(14, 14, 10, 0, 2 * Math.PI);
            tempCtx.fill();
        } else if (shape === 'triangle') {
            tempCtx.beginPath();
            tempCtx.moveTo(14, 4);
            tempCtx.lineTo(24, 24);
            tempCtx.lineTo(4, 24);
            tempCtx.closePath();
            tempCtx.fill();
        }

        // Convert to tensor
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const tensor = tf.browser.fromPixels(imageData, 3)
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(0);

        // Get prediction
        const prediction = await model.predict(tensor);
        const values = await prediction.data();

        // Get result
        const argMax = values.indexOf(Math.max(...values));
        const confidence = values[argMax];
        const predictedShape = argMax === 0 ? 'circle' : 'triangle';

        debug.log(`Test ${shape}: predicted as ${predictedShape} with confidence ${confidence.toFixed(4)}`);
        results.push({
            actualShape: shape,
            predictedShape: predictedShape,
            confidence: confidence,
            correct: (shape === predictedShape)
        });

        // Cleanup tensors
        tensor.dispose();
        prediction.dispose();
    }

    // Log summary
    const correct = results.filter(r => r.correct).length;
    debug.log(`Test complete: ${correct}/${results.length} shapes correctly identified`);

    return results;
};

// Initialize everything when the page loads
window.addEventListener('load', async () => {
    debug.log("Page loaded, initializing...");
    // Try to load the TensorFlow model for compatibility
    try {
        await loadModel();

        // Test the model after loading
        if (model) {
            await testModel();
        }
    } catch (err) {
        // Model loading failed, but that's ok, we'll use CV-based detection
        debug.log("TensorFlow model loading failed, using CV-based detection only");
        // Enable predict button anyway since CV-based detection doesn't need the model
        predictButton.disabled = false;
    }
}); 