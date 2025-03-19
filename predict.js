const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

// Debug utility for Node.js
const debug = {
    log: (message, data = null) => {
        const timestamp = new Date().toISOString();
        const logEntry = `[${timestamp}] ${message}${data ? '\n' + JSON.stringify(data, null, 2) : ''}`;
        console.log(logEntry);
    }
};

let model;

/**
 * Loads the TensorFlow.js model
 * @returns {Promise<void>}
 */
const loadModel = async () => {
    if (!model) {
        try {
            debug.log("Starting model load...");
            const modelPath = path.join(__dirname, 'model', 'model.json');
            model = await tf.loadLayersModel(`file://${modelPath}`);
            debug.log("Model loaded successfully");
        } catch (error) {
            debug.log("Error loading model:", error);
            throw new Error("Failed to load model. Please check if model files exist.");
        }
    }
    return model;
};

/**
 * Processes an image file and returns a tensor ready for prediction
 * @param {string} imagePath - Path to the image file
 * @returns {Promise<tf.Tensor>} Processed image tensor
 */
const processImage = async (imagePath) => {
    try {
        debug.log("Starting image processing");

        // Load image using node-canvas
        const image = await loadImage(imagePath);
        const canvas = createCanvas(image.width, image.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);

        // Get image data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const processedImg = tf.browser.fromPixels(imageData, 4);
        debug.log("Image converted to tensor");

        // Resize to 28x28
        const resizedImg = tf.image.resizeNearestNeighbor(processedImg, [28, 28]);
        debug.log("Image resized to 28x28");

        // Convert to float32 and normalize to [0,1]
        const normalizedImg = tf.cast(resizedImg, 'float32').div(255.0);
        debug.log("Image normalized");

        // Remove alpha channel and add batch dimension
        const inputTensor = tf.expandDims(normalizedImg.slice([0, 0, 0], [28, 28, 3]), 0);
        debug.log("Input tensor prepared");

        return inputTensor;
    } catch (error) {
        debug.log("Error processing image:", error);
        throw new Error("Failed to process image");
    }
};

/**
 * Makes a prediction on an image file
 * @param {string} imagePath - Path to the image file to predict
 * @returns {Promise<number>} Predicted class (0 for circle, 1 for triangle)
 */
const predict = async (imagePath) => {
    try {
        debug.log("Starting prediction process");

        // Ensure model is loaded
        if (!model) {
            await loadModel();
        }

        // Process the image
        const inputTensor = await processImage(imagePath);

        // Get predictions
        const predictions = await model.predict(inputTensor).data();
        debug.log("Raw predictions:", Array.from(predictions));

        // Get the predicted class
        const label = Array.from(predictions).indexOf(Math.max(...Array.from(predictions)));
        debug.log("Prediction completed:", label);

        // Clean up tensors
        tf.dispose([inputTensor]);
        debug.log("Tensors disposed");

        return label;
    } catch (error) {
        debug.log("Error during prediction:", error);
        throw new Error("Prediction failed");
    }
};

/**
 * Process a directory of images and save results
 * @param {string} inputDir - Directory containing images to process
 * @param {string} outputFile - Path to save results
 * @returns {Promise<Array>} Array of prediction results
 */
const processDirectory = async (inputDir, outputFile) => {
    try {
        debug.log(`Processing directory: ${inputDir}`);

        // Ensure model is loaded
        await loadModel();

        // Read directory contents
        const files = fs.readdirSync(inputDir);
        const results = [];

        // Process each image
        for (const file of files) {
            if (file.match(/\.(jpg|jpeg|png)$/i)) {
                const imagePath = path.join(inputDir, file);
                debug.log(`Processing image: ${file}`);

                try {
                    const prediction = await predict(imagePath);
                    results.push({
                        file,
                        prediction: prediction === 0 ? 'circle' : 'triangle',
                        confidence: prediction
                    });
                } catch (error) {
                    debug.log(`Error processing ${file}:`, error);
                    results.push({
                        file,
                        error: error.message
                    });
                }
            }
        }

        // Save results
        fs.writeFileSync(outputFile, JSON.stringify(results, null, 2));
        debug.log(`Results saved to: ${outputFile}`);

        return results;
    } catch (error) {
        debug.log("Error processing directory:", error);
        throw new Error("Failed to process directory");
    }
};

// Example usage
async function main() {
    try {
        // Example 1: Process a single image
        const imagePath = path.join(__dirname, 'data', 'test', 'circle1.png');
        console.log('\nProcessing single image:', imagePath);
        const prediction = await predict(imagePath);
        console.log('Prediction:', prediction === 0 ? 'circle' : 'triangle');

        // Example 2: Process a directory of images
        const inputDir = path.join(__dirname, 'data', 'test');
        const outputFile = path.join(__dirname, 'predictions.json');
        console.log('\nProcessing directory:', inputDir);
        const results = await processDirectory(inputDir, outputFile);
        console.log('Results:', results);

    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

// Run the example
main(); 