const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const trainImagesDir = "./data/train"
const testImagesDir = "./data/test"

let trainData, testData;

const loadImages = async (dataDir) => {
    const images = [];
    const labels = [];

    let files = fs.readdirSync(dataDir);
    for (let i = 0; i < files.length; i++) {
        let filePath = path.join(dataDir, files[i]);

        // Read and process image using sharp
        const pixels = await sharp(filePath)
            .resize(28, 28)
            .raw()
            .toBuffer();

        // Convert to float32 array and normalize to [0,1]
        const float32Data = new Float32Array(pixels.length);
        for (let j = 0; j < pixels.length; j++) {
            float32Data[j] = pixels[j] / 255.0;
        }

        // Create tensor and reshape to [28, 28, 3]
        const imageTensor = tf.tensor(float32Data).reshape([28, 28, 4]).slice([0, 0, 0], [28, 28, 3]);
        images.push(imageTensor);

        // Process labels
        const circle = files[i].toLowerCase().includes('circle');
        const triangle = files[i].toLowerCase().includes('triangle');

        if (circle) {
            labels.push(0);
        } else if (triangle) {
            labels.push(1);
        }
    }

    return [images, labels];
}

const loadData = async () => {
    console.log("Loading images...");
    trainData = await loadImages(trainImagesDir);
    testData = await loadImages(testImagesDir);
    console.log("Images loaded successfully");
}

const getTrainData = () => {
    if (!trainData) {
        throw new Error("Training data not loaded. Call loadData() first.");
    }
    return {
        images: tf.stack(trainData[0]),
        labels: tf.oneHot(tf.tensor1d(trainData[1], 'int32'), 2)
    };
}

const getTestData = () => {
    if (!testData) {
        throw new Error("Test data not loaded. Call loadData() first.");
    }
    return {
        images: tf.stack(testData[0]),
        labels: tf.oneHot(tf.tensor1d(testData[1], 'int32'), 2)
    };
}

module.exports = { getTrainData, getTestData, loadData };

