const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const { loadData, getTrainData, getTestData } = require('./getData');

const model = require("./createModel");

// Data augmentation function
const augmentData = (images, labels) => {
    console.log('Augmenting training data...');
    const augmentedImages = [];
    const augmentedLabels = [];

    // Helper to add the augmented data to the arrays
    const addAugmentation = (image, label) => {
        augmentedImages.push(image);
        augmentedLabels.push(label);
    };

    // Process each image
    for (let i = 0; i < images.shape[0]; i++) {
        const image = tf.slice(images, [i, 0, 0, 0], [1, 28, 28, 3]);
        const label = tf.slice(labels, [i, 0], [1, 2]);

        // Add original image
        addAugmentation(image, label);

        // Add flipped version (horizontal)
        const flippedH = tf.tidy(() => tf.image.flipLeftRight(image));
        addAugmentation(flippedH, label);

        // Add rotated versions (slight rotations)
        for (const angle of [-0.2, -0.1, 0.1, 0.2]) {
            const rotated = tf.tidy(() => {
                // Using affine transform for rotation
                const radians = angle * Math.PI;
                const cosAngle = Math.cos(radians);
                const sinAngle = Math.sin(radians);

                return tf.image.transform(
                    image,
                    [
                        cosAngle, sinAngle, 0,
                        -sinAngle, cosAngle, 0,
                        0, 0, 1
                    ],
                    'bilinear'
                );
            });
            addAugmentation(rotated, label);
        }
    }

    // Stack all augmented images and labels
    const augmentedImagesStacked = tf.concat(augmentedImages, 0);
    const augmentedLabelsStacked = tf.concat(augmentedLabels, 0);

    console.log(`Data augmentation complete. Original: ${images.shape[0]} images, Augmented: ${augmentedImagesStacked.shape[0]} images`);

    return {
        images: augmentedImagesStacked,
        labels: augmentedLabelsStacked
    };
};

// Count the number of samples in each class
const countClasses = (labels) => {
    const counts = [0, 0];
    const labelsArray = Array.from(labels.argMax(1).dataSync());

    for (const label of labelsArray) {
        counts[label]++;
    }

    return {
        circles: counts[0],
        triangles: counts[1],
        total: labelsArray.length
    };
};

const train = async () => {
    try {
        await loadData();

        const { images: trainImages, labels: trainLabels } = getTrainData();
        const { images: testImages, labels: testLabels } = getTestData();

        // Count before augmentation
        const beforeCounts = countClasses(trainLabels);
        console.log(`Training data before augmentation: ${beforeCounts.total} images (${beforeCounts.circles} circles, ${beforeCounts.triangles} triangles)`);

        // Augment the training data
        const augmentedData = augmentData(trainImages, trainLabels);

        // Count after augmentation
        const afterCounts = countClasses(augmentedData.labels);
        console.log(`Training data after augmentation: ${afterCounts.total} images (${afterCounts.circles} circles, ${afterCounts.triangles} triangles)`);

        // Configure the training process
        model.compile({
            optimizer: tf.train.adam(0.0005),  // Lower learning rate
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        console.log("Starting training...");
        await model.fit(augmentedData.images, augmentedData.labels, {
            epochs: 50,  // More epochs
            batchSize: 32,  // Larger batch size
            validationSplit: 0.2,
            shuffle: true,  // Ensure data is shuffled
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(3)}, accuracy = ${logs.acc.toFixed(3)}, val_loss = ${logs.val_loss.toFixed(3)}, val_acc = ${logs.val_acc.toFixed(3)}`);
                }
            }
        });

        console.log("\nEvaluating model...");
        const evalOutput = await model.evaluate(testImages, testLabels);
        const loss = evalOutput[0].dataSync()[0].toFixed(3);
        const accuracy = evalOutput[1].dataSync()[0].toFixed(3);

        console.log(`\nTest Results:`);
        console.log(`Loss: ${loss}`);
        console.log(`Accuracy: ${accuracy}`);

        // Test on specific examples
        console.log("\nTesting on individual examples:");
        const testPredictions = model.predict(testImages);
        const testLabelsArray = Array.from(testLabels.argMax(1).dataSync());
        const testPredictionsArray = Array.from(testPredictions.argMax(1).dataSync());

        // Check each prediction
        for (let i = 0; i < testLabelsArray.length; i++) {
            const actual = testLabelsArray[i] === 0 ? 'circle' : 'triangle';
            const predicted = testPredictionsArray[i] === 0 ? 'circle' : 'triangle';
            const correct = testLabelsArray[i] === testPredictionsArray[i] ? '✓' : '✗';
            console.log(`Test ${i + 30}: Actual: ${actual}, Predicted: ${predicted} ${correct}`);
        }

        // Create the public directory if it doesn't exist
        const modelDir = path.join(__dirname, 'public', 'model');
        if (!fs.existsSync(modelDir)) {
            fs.mkdirSync(modelDir, { recursive: true });
        }

        // Save the model
        console.log("\nSaving model...");

        // Save model topology as proper JSON object
        const modelJSON = {
            modelTopology: JSON.parse(model.toJSON()),
            format: 'layers-model',
            generatedBy: 'TensorFlow.js v' + tf.version.tfjs,
            convertedBy: null,
            weightsManifest: [{
                paths: ['weights.bin'],
                weights: model.getWeights().map(w => ({
                    name: w.name,
                    shape: w.shape,
                    dtype: w.dtype
                }))
            }]
        };

        fs.writeFileSync(path.join(modelDir, 'model.json'), JSON.stringify(modelJSON, null, 2));

        // Concatenate all weight values into a single buffer
        const weightData = await Promise.all(model.getWeights().map(w => w.data()));
        const totalBytes = weightData.reduce((a, b) => a + b.byteLength, 0);
        const weightsBuf = new Uint8Array(totalBytes);

        let offset = 0;
        weightData.forEach(data => {
            weightsBuf.set(new Uint8Array(data.buffer), offset);
            offset += data.byteLength;
        });

        // Save concatenated weights
        fs.writeFileSync(path.join(modelDir, 'weights.bin'), Buffer.from(weightsBuf));

        console.log(`Model saved to ${modelDir}`);

        // Clean up tensors
        tf.dispose([trainImages, trainLabels, testImages, testLabels,
            augmentedData.images, augmentedData.labels]);

    } catch (error) {
        console.error("Error during training:", error);
    }
}

train();
