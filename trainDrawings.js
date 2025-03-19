const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

const { loadData, getTrainData, getTestData } = require('./getData');

const model = require("./createModel");

const train = async () => {
    try {
        await loadData();

        const { images: trainImages, labels: trainLabels } = getTrainData();
        const { images: testImages, labels: testLabels } = getTestData();

        console.log("Starting training...");
        await model.fit(trainImages, trainLabels, {
            epochs: 10,
            batchSize: 5,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(3)}, accuracy = ${logs.acc.toFixed(3)}`);
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

        // Create the public directory if it doesn't exist
        const modelDir = path.join(__dirname, 'public', 'model');
        if (!fs.existsSync(modelDir)) {
            fs.mkdirSync(modelDir, { recursive: true });
        }

        // Save the model
        console.log("\nSaving model...");

        // Save model architecture
        const modelJSON = {
            modelTopology: model.toJSON(),
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

        fs.writeFileSync(path.join(modelDir, 'model.json'), JSON.stringify(modelJSON));

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

    } catch (error) {
        console.error("Error during training:", error);
    }
}

train();
