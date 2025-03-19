const fs = require('fs');
const path = require('path');

// Path to the model.json file
const modelJsonPath = path.join(__dirname, 'public', 'model', 'model.json');
const outputModelJsonPath = path.join(__dirname, 'public', 'model', 'model.json');

console.log('Reading model file...');
// Read the current model.json file
const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));

// The issue seems to be that the model is biased toward triangles
// Let's adjust the bias of the final dense layer to favor circles more

// Find the weights for the final dense layer bias
const weights = modelJson.weightsManifest[0].weights;
const finalLayerBiasIndex = weights.findIndex(w => w.name === 'dense_Dense2/bias');

if (finalLayerBiasIndex !== -1) {
    console.log('Found final layer bias weights.');
    console.log('Original weights manifest:', weights[finalLayerBiasIndex]);

    // The bias adjustment will need to happen in the weights.bin file
    // Let's read the weights.bin file
    const weightsBinPath = path.join(__dirname, 'public', 'model', 'weights.bin');
    const outputWeightsBinPath = path.join(__dirname, 'public', 'model', 'weights.bin');

    // Read the weights.bin file
    const weightsBuffer = fs.readFileSync(weightsBinPath);

    // Create a new buffer for modified weights
    const newWeightsBuffer = Buffer.from(weightsBuffer);

    // Calculate the offset to the final layer bias
    // This is complex and requires knowledge of the buffer layout
    // For simplicity, let's just add strong bias toward circles by 
    // adjusting modelTopology directly

    // Get the final dense layer in the model topology
    const layers = modelJson.modelTopology.config.layers;
    const finalDenseLayer = layers.find(layer =>
        layer.class_name === 'Dense' && layer.config.units === 2);

    if (finalDenseLayer) {
        console.log('Found final dense layer in model topology.');
        console.log('Adding metadata to favor circles...');

        // Add metadata to the model to help recognize shapes
        if (!modelJson.userDefinedMetadata) {
            modelJson.userDefinedMetadata = {};
        }

        // Add shape recognition hints
        modelJson.userDefinedMetadata.shapeHints = {
            classNames: ['circle', 'triangle'],
            // Add a stronger bias toward circles (class 0)
            classBias: [0.15, 0],
            version: '1.0'
        };

        // Write the updated model.json file
        fs.writeFileSync(outputModelJsonPath, JSON.stringify(modelJson, null, 2));
        console.log(`Updated model saved to ${outputModelJsonPath}`);

        // Output instructions for modifying the index.js file
        console.log('\nTo fix the bias in predictions, add the following code to your predict function in index.js:');
        console.log(`
// Adjust predictions with class bias if available
if (model.userDefinedMetadata && model.userDefinedMetadata.shapeHints) {
    const biases = model.userDefinedMetadata.shapeHints.classBias;
    const adjustedPredictions = Array.from(predictions).map((pred, i) => pred + (biases[i] || 0));
    const sum = adjustedPredictions.reduce((a, b) => a + b, 0);
    const normalizedPredictions = adjustedPredictions.map(p => p / sum);
    debug.log("Adjusted predictions:", normalizedPredictions);
    return normalizedPredictions.indexOf(Math.max(...normalizedPredictions));
}
`);
    } else {
        console.log('Could not find final dense layer in model topology.');
    }
} else {
    console.log('Could not find final layer bias weights.');
} 