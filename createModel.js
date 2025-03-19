const tf = require('@tensorflow/tfjs-node');

const model = tf.sequential();

// First convolutional layer
model.add(tf.layers.conv2d({
    inputShape: [28, 28, 3],  // Changed to match our input shape (RGB)
    filters: 32,
    kernelSize: [3, 3],
    activation: 'relu'
}));

// Max pooling
model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2]
}));

// Second convolutional layer
model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: [3, 3],
    activation: 'relu'
}));

// Max pooling
model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2]
}));

// Flatten the output
model.add(tf.layers.flatten());

// Dense layers
model.add(tf.layers.dense({
    units: 128,
    activation: 'relu'
}));

model.add(tf.layers.dropout(0.5));

model.add(tf.layers.dense({
    units: 2,  // 2 classes: circle and triangle
    activation: 'softmax'
}));

// Compile the model
model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

module.exports = model;




