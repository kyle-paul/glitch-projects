// notification when the model is loaded
const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js';

// Grab a reference to the MNIST input values (pixel data)
const INPUTS = TRAINING_DATA.inputs

// Grab a reference to the MNIST output values (labels)
const OUTPUTS = TRAINING_DATA.outputs

// Shuffle two arrays to remove any order, but the so in the same way
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Funcion to take Tensor and normalize the values
function normalize(tensor, min, max) {
    const result = tf.tidy(function() {
      const MIN_VALUES = tf.scalar(min);
      const MAX_VALUES = tf.scalar(max);
      const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
      const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
      const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
      return NORMALIZED_VALUES;
    });
    return result;
  }

// Input features Array is 2 dimensional -> normalize
const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);

// Output faetures Array is 1 dimensional -> one-hot-encoding
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Create and define a model architecture
const model = tf.sequential()

// CNN
model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],    // grayscale -> one color channel
    filters: 10,
    kernelSize: 3,              // Square filter of 3 by 3. Could also specify rectangle e.g. [2,3]
    strides: 1,
    padding: 'same',
    activation: 'relu'
}));

model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2
}));

model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    strides: 1,
    padding: 'same',
    activation: 'relu'
}));

model.add(tf.layers.maxPooling2d ({
    poolSize: 2,
    strides: 2,
}));

// MLP
model.add(tf.layers.flatten());
model.add(tf.layers.dense({units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
model.summary();

train();
// Train function
async function train () {
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1]);
    let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {
        shuffle: true,
        validationSplit: 0.15,
        epochs: 10,
        batchSize: 256,
        callbacks: {onEpochEnd: logProgress}
    });

    RESHAPED_INPUTS.dispose();
    OUTPUTS_TENSOR.dispose();
    INPUTS_TENSOR.dispose();
    evaluate();
} 


// Map the output index to label
const LOOKUP = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Scandel', 'Shirt', 'Sneaker', 'Bag', 'Ankel Boot'];
const PREDICTION_ELEMENT = document.getElementById('prediction');

function evaluate() {
    // select randomly froma all example inputs
    const OFFSET = Math.floor((Math.random() * INPUTS.length));
    let answer = tf.tidy(function() {
        let newInput = tf.normalize(tf.tensor1d(INPUTS[OFFSET]), 0, 255);
        let output = model.predict(newInput.reshape([1, 28, 28, 1])); // eexpandDims to convert 1d to 2d
        output.print();
        return output.squeeze().argMax();
    });

    answer.array().then(function(index) {
        PREDICTION_ELEMENT.innerText = LOOKUP[index];
        PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');
        answer.dispose();
        drawImage(INPUTS[OFFSET]);
    });
}

//  Draw canvas
const CANVAS = document.getElementById('canvas');
function drawImage(digit) {
  digit = tf.tensor(digit, [28, 28]).div(255);
  tf.browser.toPixels(digit, CANVAS);
  setTimeout(evaluate, interval);
}

// logProgess
function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
}

// interval
var interval = 2000;
const RANGER = document.getElementById('ranger');
const DOM_SPEED = document.getElementById('domSpeed');

// When user drags slider -> update the interval
RANGER.addEventListener('input', function(e){
    interval = this.value;
    DOM_SPEED.innerText = 'Change the speed of classification Currently:' + interval + 'ms';
});
