// notification when the model is loaded
const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

// Load the training data
import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';


// Grab a reference to thee MNIST input values (pixel data)
const INPUTS = TRAINING_DATA.inputs;

// Grab a reference to the MNIST output values (labels)
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the training data
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature array is 1 dimensional
const INPUTS_TENSOR = tf.tensor2d(INPUTS)

// Output feature array is 1 dimensional and we have to use one-hot-encoding
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Now actually create and define a model architecture
const model = tf.sequential();

// We will use one dense layer with 1 neuron (unit) and an input of 2 input feature values (representing house size and number of rooms)
model.add(tf.layers.dense({inputShape: [784], units: 16, activation: 'relu'}));
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 16, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
model.summary();

// Train the model
train();
async function train() {
    // Compile the the model with defined learning_rate and loss function
    model.compile({  
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Finally do the training
    let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
        validationSplit: 0.2,
        shuffle: true,   // Ensure data is shuffled in case it was in an order
        batchSize: 64,   // Batch size is set to 64 as the number of samples is large
        epochs: 20,       // Set the epochs (iterations) 10 times
        callbacks: {onEpochEnd: logProgress}
    });

    // dispose to avoid memory leakage
    OUTPUTS_TENSOR.dispose();
    INPUTS_TENSOR.dispose();
    evaluate(); 
}

// Prediction
const PREDICTION_ELEMENT = document.getElementById('prediction');

function evaluate() {
    // select randomly froma all example inputs
    const OFFSET = Math.floor((Math.random() * INPUTS.length));
    let answer = tf.tidy(function() {
        let newInput = tf.tensor1d(INPUTS[OFFSET])
        let output = model.predict(newInput.expandDims()); // eexpandDims to convert 1d to 2d
        output.print();
        return output.squeeze().argMax();
    });

    answer.array().then(function(index) {
        PREDICTION_ELEMENT.innerText = index;
        PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');
        answer.dispose();
        drawImage(INPUTS[OFFSET]);
    });
}


function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
}

const CANVAS = document.getElementById('canvas');
const CTX = CANVAS.getContext('2d');


function drawImage(digit) {
    var imageData = CTX.getImageData(0, 0, 28, 28);
    
    for (let i = 0; i < digit.length; i++) {
      imageData.data[i * 4] = digit[i] * 255;      // Red Channel.
      imageData.data[i * 4 + 1] = digit[i] * 255;  // Green Channel.
      imageData.data[i * 4 + 2] = digit[i] * 255;  // Blue Channel.
      imageData.data[i * 4 + 3] = 255;             // Alpha Channel.
    }
  
    // Render the updated array of data to the canvas itself.
    CTX.putImageData(imageData, 0, 0);
  
    // Perform a new classification after a certain interval.
    setTimeout(evaluate, interval);
  }
  
  
  var interval = 2000;
  const RANGER = document.getElementById('ranger');
  const DOM_SPEED = document.getElementById('domSpeed');
  
  // When user drags slider update interval.
  RANGER.addEventListener('input', function(e) {
    interval = this.value;
    DOM_SPEED.innerText = 'Change speed of classification! Currently: ' + interval + 'ms';
  });