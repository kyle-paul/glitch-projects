// notification when the model is loaded
const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

// Generate input numbers from 1 to 20 inclusive
const INPUTS = [];
for (let n = 1; n <= 20; n++) {
    INPUTS.push(n);
}

// Generate outputs that are simply each input multiplied by itself to generate some nonlinear data
const OUTPUTS = [];
for (let n = 0; n < INPUTS.length; n++) {
    OUTPUTS.push(INPUTS[n] * INPUTS[n]);
}

// Input feature Array of Arrays needs 2D tensor to store
const INPUTS_TENSOR = tf.tensor1d(INPUTS)

// Output can stay 1 dimensional
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS)

// Function to take a Tensor and normalize the values with respect to each column of values contained in that Tensor
function normalize(tensor, min, max) {
    // create an object with values is the returned values from a function
    const result = tf.tidy(function() {
        // Find the minimum value contained in the Tensor
        const MIN_VALUES = min || tf.min(tensor, 0);

        // Find the maximum value contained in the Tensor
        const MAX_VALUES = max || tf.max(tensor, 0);

        // Subtract the MIN_VALUE from everyt value in the Tensor and store the results in a new Tensor
        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

        // Calculate the range size of possible values by subtracting the MAX_VALUE and MIN_VALUE 
        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

        // Calculate the adjusted values divided by the range size as a new Tensor
        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

        return {NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES};
    });
    return result;
}

// Normalize all inputs feature arrays and then dispose of the original non-nomorlized Tensors.
const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log('Normalized values:');
FEATURE_RESULTS.NORMALIZED_VALUES.print();
console.log('Min_Values:')
FEATURE_RESULTS.MIN_VALUES.print();
console.log('Max_Values:')
FEATURE_RESULTS.MAX_VALUES.print();
INPUTS_TENSOR.dispose();

// Now actually train the model
const model = tf.sequential();

// We will use one dense layer with 1 neuron (unit) and an input of 2 input feature values (representing house size and number of rooms)
model.add(tf.layers.dense({inputShape: [1], units: 25, activation: 'relu'}));
model.add(tf.layers.dense({units: 5, activation: 'relu'}));
model.add(tf.layers.dense({units: 1}));
model.summary();

// Tuning the learning_rate that is most suitable for the data we are using
const LEARNING_RATE = 0.0001;
const OPTIMIZER = tf.train.sgd(LEARNING_RATE);

train();

async function train() {
    // Compile the the model with defined learning_rate and loss function
    model.compile({  
        optimizer: OPTIMIZER,
        loss: 'meanSquaredError',
    });

    // Finally do the training
    let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
        callbacks: {onEpochEnd: logProgress},
        shuffle: true,   // Ensure data is shuffled in case it was in an order
        batchSize: 2,   // Batch size is set to 64 as the number of samples is large
        epochs: 200       // Set the epochs (iterations) 10 times
    });

    // dispose to avoid memory leakage
    OUTPUTS_TENSOR.dispose();
    FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

    console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
    evaluate();
}

// define the evaluate function
function evaluate() {
    // Predict answer for single piece of data
    tf.tidy(function() {
        let newInput = normalize(tf.tensor1d([7]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);
        let output = model.predict(newInput.NORMALIZED_VALUES);
        output.print();
    });

    // dispose to avoid memory leakage
    FEATURE_RESULTS.MIN_VALUES.dispose();
    FEATURE_RESULTS.MAX_VALUES.dispose();
    model.dispose();

    console.log("--------------------------------");
    console.log(tf.memory().numTensors);
}

// define the logProgess function to track the loss after each epoch
function logProgress(epoch, logs) {
    console.log('Loss through epoch', epoch, Math.sqrt(logs.loss));
    if (epoch == 70) {
        OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
    }
}