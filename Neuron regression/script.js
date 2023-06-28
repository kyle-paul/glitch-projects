// Import training dataset from link
import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js';

// Input feature pairs (House size, Number of bedrooms)
const INPUTS = TRAINING_DATA.inputs;
// Current listed house prices in dollars given their features above (target output values you want to predict)
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two array in the same way so iputs still match outputs indices
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array of Arrays needs 2D tensor to store
const INPUTS_TENSOR = tf.tensor2d(INPUTS)

/// Output can stay 1 dimensional
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
model.add(tf.layers.dense({inputShape: [2], units: 1}));
model.summary();
train();

async function train() {
    // Tuning the learning_rate that is most suitable for the data we are using
    const LEARNING_RATE = 0.01;
    
    // Compile the the model with defined learning_rate and loss function
    model.compile({
        optimizer: tf.train.sgd(LEARNING_RATE),
        loss: 'meanSquaredError',
    });

    // Finally do the training
    let results = await model.fit(FEATURE_RESULTS, NORMALIZED_VALUES, OUTPUTS_TENSOR, {
        validationSplit: 0.15,  // Take aside 15% of the data to use for validation testing
        shuffle: true,   // Ensure data is shuffled in case it was in an order
        batchSize: 64,   // Batch size is set to 64 as the number of samples is large
        epochs: 10       // Set the epochs (iterations) 10 times
    });

    // dispose to avoid memory leakage
    OUTPUTS_TENSOR.dispose();
    FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

    console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
    console.log("Average validation loss:" + Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));

    evaluate();
}

train()