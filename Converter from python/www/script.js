import * as DICTIONARY from '/dictionary.js';
const ENCODING_LENGTH = 20;
const POST_COMMENT_BTN = document.getElementById('post');
const COMMENT_TEXT = document.getElementById('comment');
const COMMENTS_LIST = document.getElementById('commentsList');
const PROCESSING_CLASS = 'processing';

// Store username of logged in user. Right now you have no author, so default to Anonymous until known.
var currentUserName = 'Anonymous';

/** 
 * Function to handle the processing of submitted comments.
 **/
function handleCommentPost() {
  if (! POST_COMMENT_BTN.classList.contains(PROCESSING_CLASS)) {
    POST_COMMENT_BTN.classList.add(PROCESSING_CLASS);
    COMMENT_TEXT.classList.add(PROCESSING_CLASS);
    
    let currentComment = COMMENT_TEXT.innerText;
    let lowercaseSentenceArray = currentComment.toLowerCase().replace(/[^\w\s]/g, ' ').split(' ');
    let li = document.createElement('li');
    
    loadAndPredict(tokenize(lowercaseSentenceArray), li).then(function() {

      POST_COMMENT_BTN.classList.remove(PROCESSING_CLASS);
      COMMENT_TEXT.classList.remove(PROCESSING_CLASS);
      
      let p = document.createElement('p');
      p.innerText = COMMENT_TEXT.innerText;
      
      let spanName = document.createElement('span');
      spanName.setAttribute('class', 'username');
      spanName.innerText = currentUserName;
      
      let spanDate = document.createElement('span');
      spanDate.setAttribute('class', 'timestamp');
      let curDate = new Date();
      spanDate.innerText = curDate.toLocaleString();
      
      li.appendChild(spanName);
      li.appendChild(spanDate);
      li.appendChild(p);
      COMMENTS_LIST.prepend(li);

      COMMENT_TEXT.innerText = '';
    });
  }
}

POST_COMMENT_BTN.addEventListener('click', handleCommentPost);


// Set the URL below to the path of the model.json file you uploaded.
const MODEL_JSON_URL = 'https://cdn.glitch.global/24f130c6-c58a-4457-9186-30a6cca84f68/model.json?v=1688192740259';
const SPAM_THRESHOLD = 0.75;
var model = undefined;

async function loadAndPredict(inputTensor, domComment) {
  // Load the model.json and binary files you hosted. Note this is 
  // an asynchronous operation so you use the await keyword
  if (model === undefined) {
    model = await tf.loadLayersModel(MODEL_JSON_URL);
  }
  
  // Once model has loaded you can call model.predict and pass to it
  // an input in the form of a Tensor. You can then satore the result.
  var results = await model.predict(inputTensor);
  
  // Print the result to the console for us to inspect.
  results.print();

  results.data().then((dataArray)=>{
    if (dataArray[1] > SPAM_THRESHOLD) {
      domComment.classList.add('spam');
    }
  })
}

loadAndPredict(tf.tensor([[1,3,12,18,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]));


function tokenize(wordArray) {
  let returnArray = [DICTIONARY.START];
  for (var i = 0; i < wordArray.length; i++) {
    let encoding = DICTIONARY.LOOKUP[wordArray[i]];
    returnArray.push(encoding === undefined ? DICTIONARY.UNKNOWN : encoding);
  }
  
  while (i < ENCODING_LENGTH - 1) {
    returnArray.push(DICTIONARY.PAD);
    i++;
  }
  
  console.log([returnArray]);
  return tf.tensor([returnArray]);
}


function handleCommentPost() {
  if (! POST_COMMENT_BTN.classList.contains(PROCESSING_CLASS)) {
    POST_COMMENT_BTN.classList.add(PROCESSING_CLASS);
    COMMENT_TEXT.classList.add(PROCESSING_CLASS);
    
    let currentComment = COMMENT_TEXT.innerText;
    let lowercaseSentenceArray = currentComment.toLowerCase().replace(/[^\w\s]/g, ' ').split(' ');
    let li = document.createElement('li');
    
    loadAndPredict(tokenize(lowercaseSentenceArray), li).then(function() {

      POST_COMMENT_BTN.classList.remove(PROCESSING_CLASS);
      COMMENT_TEXT.classList.remove(PROCESSING_CLASS);
      
      let p = document.createElement('p');
      p.innerText = COMMENT_TEXT.innerText;
      
      let spanName = document.createElement('span');
      spanName.setAttribute('class', 'username');
      spanName.innerText = currentUserName;
      
      let spanDate = document.createElement('span');
      spanDate.setAttribute('class', 'timestamp');
      let curDate = new Date();
      spanDate.innerText = curDate.toLocaleString();
      
      li.appendChild(spanName);
      li.appendChild(spanDate);
      li.appendChild(p);
      COMMENTS_LIST.prepend(li);

      COMMENT_TEXT.innerText = '';
    });
  }
}