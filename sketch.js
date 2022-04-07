let loadedData = null
let predictedDataRange = null
let tensorData = null
let model = null
let predictedData = null


let xSel
let ySel
let visorElement
let displayElement
let displayCanvas
let inputElement

const batchSize = 32;
const epochs = 100;

function createTFModel() {
    // Create a sequential model
    const model = tf.sequential()
    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1 }))

    // Add two more layers
    // model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}))
    // model.add(tf.layers.dense({units: 20, activation: 'sigmoid'}))
    
    // Add an output layer
    model.add(tf.layers.dense({ units: 1 }))
    return model
}

function trainModel() {
    const {inputs, labels} = tensorData
    let model = createTFModel()
    tfvis.show.modelSummary(visorElement.elt, model);
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError
    });

    return model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            visorElement.elt,
            ['loss'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    }).then(() => {
        return model
    })
}

function predict(model) {
    const {inputMax, inputMin, labelMin, labelMax} = tensorData
    
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
  
      const xs = tf.linspace(0, 1, 100)
      const preds = model.predict(xs.reshape([100, 1]))
  
      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin)
  
      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin)
  
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()]
    })

    const predictedPoints = Array.from(xs).map((val, i) => {
      return [val, preds[i]]
    });
  
    return predictedPoints
}

function predictLabel(model, input) {
    const {inputMax, inputMin, labelMin, labelMax} = tensorData

    const [xs, preds] = tf.tidy(() => {
  
        const xs = tf.tensor1d([input])
            .sub(inputMin)
            .div(inputMax.sub(inputMin))

        const preds = model.predict(xs)
    
        const unNormXs = xs
          .mul(inputMax.sub(inputMin))
          .add(inputMin)
    
        const unNormPreds = preds
          .mul(labelMax.sub(labelMin))
          .add(labelMin)
    
        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()]
    })
  
    return [xs[0], preds[0]]
}


function setupUI() {
    displayElement = createDiv()
    displayElement.position(650, 100)

    visorElement = createDiv()
    visorElement.position(650, 500)

    let canvas = createCanvas(500, 500);
    canvas.position(100, 100)

    xSel = createSelect()
    xSel.size(80, 30)
    xSel.position(320, 620)

    ySel = createSelect()
    ySel.size(80, 30)
    ySel.position(10, 320)

    xSel.changed(() => {
        predictedDataRange = null
    })

    ySel.changed(() => {
        predictedDataRange = null
    })

    let trainButton = createButton('Train model')
    trainButton.position(100, 30)
    trainButton.mousePressed(() => {
        let xName = xSel.value()
        let yName = ySel.value()
        tensorData = loadedData.convertToTensor(xName, yName)
        trainModel().then(m => {
            model = m
            predictedDataRange = predict(model, tensorData)
        })
    })

    inputElement = createInput()
    inputElement.position(10, 700)

    inputElement.changed(() => {
        let inputNumber = Number.parseFloat(inputElement.value())
        predictedData = predictLabel(model, inputNumber)
    })

    for (let key of loadedData.keys) {
        xSel.option(key)
    }

    for (let key of loadedData.keys) {
        ySel.option(key)
    }
}

function setup() {
    loadedData = new Data()

    loadedData.loadJSON('./data/carsData.json').then(() => {
        setupUI()
    })
    // loadedData.loadCSV('./data/life_expectancy.csv').then(() => {
    //     setupUI()
    // })
}

function draw() {
    drawScatterplot()

    if (predictedDataRange) {
        drawPredictedRange(predictedDataRange)
    }
}

function drawScatterplot() {
    const border = 50

    background(220)

    let xName = xSel.value()
    let yName = ySel.value()

    let data = loadedData.data.filter(e => e[xName] !== null && e[yName] !== null)
    
    const xData = data.map(e => e[xName])
    const yData = data.map(e => e[yName])

    const xDataMin = min(xData)
    const xDataMax = max(xData)
    const yDataMin = min(yData)
    const yDataMax = max(yData)

    push()

    for (let element of data) {
        const x = map(element[xName], xDataMin, xDataMax, border, width-border)
        const y = map(element[yName], yDataMin, yDataMax, height-border, border)
        fill(255, 0, 0)
        circle(x, y, 10)
    }

    for (let child of displayElement.child()) {
        child.remove()
    }
    let elementData = createDiv()

    for (let element of data) {
        const x = map(element[xName], xDataMin, xDataMax, border, width-border)
        const y = map(element[yName], yDataMin, yDataMax, height-border, border)

        if (dist(x, y, mouseX, mouseY) < 5) {
            const decodedElement = loadedData.decodeElement(element)
            
            for (let key in decodedElement) {
                createDiv(key + ': ' + decodedElement[key]).parent(elementData)
            }
            break
        }
    }

    elementData.parent(displayElement)
    pop()
}

function drawPredictedRange() {
    let xName = xSel.value()
    let yName = ySel.value()

    let data = loadedData.data.filter(e => e[xName] !== null && e[yName] !== null)
    
    const xData = data.map(e => e[xName])
    const yData = data.map(e => e[yName])

    const xDataMin = min(xData)
    const xDataMax = max(xData)
    const yDataMin = min(yData)
    const yDataMax = max(yData)

    for (let element of predictedDataRange) {
        const x = map(element[0], xDataMin, xDataMax, 20, width-20)
        const y = map(element[1], yDataMin, yDataMax, height-20, 20)

        fill(0, 0, 255)
        square(x, y, 5)
    }
}

function drawPrediction() {
    let xName = xSel.value()
    let yName = ySel.value()

    let data = loadedData.data.filter(e => e[xName] !== null && e[yName] !== null)
    
    const xData = data.map(e => e[xName])
    const yData = data.map(e => e[yName])

    const xDataMin = min(xData)
    const xDataMax = max(xData)
    const yDataMin = min(yData)
    const yDataMax = max(yData)

    const x = map(predictedData[0], xDataMin, xDataMax, 20, width-20)
    const y = map(predictedData[1], yDataMin, yDataMax, height-20, 20)

    fill(0, 255, 0)
    square(x, y, 10)
}