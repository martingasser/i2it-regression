let loadedData = null

let xSel
let ySel
let visorElement
let displayElement
let displayCanvas

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
