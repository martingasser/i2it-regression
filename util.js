function parseSafe(value) {
    const parsed = Number.parseFloat(value)
    if (Number.isNaN(parsed)) {
        if (value == '')
            return null
        return value
    }
    return parsed
}

class Data {

    constructor() {
        this.stringToInt = {}
        this.intToString = {}
    }

    loadJSON(path) {
        return httpGet(path, 'text')
        .then(response => {
            let data = JSON.parse(response)
            this._encodeStrings(data)
            this.keys = Object.keys(data[0])
            this.data = data
        })
    }

    loadCSV(path) {
        return new Promise((resolve, reject) => {
            loadTable(path, 'csv', 'header', table => {
                let data = []
                for (let row of table.rows) {
                    let obj = {}
                    for (let columnName of table.columns) {
                        obj[columnName.trim()] = parseSafe(row.get(columnName))
                    }
                    data.push(obj)
                }
                this.data = this._encodeStrings(data)
                this.keys = Object.keys(data[0])
                resolve()
            })
        })
    }

    decodeElement(element) {
        const decoded = {}
        for (let key in element) {
            const value = element[key]
            decoded[key] = this.decodeValue(key, value)
        }
        return decoded
    }

    decodeValue(key, value) {
        if (this.intToString[key] !== undefined) {
            return this.intToString[key][value]
        } else {
            return value
        }
    }

    convertToTensor(xName, yName) {
        return tf.tidy(() => {
            const data = this.data
                .filter(e => e[xName] !== null && e[yName] !== null)
                .map(e => [e[xName], e[yName]])
            
            // Step 1. Shuffle the data
            tf.util.shuffle(data)
    
            // Step 2. Convert data to Tensor
            const inputs = data.map(d => d[0])
            const labels = data.map(d => d[1])
    
            const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
            const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
    
            //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
            const inputMax = inputTensor.max();
            const inputMin = inputTensor.min();
            const labelMax = labelTensor.max();
            const labelMin = labelTensor.min();
    
            const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
            const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
    
            return {
                inputs: normalizedInputs,
                labels: normalizedLabels,
                // Return the min/max bounds so we can use them later.
                inputMax,
                inputMin,
                labelMax,
                labelMin,
            }
        })
    }
    
    _encodeStrings(data) {
        for (let d of data) {
            for (let entry of Object.entries(d)) {
                let key = entry[0]
                let value = entry[1]
                if (typeof(value) == 'string') {
                    if (this.stringToInt[key] == undefined) {
                        this.stringToInt[key] = {}
                        this.intToString[key] = {}
                    }
                    if (this.stringToInt[key][value] == undefined) {
                        let index = Object.keys(this.stringToInt[key]).length
                        this.stringToInt[key][value] = index
                        this.intToString[key][index] = value
                    }
                    d[key] = this.stringToInt[key][value]
                }
            }
        }
        return data
    }

}
