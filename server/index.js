const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const port = 3000;

// Use CORS middleware
app.use(cors());

app.get('/predictions', (req, res) => {
    const files = ['prediction.json', 'prediction-avax.json'];
    const filePathBase = path.join(__dirname, '..', 'data');

    const readFiles = files.map(file => {
        return new Promise((resolve, reject) => {
            const filePath = path.join(filePathBase, file);
            fs.readFile(filePath, 'utf8', (err, data) => {
                if (err) {
                    console.error(`Error reading the file ${file}`, err);
                    return reject(err);
                }

                try {
                    const jsonData = JSON.parse(data);
                    resolve(jsonData);
                } catch (err) {
                    console.error(`Error parsing JSON from file ${file}`, err);
                    reject(err);
                }
            });
        });
    });

    Promise.all(readFiles)
        .then(results => {
            res.json(results);
        })
        .catch(err => {
            res.status(500).json({ error: 'Internal Server Error' });
        });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
