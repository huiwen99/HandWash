const express = require('express');
const os = require('os');
const fileUpload = require('express-fileupload');
const cors = require('cors')
const app = express();
const fs = require('fs');
var path = require('path');



app.use(cors())
app.use(fileUpload());

// kill $(lsof -t -i:8080)

app.get('/api/processFile', (req, res) => {
    var fileName = req.query.fileName
    var spawn = require("child_process").spawn;
    var process = spawn('python3', ["./machine_learning/predict.py",
        fileName]);
    process.stdout.on('data', (data) => {
        res.send({ mloutput: data.toString() });
    })
})


app.get('/api/getFileName', (req, res) => {
    var outputFileList = []
    var outputArr = []
    const directoryPath = path.join(__dirname, '../../uploads/');
    //passsing directoryPath and callback function
    fs.readdir(directoryPath, function (err, files) {
        //handling error
        if (err) {
            return console.log('Unable to scan directory: ' + err);
        }
        //listing all files using forEach
        files.forEach(function (file) {
            outputFileList.push(file)
        });

        var i;
        for (i = 0; i < outputFileList.length; i++) {
            var tempObj = {
                name: outputFileList[i],
                value: outputFileList[i],
                isChecked: false
            }
            outputArr.push(tempObj)
        }
        res.send({ allFilesName: outputArr });
    });
})

app.post('/api/upload', (req, res) => {

    if (!req.files) {
        return res.status(500).send({ msg: "file is not found" })
    }

    const myFile = req.files.file;

    myFile.mv(`./uploads/${myFile.name}`, function (err) {
        if (err) {
            console.log(err)
            return res.status(500).send({ msg: "error" });
        }

        var params = [];
        params.push(myFile.name)
        res.send({ file: myFile.name, path: `/${myFile.name}`, ty: myFile.type});

    });
})

app.listen(process.env.PORT || 8080, () => console.log(`Listening on port ${process.env.PORT || 8080}!`));
