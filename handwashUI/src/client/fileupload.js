import React, { useRef, useState } from 'react';
import axios from 'axios';


// axios.defaults.baseURL = process.env.apiURL
// const apiURL = process.env.apiURL

function FileUpload() {

    const [file, setFile] = useState('');
    const [data, getFile] = useState({ name: "", path: "" });
    const [progress, setProgress] = useState(0);

    const el = useRef();

    const handleChange = (e) => {
        setProgress(0)
        const file = e.target.files[0]
        console.log(file);
        setFile(file)
    }

    const uploadFile = () => {
        const formData = new FormData();
        formData.append('file', file)
        axios.post(`http://0.0.0.0:8080/api/upload`, formData, {
            onUploadProgress: (ProgressEvent) => {
                let progress = Math.round(ProgressEvent.loaded / ProgressEvent.total * 100) + '%';
                setProgress(progress)
            }
        }).then(res => {
            console.log(res);
            getFile({ name: res.data.name, path: 'http://0.0.0.0:8080' + res.data.path, py: res.data.py })
        }).catch(err => console.log(err))
    }

    return (
        <div className="centered">
            <div>
                <input className="mt-2" type="file" ref={el} onChange={handleChange} />
                <div className="progessBar mt-2" style={{ width: progress }}>{progress}</div>
                <button onClick={uploadFile} className="upbutton mt-2">Upload</button>
            </div>
            <hr />
            {data.path &&
                <div>
                    <button type="button" className="btn btn-primary" onClick={() => window.location.reload(false)}>Upload Complete</button>
                </div>}
        </div>
    );
}

export default FileUpload;
