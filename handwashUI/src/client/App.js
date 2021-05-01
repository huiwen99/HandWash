import React, { Component } from 'react';
import './app.css';
import FileUpload from './fileupload';
import VideoPlayer from './videoplayer';
import { Dropdown, Button } from 'react-bootstrap';
import axios from 'axios';
import LoadingOverlay from 'react-loading-overlay';
import CheckBox from './checkbox';

// axios.defaults.baseURL = process.env.apiURL

export default class App extends Component {

  constructor() {
    super();
    this.state = {
      uploadedFiles: [],
      videoUrl: "",
      pythonOutput: [],
      missingSteps: [],
      isLoading: false
    };
  }

  componentDidMount() {
    this.getLocalFileNames();
  }

  getLocalFileNames() {
    axios.get('http://0.0.0.0:8080/api/getFileName', {
    }).then(res => {
      this.setState({
        uploadedFiles: res.data.allFilesName
      })
    }).catch(err => console.log(err))
  }

  handleDropdown(chosenFile) {
    this.setState((state) => {
      return { videoUrl: chosenFile }
    });
  }

  processSelect() {
    let allFiles = this.state.uploadedFiles.filter(name => name.isChecked === true);
    let chosenFiles = []
    allFiles.map(({ name, value }) => chosenFiles.push(`./uploads/${name}`))
    this.setState((state) => {
      return { isLoading: true }
    });
    axios.get('http://0.0.0.0:8080/api/processFile', {
      params: {
        fileName: chosenFiles
      }
    }).then(res => {
      let tempOut = res.data.mloutput
      tempOut = tempOut.slice(1, -2)
      tempOut = tempOut.trim()
      tempOut = tempOut.split(",")
      let tempList = []
      let tempPYList = []
      var i;
      for (i = 0; i < tempOut.length; i++) {
        var perFile = tempOut[i].split(':')[0]
        perFile = perFile.trim()
        perFile = perFile.trim("'")
        perFile = perFile.replace(/'/g, '')
        var perPred = tempOut[i].split(':')[1]
        perPred = perPred.trim()
        perPred = perPred.replace(/'/g, '')
        let tempObj = {
          FileName: perFile,
          Pred: perPred
        }
        tempList.push(tempObj)
      }

      var flags = [], PYoutput = [], l = tempList.length, i;
      for( i=0; i<l; i++) {
          if( flags[tempList[i].Pred]) continue;
          flags[tempList[i].Pred] = true;
          PYoutput.push(tempList[i].Pred);
      }
      var classKeys = ['Step 1','Step 2 Left','Step 2 Right','Step 3','Step 4 Left','Step 4 Right','Step 5 Left','Step 5 Right','Step 6 Left','Step 6 Right','Step 7 Left','Step 7 Right']

      classKeys = classKeys.filter(function(val){
        return (PYoutput.indexOf(val) == -1 ? true : false)
      })

      var i;
      var img;
      for (i = 0; i < classKeys.length; i++) {
        if(classKeys[i]=='Step 1'){
          img = '/public/steps/step1.png'
        }
        if(classKeys[i]=='Step 2 Left'){
          img = '/public/steps/step2l.png'
        }
        if(classKeys[i]=='Step 2 Right'){
          img = '/public/steps/step2r.png'
        }
        if(classKeys[i]=='Step 3'){
          img = '/public/steps/step3.png'
        }
        if(classKeys[i]=='Step 4 Left'){
          img = '/public/steps/step4l.png'
        }
        if(classKeys[i]=='Step 4 Right'){
          img = '/public/steps/step4r.png'
        }
        if(classKeys[i]=='Step 5 Left'){
          img = '/public/steps/step5l.png'
        }
        if(classKeys[i]=='Step 5 Right'){
          img = '/public/steps/step5r.png'
        }
        if(classKeys[i]=='Step 6 Left'){
          img = '/public/steps/step6l.png'
        }
        if(classKeys[i]=='Step 6 Right'){
          img = '/public/steps/step6r.png'
        }
        if(classKeys[i]=='Step 7 Left'){
          img = '/public/steps/step7l.png'
        }
        if(classKeys[i]=='Step 7 Right'){
          img = '/public/steps/step7r.png'
        }
        let missingSteps = {
          Pred: classKeys[i],
          Img: img
        }
        tempPYList.push(missingSteps)
      }
      this.setState({
        pythonOutput: tempList,
        missingSteps: tempPYList,
        isLoading: false
      })
    }).catch(err => console.log(err))
  }

  handleAllChecked = (event) => {
    let uploadedFiles = this.state.uploadedFiles
    uploadedFiles.forEach(uploadedFile => uploadedFile.isChecked = event.target.checked)
    this.setState({ uploadedFiles: uploadedFiles })
  }

  handleCheckChieldElement = (event) => {
    let uploadedFiles = this.state.uploadedFiles
    uploadedFiles.forEach(uploadedFile => {
      if (uploadedFile.value === event.target.value)
        uploadedFile.isChecked = event.target.checked
    })
    this.setState({ uploadedFiles: uploadedFiles })
  }

  renderTableHeader(e) {
    if(e=='r'){
      let header = Object.keys(this.state.pythonOutput[0])
      return header.map((key, index) => {
        return <th key={index}>{key.toUpperCase()}</th>
      })
    } else {
      let header = Object.keys(this.state.missingSteps[0])
      return header.map((key, index) => {
        return <th key={index}>{key.toUpperCase()}</th>
      })
    }

  }

  renderTableData(e) {
    if(e=="r"){
      return this.state.pythonOutput.map((eachFile, index) => {
        const { FileName, Pred } = eachFile //destructuring
        return (
          <tr key={index}>
            <td>{FileName}</td>
            <td>{Pred}</td>
          </tr>
        )
      })
    } else {
      return this.state.missingSteps.map((eachFile, index) => {
        const { Pred, Img } = eachFile //destructuring
        return (
          <tr key={index}>
            <td>{Pred}</td>
            <td>
              <img src={Img} width='200px' height='200px'></img>
            </td>
          </tr>
        )
      })
    }
  }

  render() {
    return (
      <div>
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
          </style>
        <header>
          <a href="./"><h1>Do The 12 Steps Hand Wash!</h1></a>
        </header>
        <div>
          <FileUpload />
        </div>
        <div style={{ backgroundColor: "white", justifyContent: "center" }}>
          <section id='section2'>
            <div className="dropdown" style={{ textAlign: 'center', backgroundColor: 'orange' }}>
              <h3 style={{ textAlign: 'left' }} >Video Player</h3>
              <Dropdown>
                <Dropdown.Toggle
                  variant="secondary btn-sm"
                  id="dropdown-basic">
                  Choose Your File
        </Dropdown.Toggle>
                <Dropdown.Menu style={{ backgroundColor: '#73a47' }}>
                  {this.state.uploadedFiles.map(({ name, value }) => (
                    <Dropdown.Item key={name} value={value} onSelect={() => { this.handleDropdown(value) }} >{name}</Dropdown.Item>
                  ))}
                </Dropdown.Menu>
              </Dropdown>
            </div>
          </section>
          <section id="section99">
            {this.state.videoUrl &&
              <div style={{ display: "flex", justifyContent: 'flex-end' }}>
                <VideoPlayer videoUrl={this.state.videoUrl} />
              </div>
            }
          </section>
          <h1>Select Video Files for Classification</h1>
          <input type="checkbox" onClick={this.handleAllChecked} value="checkedall" /> Check / Uncheck All
        <ul>
            {
              this.state.uploadedFiles.map((uploadedFile) => {
                return (<CheckBox handleCheckChieldElement={this.handleCheckChieldElement}  {...uploadedFile} />)
              })
            }
          </ul>
          <Button style={{ float: 'right', display: 'inline-block' }} onClick={() => { this.processSelect() }} variant="warning">Process</Button>
        </div>
        <section id='section1' style={{ textAlign: 'center', backgroundColor: 'white' }}>
          {
            this.state.isLoading ? <LoadingOverlay style={{ justifyContent: 'center', display: 'flex' }}
              active={this.state.isLoading}
              spinner
            >
              <h2 style={{ color: 'red' }}>Processing Your Model....</h2>
            </LoadingOverlay> : <div />
          }
          {
            this.state.pythonOutput.length > 0 &&
            <div>
              <h3>Check Out Your Results!</h3>
            </div>
          }
          {this.state.pythonOutput.length > 0 &&
            <div className='outputpython'>
              <table id='results'>
                <tbody>
                  <tr>{this.renderTableHeader('r')}</tr>
                  {this.renderTableData('r')}
                </tbody>
              </table>
            </div>
          }
          {
            this.state.pythonOutput.length > 0 &&
            <div>
              <h3>Find Out Which Step(s) You're Missing!</h3>
            </div>
          }
          {this.state.pythonOutput.length > 0 &&
            <div className='outputpython'>
              <table id='results'>
                <tbody>
                  <tr>{this.renderTableHeader()}</tr>
                  {this.renderTableData()}
                </tbody>
              </table>
            </div>
          }
        </section>
      </div>
    );
  }
}