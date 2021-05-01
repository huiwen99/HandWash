import React, { Component } from 'react'
import ReactPlayer from "react-player";
import './app.css';

export default class videoplayer extends Component {
    render() {
        const videoFilePath = `../../uploads/${this.props.videoUrl}`
        // if (this.props.urlVideo = !undefined) {
            return (
                <div>
                    <ReactPlayer
                        className='react-player'
                        url= {videoFilePath}
                        controls={true}
                    />
                </div>
            )
        // }   
    }
}
