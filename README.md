# radar-camera-visual-sync
A project to synchronize and align camera and radar recordings and display data visually. Associated with the Centre of Excellence for Artificial Intelligence and Smart Mobility.

## Setting Up

Clone the repository and install dependencies
```sh
git clone https://github.com/hfyxin/fare-gate-visual-game.git
# inside folder
python3 -m pip install -r requirements.txt
```

### Required files
Add the following files in the specified directories
- ```/data/Controlled_test.avi```: Controlled test video file. Download from SharePoint [Here](https://mcmasteru365.sharepoint.com/:v:/r/sites/RTA2SmartDevicesandSensors/Shared%20Documents/RTA%202.4%20Gate%20Profiles%20Fare%20Evasion%20Detector/YoloModelTests/Synchronized%20Data/Controlled_test.avi?csf=1&web=1&e=ZyYBb4)
- ```/data/frames/*```: Add frames with timestamp filenames. Download most recent test outputs from Sharepoint [Here]( https://mcmasteru365.sharepoint.com/sites/RTA2SmartDevicesandSensors/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=nz0YaK&ovuser=44376307%2Db429%2D42ad%2D8c25%2D28cd496f4772%2Csiddih38%40mcmaster%2Eca&OR=Teams%2DHL&CT=1693294483085&clickparams=eyJBcHBOYW1lIjoiVGVhbXMtRGVza3RvcCIsIkFwcFZlcnNpb24iOiIyNy8yMzA3MDMwNzM0NiIsIkhhc0ZlZGVyYXRlZFVzZXIiOmZhbHNlfQ%3D%3D&cid=73287738%2D9e5c%2D4d99%2D8027%2D255dd2e86b26&FolderCTID=0x01200019C94817820BD942AD93905CD555A835&id=%2Fsites%2FRTA2SmartDevicesandSensors%2FShared%20Documents%2FRTA%202%2E4%20Gate%20Profiles%20Fare%20Evasion%20Detector%2Flogged%20data%2FTest%201%2Fframes%2Erar&parent=%2Fsites%2FRTA2SmartDevicesandSensors%2FShared%20Documents%2FRTA%202%2E4%20Gate%20Profiles%20Fare%20Evasion%20Detector%2Flogged%20data%2FTest%201)

## Usage
### Configuration
Refer to descriptions within [config.yaml](config.yaml) for configuration parameters. The default video config is optimized for video `Controlled_test.avi` with corresponding `Control_test1.json` radar data.
### Main Visualization
The main visualization can be run in video mode or frame mode.
#### Video Mode
Receives footage from video file. Ensure `mode = "video_mode"` at start of code in [rta2_cv2visual.py](rta2_cv2visual.py): 
```sh
python rta2_cv2visual.py
```
The openCV GUI contains trackbars which can be modified during playback for accurate adjustments.
#### Frame Mode
Receives footage from frame images. Ensure `mode = "frame_mode"` at start of code in [rta2_cv2visual.py](rta2_cv2visual.py):
```sh
python rta2_cv2visual.py
```
After running the visualization, trackbar values for gate area can be saved to YAML config based on response to prompt.
### Tools
Additional visualization tools for analysis purposes can be used
#### Radar_visualization
An animation of radar points in 3D scatter plot format (standard visual transforms to 2D perspective) with additional data.

Additional requirements: Make sure you have installed [FFmpeg](https://ffmpeg.org/download.html).
```sh
python tools/radar_visualization.py
```
Animation can be saved based on response to prompt at the end of animation.
#### tlv_scatter
Generate single scatter plot of all radar points, color coded by TLV type.
```sh
python tools/tlv_scatter.py
```
### Tests
To run tests, run as module from main directory
```sh
python -m tests.<script-name>
```
## Next Steps
- Merge with MQTT repository for live visualization
- YOLO fusion tracking addition
