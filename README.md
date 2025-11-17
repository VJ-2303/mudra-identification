# Mudra Detection System

A real-time hand gesture (mudra) detection system using OpenCV and MediaPipe that recognizes classical Indian dance mudras from webcam input.

## Features

- **Real-time Detection**: Processes webcam feed to detect hand mudras in real-time
- **25+ Mudras Supported**: Recognizes over 25 classical mudras including:
  - Pataka Mudra
  - Tripataka Mudra
  - Ardha Chandra Mudra
  - Kartari Mukham Mudra
  - Mukula Mudra
  - Hamsasya Mudra
  - Bhramara Mudra
  - Chatura Mudra
  - And many more...
- **Visual Feedback**: Displays detected mudra name on screen with color-coded status
- **Hand Landmark Visualization**: Shows MediaPipe hand landmarks overlaid on the video feed

## Requirements

```
python 3.7+
opencv-python (cv2)
mediapipe
```

## Installation

1. Clone or download this repository
2. Install required packages:

```bash
pip install opencv-python mediapipe
```

## Usage

### Running the Application

```bash
python smallcode.py
```

or

```bash
python app.py
```

### Controls

- The webcam feed will open automatically
- Perform mudras in front of the camera
- Press **'q'** to quit the application

## Project Structure

```
├── main.py          # Main application with all mudra detection logic
├── app.py               # Alternative implementation
└── README.md            # This file
```

## How It Works

1. **Hand Landmark Detection**: Uses MediaPipe Hands to detect 21 hand landmarks
2. **Distance Normalization**: Calculates normalized distances between landmarks (scale-invariant)
3. **Finger Straightness Check**: Determines if fingers are straight or bent using geometric ratios
4. **Mudra Recognition**: Applies specific rules for each mudra based on:
   - Finger positions (straight/bent)
   - Distance between fingertips
   - Thumb positioning
   - Finger angles and orientation
5. **Priority-based Matching**: Checks mudras in priority order to avoid false positives

## Technical Details

### Key Functions

- `is_finger_straight()`: Checks if a finger is straight using MCP→PIP→TIP distance ratios
- `compute_distance_tables()`: Pre-computes all pairwise distances between landmarks
- `get_scale_ref()`: Calculates palm width for scale-invariant detection
- Individual mudra detection functions (e.g., `is_pataka_mudra()`, `is_mukula_mudra()`)

### Detection Algorithm

The system uses a combination of:
- **Geometric constraints**: Distance ratios between fingertips
- **Angle measurements**: Relative orientation of fingers
- **Bounding box analysis**: Clustering of fingertips (for mudras like Mukula)
- **Threshold-based classification**: Adjustable thresholds for different hand sizes

## Supported Mudras

| Mudra Name | Description |
|------------|-------------|
| Pataka | All fingers straight, thumb tucked |
| Tripataka | Index, middle, pinky straight; ring bent |
| Ardha Chandra | All fingers straight in L-shape |
| Kartari Mukham | Index & middle extended, ring & pinky bent |
| Arala | Index bent touching thumb, others straight |
| Mukula | All fingertips converge in bud shape |
| Hamsasya | Thumb-index circle, others straight |
| Bhramara | Thumb-middle touching, index rolled, ring & pinky straight |
| Chatura | All fingers straight together, thumb inside |
| Kapitha | All fingers bent, thumb touching index/middle |
| ... | 15+ more mudras |

## Troubleshooting

### Mudra Not Detecting

1. **Lighting**: Ensure good lighting conditions
2. **Hand Position**: Keep hand clearly visible and centered
3. **Distance**: Maintain appropriate distance from camera (30-60 cm recommended)
4. **Background**: Use a plain background for better detection
5. **Mudra Form**: Ensure proper mudra formation as per classical definitions

### Performance Issues

- Close other applications using the webcam
- Reduce video resolution if needed
- Ensure sufficient CPU resources

## Configuration

You can adjust detection sensitivity by modifying thresholds in the code:

```python
# Example: Adjust finger straightness threshold
is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.90)
```

Lower threshold = more relaxed detection  
Higher threshold = stricter detection

## Contributing

Feel free to:
- Add new mudra detection functions
- Improve existing detection algorithms
- Optimize performance
- Report bugs or issues

## License

This project is open-source and available for educational and research purposes.

## Acknowledgments

- **MediaPipe**: Google's hand tracking solution
- **OpenCV**: Computer vision library
- Classical Indian dance mudra definitions and references

## Future Enhancements

- [ ] Two-hand mudra detection
- [ ] Gesture sequence recognition
- [ ] Mobile app version
- [ ] Training mode with visual guides
- [ ] Export detection data for analysis
- [ ] Support for regional mudra variations

---

**Note**: This system is designed for educational and artistic purposes. Detection accuracy may vary based on camera quality, lighting conditions, and individual hand characteristics.
