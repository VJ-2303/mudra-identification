from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import time
import sys
import os

# Import all the mudra detection functions from main.py
# Import only the functions, not the module-level objects
import main
compute_distance_tables = main.compute_distance_tables
mudra_functions = main.mudra_functions

# Import mudra information database
from mudra_info import get_mudra_info, get_all_mudras

# Create our own MediaPipe instances for the web app
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)

# Global variables for shared state
current_mudra = "No Mudra Detected"
mudra_lock = threading.Lock()
camera_lock = threading.Lock()
camera_enabled = True  # Camera on/off state

# Camera configuration
CAMERA_INDEX = 0  # Change if needed
camera = None


def get_camera():
    """Initialize or return existing camera instance"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(CAMERA_INDEX)
            # Optimize camera settings for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        return camera


def generate_frames():
    """Generate video frames with mudra detection - optimized version"""
    global current_mudra, camera_enabled
    
    # Create hands instance once
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    frame_count = 0
    skip_frames = 1  # Process every frame for better responsiveness
    
    try:
        while True:
            # Check if camera is enabled
            if not camera_enabled:
                # Generate a black frame with text
                blank_frame = cv2.imread(os.path.join(os.path.dirname(__file__), 'static', 'camera_off.png'))
                if blank_frame is None:
                    # If image doesn't exist, create a black frame with text
                    blank_frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Camera Off", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)
                    cv2.putText(blank_frame, "Click 'Turn On Camera' to start", (120, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
                
                # Reset mudra status when camera is off
                with mudra_lock:
                    current_mudra = "Camera Off"
                
                # Encode and yield the blank frame
                ret, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
                continue
            
            cap = get_camera()
            success, frame = cap.read()
            
            if not success:
                print("Failed to read frame")
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process every nth frame to reduce CPU load
            if frame_count % skip_frames == 0:
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = hands.process(frame_rgb)
                frame_rgb.flags.writeable = True
                
                mudra_status = "No Mudra Detected"
                text_color = (0, 0, 255)  # Red
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw hand landmarks with optimized settings
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    landmarks = hand_landmarks.landmark
                    dist_table, ndist_table, scale_ref = compute_distance_tables(landmarks)
                    
                    # Check each mudra in priority order
                    for name, func in mudra_functions.items():
                        try:
                            detected = func(landmarks, dist_table, ndist_table, scale_ref)
                            if detected:
                                mudra_status = name
                                text_color = (0, 255, 0)  # Green
                                break
                        except Exception:
                            continue
                
                # Update global mudra status
                with mudra_lock:
                    current_mudra = mudra_status
            
            # Encode frame to JPEG with optimized quality
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to prevent overloading
            time.sleep(0.01)
    
    finally:
        hands.close()
        print("Video generation stopped")


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/current_mudra')
def get_current_mudra():
    """API endpoint to get current mudra status"""
    with mudra_lock:
        return jsonify({
            'mudra': current_mudra,
            'timestamp': time.time()
        })


@app.route('/mudra_list')
def get_mudra_list():
    """API endpoint to get list of all supported mudras"""
    return jsonify({
        'mudras': list(mudra_functions.keys()),
        'count': len(mudra_functions)
    })


@app.route('/mudra_info/<mudra_name>')
def mudra_info(mudra_name):
    """API endpoint to get detailed information about a specific mudra"""
    info = get_mudra_info(mudra_name)
    return jsonify(info)


@app.route('/images/<filename>')
def serve_image(filename):
    """Serve mudra images from the images folder"""
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    return send_from_directory(images_dir, filename)


@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    """API endpoint to toggle camera on/off"""
    global camera_enabled, camera
    
    camera_enabled = not camera_enabled
    
    # Release camera when turning off
    if not camera_enabled and camera is not None:
        with camera_lock:
            camera.release()
            camera = None
            print("📹 Camera released")
    
    return jsonify({
        'enabled': camera_enabled,
        'status': 'on' if camera_enabled else 'off'
    })


@app.route('/camera_status')
def camera_status():
    """API endpoint to get current camera status"""
    return jsonify({
        'enabled': camera_enabled,
        'status': 'on' if camera_enabled else 'off'
    })


def cleanup():
    """Cleanup resources"""
    global camera
    if camera is not None:
        camera.release()


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("🙏 MUDRA DETECTION WEB APPLICATION 🙏")
        print("=" * 60)
        print("\n✅ Starting Flask server...")
        print("📹 Initializing camera...")
        print(f"🌐 Open your browser and go to: http://localhost:5000")
        print("\n💡 Press Ctrl+C to stop the server\n")
        print("=" * 60)
        
        app.run(
            host='0.0.0.0',  # Accessible from network
            port=5000,
            debug=False,  # Set to True for development
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        cleanup()
        print("✅ Cleanup complete. Goodbye!")
