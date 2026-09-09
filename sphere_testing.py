import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response, request, jsonify

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
STREAM_PORT = 5000
STREAM_JPEG_QUALITY = 80

# ----------------------------------------------------------------
# Tunable detection parameters, adjustable live via sliders on the
# web page. This is the single source of truth the detection loop
# reads from each frame - the sliders just write into it.
# ----------------------------------------------------------------
params_lock = threading.Lock()
detect_params = {
    'h_low': 5,
    's_low': 170,
    'v_low': 166,
    'h_high': 11,
    's_high': 255,
    'v_high': 255,
    'min_circularity': 0.65,     # contour circularity filter (0-1, x100 on slider)
    'blob_min_circularity': 0.35,  # SimpleBlobDetector circularity filter (0-1, x100 on slider)
}

stream_lock = threading.Lock()
latest_jpeg = None
isRunning = True


# ----------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------
flask_app = Flask(__name__)

PAGE_HTML = """
<html>
<head>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; display:flex; }
  #video { flex:1; display:flex; align-items:center; justify-content:center; background:#000; }
  #video img { max-width:100%; max-height:100vh; width:auto; height:auto; object-fit:contain; }
  #controls { width:280px; padding:16px; box-sizing:border-box; overflow-y:auto; }
  #controls label { display:block; margin-top:12px; font-size:13px; }
  #controls input[type=range] { width:100%; }
  .val { float:right; opacity:0.7; }
</style>
</head>
<body>
  <div id="video"><img src="/stream"></div>
  <div id="controls">
    <h3>Detection tuning</h3>
    <div id="sliders"></div>
  </div>

<script>
const sliderDefs = [
  {key: 'h_low',  label: 'Hue low',  min: 0,   max: 179, step: 1},
  {key: 'h_high', label: 'Hue high', min: 0,   max: 179, step: 1},
  {key: 's_low',  label: 'Sat low',  min: 0,   max: 255, step: 1},
  {key: 's_high', label: 'Sat high', min: 0,   max: 255, step: 1},
  {key: 'v_low',  label: 'Val low',  min: 0,   max: 255, step: 1},
  {key: 'v_high', label: 'Val high', min: 0,   max: 255, step: 1},
  {key: 'min_circularity',      label: 'Contour circularity min', min: 0, max: 100, step: 1, scale: 0.01},
  {key: 'blob_min_circularity', label: 'Blob circularity min',    min: 0, max: 100, step: 1, scale: 0.01},
];

const container = document.getElementById('sliders');

async function loadInitial() {
  const res = await fetch('/params');
  const current = await res.json();

  sliderDefs.forEach(def => {
    const scale = def.scale || 1;
    const raw = Math.round(current[def.key] / scale);

    const wrap = document.createElement('label');
    wrap.innerHTML = `${def.label} <span class="val" id="val-${def.key}">${current[def.key]}</span>`;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = def.min;
    input.max = def.max;
    input.step = def.step;
    input.value = raw;

    input.addEventListener('input', () => {
      const scaled = input.value * scale;
      document.getElementById(`val-${def.key}`).textContent =
        scale === 1 ? scaled : scaled.toFixed(2);
      fetch('/params', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({[def.key]: scaled})
      });
    });

    wrap.appendChild(document.createElement('br'));
    wrap.appendChild(input);
    container.appendChild(wrap);
  });
}

loadInitial();
</script>
</body>
</html>
"""


@flask_app.route('/')
def index():
    return PAGE_HTML


@flask_app.route('/stream')
def stream_route():
    def generate():
        while isRunning:
            with stream_lock:
                jpeg = latest_jpeg
            if jpeg is not None:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
                )
            time.sleep(0.02)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/params', methods=['GET', 'POST'])
def params_route():
    if request.method == 'POST':
        update = request.get_json(force=True)
        with params_lock:
            for key, value in update.items():
                if key in detect_params:
                    detect_params[key] = value
        return jsonify({'ok': True})

    with params_lock:
        return jsonify(dict(detect_params))


def web_stream_loop():
    flask_app.run(host='0.0.0.0', port=STREAM_PORT, threaded=True, use_reloader=False)


# ----------------------------------------------------------------
# Detection loop
# ----------------------------------------------------------------
def make_blob_detector(min_circularity):
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = False
    detector_params.filterByCircularity = True
    detector_params.minCircularity = min_circularity
    detector_params.filterByInertia = False
    detector_params.filterByConvexity = False
    detector_params.filterByColor = True
    detector_params.blobColor = 255
    return cv2.SimpleBlobDetector_create(detector_params)


def detection_loop():
    global latest_jpeg

    picamera = Picamera2()
    picamera.configure(picamera.create_preview_configuration())
    picamera.start()

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]

    # Cache the blob detector and only rebuild it when its circularity
    # setting actually changes - SimpleBlobDetector_create isn't free,
    # and this runs every frame.
    cached_blob_circularity = None
    detector = None

    try:
        while isRunning:
            with params_lock:
                p = dict(detect_params)

            if detector is None or p['blob_min_circularity'] != cached_blob_circularity:
                detector = make_blob_detector(p['blob_min_circularity'])
                cached_blob_circularity = p['blob_min_circularity']

            frame = picamera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_orange = np.array([p['h_low'], p['s_low'], p['v_low']])
            upper_orange = np.array([p['h_high'], p['s_high'], p['v_high']])

            mask = cv2.inRange(hsvFrame, lower_orange, upper_orange)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            keypoints = detector.detect(mask)

            circular_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > p['min_circularity']:
                    circular_contours.append((contour, circularity))

            frame = cv2.drawKeypoints(
                frame,
                keypoints,
                None,
                (0, 255, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv2.putText(frame, f'blobs: {len(keypoints)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if keypoints:
                best_keypoint = max(keypoints, key=lambda kp: kp.size)
                kpX, kpY = int(best_keypoint.pt[0]), int(best_keypoint.pt[1])
                cv2.line(frame, (frame.shape[1] // 2, frame.shape[0] // 2), (kpX, kpY), (0, 0, 255), 2)

            if circular_contours:
                biggest_contour, best_circularity = max(circular_contours, key=lambda c: cv2.contourArea(c[0]))
                contour_area = cv2.contourArea(biggest_contour)

                if contour_area > 0:
                    moments = cv2.moments(biggest_contour)

                    if moments['m00'] != 0:
                        cX = int(moments['m10'] / moments['m00'])
                        cY = int(moments['m01'] / moments['m00'])
                        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                        cv2.drawContours(frame, [biggest_contour], -1, (0, 255, 0), 2)
                        cv2.putText(frame, f'circularity: {best_circularity:.2f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ok, encoded = cv2.imencode('.jpg', frame, encode_params)
            if ok:
                with stream_lock:
                    latest_jpeg = encoded.tobytes()

    finally:
        picamera.stop()


def main():
    global isRunning

    stream_thread = threading.Thread(target=web_stream_loop, daemon=True)
    stream_thread.start()
    print(f"Open http://<pi-ip>:{STREAM_PORT} to view + tune")

    try:
        detection_loop()
    except KeyboardInterrupt:
        pass
    finally:
        isRunning = False
        with params_lock:
            final_params = dict(detect_params)
        print("\nFinal detection params:")
        for key, value in final_params.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()