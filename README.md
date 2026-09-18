# Armnin controller

This folder contains the modular controller for the Pi robot. `Armin_main.py`
is the entry point; it composes the typed configuration, live state, motor
controller, vision service, and WebSocket controller. `steelbar_powerful_bldc_driver.py`
is the hardware driver and should be treated as a dependency rather than a
place to put robot behaviour.

## Run

Run from this folder so the local driver can be found:

```bash
python Armin_main.py
```

The controller initializes four drive motors, starts Picamera2, and serves the
browser control socket on port `8765`. `Ctrl+C` stops the loops and commands all
motors to zero. Hardware imports require the Raspberry Pi environment; use
`python -m py_compile *.py` for a syntax-only check elsewhere.

## Module guide

- `Armin_config.py`: dataclasses for addresses, FOC calibration, HSV thresholds,
  camera timing, network settings, and control timing.
- `Armin_state.py`: mutable observations and commands shared by async tasks.
- `Armin_motors.py`: driver initialization, Armin's wheel equations, manual WASD
  handling, auto search/follow behaviour, and optional dribbler control.
- `Armin_vision.py`: camera configuration, exposure and white-balance controls, CLAHE brightness
  correction, contour selection, and debug overlay drawing.
- `Armin_websocket.py`: browser command parsing and JPEG broadcasting.
- `Armins_camera_code.py`: standalone interactive calibration utility.

## Coordinate and motor conventions

Vision rotates each frame counter-clockwise for the camera's physical mounting.
The returned offset is `(dx, dy)` from image centre, and the bearing calculation
uses `atan2(dx, -dy)`, making `0` degrees forward and positive angles clockwise.
Drive motors are ordered `FR, BR, BL, FL`; do not reorder the addresses or
calibration tuple without recalculating the wheel mapping.

## Browser messages

The WebSocket accepts JSON text messages:

| Type | Fields | Purpose |
| --- | --- | --- |
| `mode` | `mode: "auto"` or `"manual"` | Select driving mode. |
| `keys` | `keys: ["w", "a", ...]` | Replace held manual keys. Valid keys are `WASD`, `Q`, `E`, and `K`. |
| `params` | HSV/geometry field names | Change vision thresholds live. |
| `clahe` | `clip_limit`, `tile_grid` | Tune low-light brightness correction. |
| `debug_vector` | none | Print the current centre-to-ball vector. |
| `debug_motor` | `enabled: true/false` | Toggle rate-limited motor logging. |

Binary messages sent by the controller are JPEG camera frames. Multiple browser
clients may connect; each receives the same frame stream.

## Calibration

The main controller uses its exposure and white-balance values from
`CameraConfig` at startup. Its live HSV and CLAHE values still come from
`VisionConfig` and browser messages.
