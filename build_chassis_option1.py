#!/usr/bin/env python3
"""
Lower the rear-wheel sumo chassis to 68 mm, cut the rear MH sensor window,
and add a perfboard roof at the side-window sill plus stepped motor seats.
It also adds a flat upper platform that screws onto the roof seats.

The source is the verified 200 x 200 x 80 mm rear-open chassis
``chassis_sensor_rear65.step``.  This script:

* removes the top 12 mm of the tub, keeping the existing front and side
  sensor windows fully framed (their top edges are at 65.90 and 60.50 mm);
* cuts one plain 31 x 7 mm window in the rear wall for the low MH
  infrared obstacle sensor (no divider, no header pocket, no wire notch;
  the board is 31 x 13 mm, 42 mm long with LEDs, pins pitched 2.4 mm);
* adds no sensor holder or case: the rear wall has only the window, and
  the user mounts the MH board;
* leaves the whole electronics bay open from the floor (z = 5) up to the
  new perfboard roof, so the two motor drivers and the wiring stay
  accessible from above;
* keeps the side roof seats/pilots; exports a separate flat upper platform
  that screws onto those seats (M3 self-tapping screws into the pilots);
* adds a perfboard frame at z = 40 mm, flush with the side-window sills:
  the 190 x 62.5 mm board takes the old roof opening, carries the two
  HC-SR04 side sensors (they stand on it, eyes at z = 50) and the ESP32;
  the frame only supports the board, with no battery/ESP/sensor cradle;
* adds a stepped motor seat in the rear floor under each JGB37-520:
  a 1.5 mm deep gearbox pocket, a 0.5 mm high body pad, the step at the
  gearbox/body joint, and four Ø3.5 mm clearance holes per motor
  (datasheet pattern, 30 x 24 mm); M3 x 10 screws from above, M3 nuts
  below the floor;
* exports the chassis and the upper platform; there are no separate motor mounts.

There are no battery or ESP holders: the perfboard is the electronics shelf.
The upper platform is a flat screwed-on cover, not a holder.
The user mounts those parts.  The MH board, LEDs, header, battery pack, ESP board
and the JGB37-520 motors/wheels in the renders are reference geometry
only; they are never exported.

Run with smlibpy from the uv cache:

    uv run --with 'smlibpy' python build_chassis_option1.py

Outputs:
    chassis_option1_light.step               chassis, STEP AP214
    chassis_option1_light.stl                chassis, watertight STL
    platform_upper.step                           upper platform, STEP AP214
    platform_upper.stl                            upper platform, watertight STL
    chassis_option1_viewer.html        interactive view
    renders/option1_*.png             verification renders
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from smlibpy import (
    Box,
    Cone,
    Cylinder,
    Plane,
    Polygon,
    Vertex,
    X,
    Y,
    export_step,
    export_stl,
    read_step,
    render,
    show,
)

# ---------------------------------------------------------------------------
# Parameters — change these and rerun; the source STEP is never modified.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SOURCE_STEP = HERE / "chassis_sensor_rear65.step"
OUT_STEP = HERE / "chassis_option1_light.step"
OUT_STL = HERE / "chassis_option1_light.stl"
PLATFORM_STEP = HERE / "platform_upper.step"
PLATFORM_STL = HERE / "platform_upper.stl"
VIEWER_HTML = HERE / "chassis_option1_viewer.html"
RENDER_DIR = HERE / "renders"

SOURCE_HEIGHT = 80.0
BODY_HEIGHT = 68.0             # mm, 12 mm lower; keeps all old windows framed
CUT_TOOL_HEIGHT = 120.0
STL_TOLERANCE = 0.02
STL_ANGLE = 15.0

FRONT_WINDOW_TOP_Z = 65.8952   # existing GY-VL53L0X window top
SIDE_WINDOW_TOP_Z = 60.5       # existing HC-SR04 window top
MIN_TOP_MATERIAL = 2.0         # mm of wall required above the front windows

# Reference wheels (Ø65 x 27) and JGB37-520 motors.  Never exported.
WHEEL_RADIUS = 32.5
WHEEL_WIDTH = 27.0
WHEEL_X = 86.5
WHEEL_Y = 67.5
WHEEL_Z = 24.5

MOTOR_Y = 60.5
MOTOR_Z = 24.5
GEARBOX_R = 18.5
GEARBOX_LEN = 24.0
BODY_R = 16.5
BODY_LEN = 22.7
CAP_R = 15.0
CAP_LEN = 3.5
BOSS_R = 6.0
BOSS_LEN = 6.0
SHAFT_R = 3.0
SHAFT_LEN = 15.5

# Stepped motor seat in the rear floor.  The gearbox drops into a pocket;
# the body sits on a raised pad.  Both seat at the same axis height, so the
# motor sits flat instead of rocking on the gearbox shoulder.  Four M3
# screws per motor (M3 x 10, from above, through Ø3.5 mm clearance holes),
# with M3 nuts below the floor; nothing about the holes is blind.
MOTOR_SEAT_DROP = 2.5
MOTOR_BAY_FLOOR_Z = 5.0
GEARBOX_POCKET_W = 39.4       # 1.2 mm per side on the Ø37 gearbox
GEARBOX_POCKET_DEPTH = 1.5
GEARBOX_POCKET_X0, GEARBOX_POCKET_X1 = 40.0, 64.5
BODY_PAD_W = 35.4             # 1.2 mm per side on the Ø33 body
BODY_PAD_HEIGHT = 0.5
BODY_PAD_X0, BODY_PAD_X1 = 13.3, 40.0

# Motor mount: 4 x M3 screws per motor (M3 x 10), inserted from above
# through Ø3.5 mm clearance holes (datasheet) with M3 nuts + washers
# below the floor.  Datasheet pattern for the JGB37-520:
# 30 mm lengthwise (X) x 24 mm widthwise (Y), centred on the body pad.
MOTOR_HOLE_R = 1.75           # Ø3.5 mm clearance
MOTOR_HOLE_X = 26.65          # body pad centre
MOTOR_HOLE_DX = 15.0          # +/- 15 mm (30 mm lengthwise)
MOTOR_HOLE_DY = 12.0          # +/- 12 mm (24 mm widthwise)

# MH module (31 x 13 x 1.6 mm board, 42 mm total length with LEDs).
# Reference only; the user mounts it.
BOARD_LEN = 31.0
BOARD_WIDTH = 13.0
BOARD_T = 1.6
BOARD_Z0 = 6.5                # nominal board bottom, 1.5 mm above the floor
BOARD_REAR_Y = 88.5           # LED ends extend from here to y = 99.5
BOARD_FRONT_Y = BOARD_REAR_Y - BOARD_LEN
BOARD_CENTER_Y = (BOARD_REAR_Y + BOARD_FRONT_Y) / 2.0
LED_R = 2.5
LED_LEN = 11.0                # 31 mm board + 11 mm LEDs = 42 mm total
LED_X = 3.2
LED_Z = BOARD_Z0 + BOARD_T + LED_R   # 10.5 mm optical axis
SENSOR_HOLE_Y = BOARD_FRONT_Y + 7.5  # 7.5 mm from the header end = y 65
WINDOW_W = 31.0
WINDOW_H = 7.0
WINDOW_Z = LED_Z
WINDOW_Y = 97.5
WALL_T = 5.2                  # cut through the 5 mm rear wall with margin

# ---------------------------------------------------------------------------
# Open electronics bay and roof seats.
#
# There is no lower deck any more: the bay runs clear from the floor
# (z = 5) up to the perfboard roof (z = 38.4).  The side roof seats and
# M3 pilots stay for the upper platform; the platform itself is printed
# and screwed on separately, it is not part of the chassis.
# ---------------------------------------------------------------------------
ROOF_SEAT_Y0, ROOF_SEAT_Y1 = -30.0, 33.5

LEDGE_BOTTOM = 61.0           # above the side window top (60.5)
LEDGE_TOP = 65.0              # roof seats; a cover would be flush at 68
LEDGE_W = 8.0

SCREW_PILOT_R = 1.25          # 2.5 mm pilot for M3 self-tapping screws
SCREWS = ((-91.0, -20.0), (-91.0, 20.0),
          (91.0, -20.0), (91.0, 20.0))

# ---------------------------------------------------------------------------
# Perfboard frame ("plaque perforée") at the side-window sill.
#
# The HC-SR04 side windows are 44 x 20 mm at z = 40..60.  The board top is
# at z = 40, flush with the sill, so both side sensors stand on it with
# their eyes at z = 50.  The board covers the old roof opening, 190 x
# 62.5 mm.  The frame is only a support: side/front ledges, a rear beam
# and a rear stop under the board.  No battery/ESP/sensor cradles.
# ---------------------------------------------------------------------------
PERF_BOARD_TOP = 40.0
PERF_BOARD_T = 1.6
PERF_BOARD_BOTTOM = PERF_BOARD_TOP - PERF_BOARD_T       # 38.4
PERF_BOARD_X0, PERF_BOARD_X1 = -95.0, 95.0
PERF_BOARD_LEN = 62.5
# Dropped-in position: a little play in the 190 x 62.7 mm pocket so a cut
# board slides in and rests on the frame (the source bay face sits at
# y = -29.99, not exactly -30).
PERF_BOARD_Y0 = -29.9
PERF_BOARD_Y1 = PERF_BOARD_Y0 + PERF_BOARD_LEN           # 32.6
PERF_LEDGE_W = 8.0            # side support under the board
PERF_FRONT_W = 5.0
PERF_REAR_W = 5.0
PERF_LEDGE_Z0 = 35.0
PERF_LIP_Y0, PERF_LIP_Y1 = 32.7, 33.5                    # rear stop

# HC-SR04 side sensors (reference only): PCB 45 x 20 mm standing on the
# board, eyes Ø16 x 12 mm through the 44 x 20 mm side windows.
HC_PCB_T = 1.6
HC_PCB_LEN = 45.0
HC_PCB_H = 20.0
HC_PCB_X0 = 88.0
HC_EYE_R = 8.0
HC_EYE_LEN = 12.0
HC_EYE_DY = 13.0
HC_EYE_Z = 50.0

# Reference envelopes only (3 x 18650 pack, ESP32-S3 dev board) - visual
# only.  Both now sit on the perfboard; the board is user-supplied.
BATTERY_X0, BATTERY_X1 = -54.0, 24.0
BATTERY_Y0, BATTERY_Y1 = -29.5, 30.5
BATTERY_Z0 = PERF_BOARD_TOP + 0.1
BATTERY_T = 21.0
ESP_X0, ESP_X1 = 44.75, 73.25
ESP_Y0, ESP_Y1 = -29.5, 32.5
ESP_Z0 = PERF_BOARD_TOP + 0.1
ESP_T = 13.0


def bbox_tuple(shape):
    bb = shape.bbox()
    lo, hi = bb.min, bb.max
    return (lo.x, lo.y, lo.z), (hi.x, hi.y, hi.z)


def verify(shape, label: str, expected_top: float, **kwargs) -> None:
    lo, hi = bbox_tuple(shape)
    volume = shape.volume
    solids = len(shape.solids())
    voids = len(shape.voids())
    valid = shape.is_valid()
    problems = shape.check()

    print(f"\n{label}")
    print(f"  volume : {volume:,.3f} mm^3")
    print(f"  bbox   : ({lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f})"
          f" .. ({hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f})")
    print(f"  solids : {solids}   voids: {voids}")
    print(f"  faces  : {len(shape.faces())}   edges: {len(shape.edges())}"
          f"   vertices: {len(shape.vertices())}")
    print(f"  valid  : {valid}   check: {problems}")

    if solids != 1 or voids != 0:
        raise RuntimeError(f"{label}: expected one solid and no voids, got {solids}/{voids}")
    if not valid or problems:
        raise RuntimeError(f"{label}: invalid geometry: {problems}")
    if abs(hi[2] - expected_top) > 1e-6:
        raise RuntimeError(f"{label}: top Z is {hi[2]:.6f}, expected {expected_top}")
    footprint = kwargs.get("expected_footprint", 100.0)
    expected_bottom = kwargs.get("expected_bottom")
    if abs(hi[0] - footprint) > 1e-6 or abs(hi[1] - footprint) > 1e-6:
        raise RuntimeError(f"{label}: footprint changed: {hi[0]} x {hi[1]}")
    if expected_bottom is not None and abs(lo[2] - expected_bottom) > 1e-6:
        raise RuntimeError(f"{label}: bottom Z is {lo[2]:.6f}, expected {expected_bottom}")


def report(shape, label: str) -> None:
    """Short report for a separate part."""
    lo, hi = bbox_tuple(shape)
    print(f"\n{label}")
    print(f"  volume : {shape.volume:,.3f} mm^3")
    print(f"  bbox   : ({lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f})"
          f" .. ({hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f})")
    print(f"  solids : {len(shape.solids())}   voids: {len(shape.voids())}"
          f"   faces: {len(shape.faces())}")
    print(f"  valid  : {shape.is_valid()}   check: {shape.check()}")
    if len(shape.solids()) != 1 or not shape.is_valid() or shape.check():
        raise RuntimeError(f"{label}: invalid part")


def stl_stats(path: Path) -> dict:
    """Read a binary STL: triangle count, bbox, signed volume, open edges."""
    with path.open("rb") as fh:
        fh.read(80)
        (ntri,) = struct.unpack("<I", fh.read(4))
        raw = np.frombuffer(fh.read(ntri * 50), dtype=np.uint8).reshape(ntri, 50)
    tri = raw[:, 12:48].copy().view("<f4").reshape(ntri, 3, 3).astype(np.float64)

    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    volume = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)

    edges = np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0
    )
    edges = np.round(np.sort(edges, axis=1), 6)
    _, counts = np.unique(edges.reshape(-1, 6), axis=0, return_counts=True)
    open_edges = int(np.count_nonzero(counts != 2))

    pts = tri.reshape(-1, 3)
    return {
        "triangles": ntri,
        "min": pts.min(axis=0),
        "max": pts.max(axis=0),
        "volume": volume,
        "open_edges": open_edges,
    }


def sensor_reference():
    """MH module reference: flat board, LEDs rearward, header up at the front."""
    board = Box(BOARD_WIDTH, BOARD_LEN, BOARD_T).translate(
        (0.0, BOARD_CENTER_Y, BOARD_Z0 + BOARD_T / 2.0)
    )
    board = board - Cylinder(1.5, BOARD_T + 1.0).translate(
        (0.0, SENSOR_HOLE_Y, BOARD_Z0 + BOARD_T / 2.0)
    )
    led_r = (
        Cylinder(LED_R, LED_LEN)
        .rotate(X, 90)
        .translate((LED_X, BOARD_REAR_Y + LED_LEN / 2.0, LED_Z))
    )
    led_l = (
        Cylinder(LED_R, LED_LEN)
        .rotate(X, 90)
        .translate((-LED_X, BOARD_REAR_Y + LED_LEN / 2.0, LED_Z))
    )
    header = Box(8.0, 2.5, 2.5).translate(
        (0.0, BOARD_FRONT_Y + 2.5, BOARD_Z0 + BOARD_T + 1.2)
    )
    sensor = board + led_r + led_l + header
    for px in (-2.4, 0.0, 2.4):
        sensor = sensor + Cylinder(0.3, 8.5).translate(
            (px, BOARD_FRONT_Y + 2.5, BOARD_Z0 + BOARD_T + 4.2)
        )
    return board, led_r, led_l, header, sensor


def block(x0, x1, y0, y1, z0, z1):
    return Box(x1 - x0, y1 - y0, z1 - z0).translate(
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)
    )


def motor_pair(z=MOTOR_Z):
    gearbox_r = Cylinder(GEARBOX_R, GEARBOX_LEN).rotate(Y, 90).translate(
        (52.0, MOTOR_Y, z)
    )
    body_r = Cylinder(BODY_R, BODY_LEN).rotate(Y, 90).translate(
        (28.65, MOTOR_Y, z)
    )
    cap_r = Cylinder(CAP_R, CAP_LEN).rotate(Y, 90).translate(
        (15.55, MOTOR_Y, z)
    )
    boss_r = Cylinder(BOSS_R, BOSS_LEN).rotate(Y, 90).translate(
        (67.0, MOTOR_Y, z)
    )
    shaft_r = Cylinder(SHAFT_R, SHAFT_LEN).rotate(Y, 90).translate(
        (77.75, MOTOR_Y, z)
    )
    motor_r = gearbox_r + body_r + cap_r + boss_r + shaft_r
    return motor_r, motor_r.mirror(Plane.YZ)


def motor_seat():
    """Right-side stepped motor seat: body pad, gearbox pocket, M3 holes."""
    pad = block(
        BODY_PAD_X0, BODY_PAD_X1,
        MOTOR_Y - BODY_PAD_W / 2.0, MOTOR_Y + BODY_PAD_W / 2.0,
        MOTOR_BAY_FLOOR_Z - BODY_PAD_HEIGHT,
        MOTOR_BAY_FLOOR_Z + BODY_PAD_HEIGHT,
    )
    pocket = block(
        GEARBOX_POCKET_X0, GEARBOX_POCKET_X1,
        MOTOR_Y - GEARBOX_POCKET_W / 2.0, MOTOR_Y + GEARBOX_POCKET_W / 2.0,
        MOTOR_BAY_FLOOR_Z - GEARBOX_POCKET_DEPTH,
        MOTOR_BAY_FLOOR_Z + 0.5,
    )
    holes = [
        Cylinder(MOTOR_HOLE_R, BODY_HEIGHT).translate(
            (MOTOR_HOLE_X + dx, MOTOR_Y + dy, BODY_HEIGHT / 2.0)
        )
        for dx in (-MOTOR_HOLE_DX, MOTOR_HOLE_DX)
        for dy in (-MOTOR_HOLE_DY, MOTOR_HOLE_DY)
    ]
    return pad, pocket, holes


def perfboard_frame():
    """Support frame for the 190 x 62.5 mm perfboard at z = 40.

    Attached to the side walls (ledges), the front wall (ledge) and closed
    by a rear beam.  The board drops in from above and rests on the
    ledges/beam; the rear lip stops it sliding back.
    """
    side_l = block(-96.0, -95.0 + PERF_LEDGE_W, -31.0, 33.5,
                   PERF_LEDGE_Z0, PERF_BOARD_BOTTOM)
    side_r = block(95.0 - PERF_LEDGE_W, 96.0, -31.0, 33.5,
                   PERF_LEDGE_Z0, PERF_BOARD_BOTTOM)
    front = block(-96.0, 96.0, -31.0, -30.0 + PERF_FRONT_W,
                  PERF_LEDGE_Z0, PERF_BOARD_BOTTOM)
    rear = block(-96.0, 96.0, 33.5 - PERF_REAR_W, 33.5,
                 PERF_LEDGE_Z0, PERF_BOARD_BOTTOM)
    lip = block(PERF_BOARD_X0, PERF_BOARD_X1, PERF_LIP_Y0, PERF_LIP_Y1,
                PERF_BOARD_BOTTOM - 0.5, PERF_BOARD_TOP)
    frame = side_l + side_r + front + rear + lip
    return frame


def perfboard_reference():
    """The user-supplied 190 x 62.5 x 1.6 mm board (never exported)."""
    return block(PERF_BOARD_X0, PERF_BOARD_X1, PERF_BOARD_Y0, PERF_BOARD_Y1,
                 PERF_BOARD_BOTTOM, PERF_BOARD_TOP)


def platform_top():
    """The upper platform: a lightweight screwed-on cover.

    It sits on the 8 mm roof ledges (LEDGE_TOP = 65) and is held by four
    M3 self-tapping screws that pass through the plate's countersunk holes
    and cut into the M3 pilots already cut into the ledges at (±91, ±20).
    The middle is hollowed out (asymmetric front/rear rims keep the screw
    holes on solid material) and a central spine keeps it stiff.
    No hinges, latches, rims or straps: just a light cover.
    """
    PLATE_X0, PLATE_X1 = -94.75, 94.75
    PLATE_Y0, PLATE_Y1 = -29.0, ROOF_SEAT_Y1     # -29 .. 33.5
    T = 2.0
    z0 = LEDGE_TOP                               # 65.0
    z1 = z0 + T                                  # 67.0

    plate = block(PLATE_X0, PLATE_X1, PLATE_Y0, PLATE_Y1, z0, z1)
    # Hollow out the middle; rims stay >= 4.9/9.5 mm in y and 11 mm in x so
    # the Ø 6.1 countersinks at (±91, ±20) always land on material.
    plate = plate - block(-83.75, 83.75, -24.0, 24.0, z0 - 1.0, z1 + 1.0)
    # Central stiffening spine across the opening.
    plate = plate + block(-83.75, 83.75, -2.25, 6.75, z0, z1)
    for px, py in SCREWS:
        # M3 clearance through-hole, Ø 3.2.
        plate = plate - Cylinder(1.6, T + 2.0).translate(
            (px, py, z0 + T / 2.0))
        # 90° countersink for a flat-head M3 (ISO 10642, head max Ø 6.0).
        plate = plate - Cone(1.6, 3.05, 1.45).translate(
            (px, py, z1 - 1.45 / 2.0))
    return plate


def hc_sr04_reference():
    """Right HC-SR04 (PCB + two eyes); the left one is its mirror."""
    pcb = block(HC_PCB_X0, HC_PCB_X0 + HC_PCB_T,
                -HC_PCB_LEN / 2.0, HC_PCB_LEN / 2.0,
                PERF_BOARD_TOP, PERF_BOARD_TOP + HC_PCB_H)
    eyes = []
    for dy in (-HC_EYE_DY, HC_EYE_DY):
        eyes.append(
            Cylinder(HC_EYE_R, HC_EYE_LEN).rotate(Y, 90).translate(
                (HC_PCB_X0 + HC_PCB_T + HC_EYE_LEN / 2.0, dy, HC_EYE_Z)
            )
        )
    sensor = pcb + eyes[0] + eyes[1]
    return pcb, eyes[0], eyes[1], sensor, sensor.mirror(Plane.YZ)


def main() -> None:
    # ------------------------------------------------------------------
    # Read the verified 80 mm source.
    # ------------------------------------------------------------------
    print(f"Reading source: {SOURCE_STEP}")
    source = read_step(SOURCE_STEP).bodies.one()
    verify(source, f"source ({SOURCE_HEIGHT:.0f} mm)", expected_top=SOURCE_HEIGHT)

    # The old windows must stay framed above the cut.
    print(f"\n  front window margin: {BODY_HEIGHT - FRONT_WINDOW_TOP_Z:.2f} mm")
    print(f"  side window margin : {BODY_HEIGHT - SIDE_WINDOW_TOP_Z:.2f} mm")
    front_margin = BODY_HEIGHT - FRONT_WINDOW_TOP_Z
    side_margin = BODY_HEIGHT - SIDE_WINDOW_TOP_Z
    if front_margin < MIN_TOP_MATERIAL:
        raise RuntimeError(
            f"front window would lose its frame "
            f"(minimum {MIN_TOP_MATERIAL} mm)"
        )
    if side_margin < MIN_TOP_MATERIAL:
        raise RuntimeError("side window would lose its frame")

    # ------------------------------------------------------------------
    # Lower the tub.  A large box above BODY_HEIGHT gives a clean trim.
    # ------------------------------------------------------------------
    top_tool = Box(300.0, 300.0, CUT_TOOL_HEIGHT).translate(
        (0.0, 0.0, BODY_HEIGHT + CUT_TOOL_HEIGHT / 2.0)
    )
    low = source - top_tool
    verify(low, f"lowered tub ({BODY_HEIGHT:.0f} mm)", expected_top=BODY_HEIGHT)

    # ------------------------------------------------------------------
    # Option 1: thin the big flat skins from the OUTSIDE in.  Walls and
    # the rear wall lose 1.25 mm (now 3.75 mm), the floor loses 1.5 mm
    # (now 3.5 mm, bottom at z = 1.5).  Footprint becomes 197.5 mm.
    # Interior faces do not move, so every interior feature is unchanged.
    # ------------------------------------------------------------------
    opt1_faces = list(low.faces())
    wall_faces = [f for f in opt1_faces
                  if f.geom_type == "plane"
                  and abs(f.normal_at(f.interior_point()).x) > 0.99
                  and abs(f.interior_point().x) > 98.0]
    rear_faces = [f for f in opt1_faces
                  if f.geom_type == "plane"
                  and f.normal_at(f.interior_point()).y > 0.99
                  and f.interior_point().y > 98.0]
    floor_faces = [f for f in opt1_faces
                   if f.geom_type == "plane"
                   and abs(f.normal_at(f.interior_point()).z) > 0.99
                   and abs(f.interior_point().z) < 0.01]
    print(f"\n  option 1 offsets: walls {len(wall_faces)} x -1.25 mm, "
          f"rear {len(rear_faces)} x -1.25 mm, floor {len(floor_faces)} x -1.5 mm")
    low = low.offset_faces(wall_faces, -1.25)
    low = low.offset_faces(rear_faces, -1.25)
    low = low.offset_faces(floor_faces, -1.5)
    FLOOR_BOTTOM_Z = 1.5
    verify(low, "option 1: thinned walls / rear / floor",
           expected_top=BODY_HEIGHT,
           expected_footprint=98.75,
           expected_bottom=FLOOR_BOTTOM_Z)

    # ------------------------------------------------------------------
    # Restore the original scoop nose.  Raising the floor moved the
    # inclined nose slope's lower edge back (z = 0 -> z = 1.5); rebuild
    # the original sharp leading edge as a triangular prism.  Adds back
    # less than 1 g and keeps the front at y = -100, z = 0.
    # ------------------------------------------------------------------
    nose_face = [f for f in low.faces()
                 if f.geom_type == "plane" and f.bbox().min.y < -95.0
                 and abs(f.normal_at(f.interior_point()).y) > 0.1][0]
    nn = nose_face.normal_at(nose_face.interior_point())
    y1 = nose_face.bbox().min.y          # y where the slope meets z = 1.5
    k = abs(nn.y) / nn.z                 # dz per dy of the slope
    y0 = y1 - FLOOR_BOTTOM_Z / k         # y where the slope meets z = 0
    d = 2.0                              # tail overlapping the solid above z = 1.5
    print(f"\n  nose restore: y0 = {y0:.3f}, y1 = {y1:.3f}, slope k = {k:.4f}")
    prof = Polygon((y0, 0.0),
                   (y1 + d, FLOOR_BOTTOM_Z),
                   (y1 + d, FLOOR_BOTTOM_Z + k * d))
    nose_plane = Plane(origin=(-98.75, 0.0, 0.0),
                       x_dir=(0, 1, 0), z_dir=(1, 0, 0))
    low = low + prof.on(nose_plane).extrude(197.5)
    FLOOR_BOTTOM_Z = 0.0
    verify(low, "option 1: nose restored",
           expected_top=BODY_HEIGHT,
           expected_footprint=98.75,
           expected_bottom=0.0)

    # ------------------------------------------------------------------
    # Rear MH sensor window only.  No tray, rails, boss or M3 pilot: the
    # user mounts the sensor, the chassis just has the 13 x 7 mm hole.
    # ------------------------------------------------------------------
    window = Box(WINDOW_W, WALL_T, WINDOW_H).translate(
        (0.0, WINDOW_Y, WINDOW_Z)
    )
    result = low - window
    verify(result, f"result ({BODY_HEIGHT:.0f} mm, sensor window only)",
           expected_top=BODY_HEIGHT,
           expected_footprint=98.75, expected_bottom=FLOOR_BOTTOM_Z)

    # The tub is symmetric about the YZ plane.  Check that here, before
    # the frame and roof seats are added.
    mirrored = result.mirror(Plane.YZ)
    sym_diff = result.volume - (result & mirrored).volume
    print(f"\n  mirror volume difference (tub): {sym_diff:.9f} mm^3")
    if abs(sym_diff) > 1e-6:
        raise RuntimeError("tub is not symmetric about the YZ plane")

    # ------------------------------------------------------------------
    # Roof seats: 8 mm ledges on the side walls, above the side windows.
    # Embedded 1 mm into the walls.  There is no centre spine/divider.
    # ------------------------------------------------------------------
    seat_l = block(-96.0, -95.0 + LEDGE_W, ROOF_SEAT_Y0, ROOF_SEAT_Y1,
                   LEDGE_BOTTOM, LEDGE_TOP)
    seat_r = block(95.0 - LEDGE_W, 96.0, ROOF_SEAT_Y0, ROOF_SEAT_Y1,
                   LEDGE_BOTTOM, LEDGE_TOP)

    # Perfboard frame at the side-window sill: the only roof/shelf.  The
    # bay below it is left completely open (no driver deck).
    result = result + perfboard_frame()
    result = result + seat_l
    result = result + seat_r

    # M3 self-tapping pilots in the side seats (3.5 mm deep).
    for px, py in SCREWS:
        result = result - Cylinder(SCREW_PILOT_R, 3.8).translate(
            (px, py, LEDGE_TOP - 1.5)
        )

    # ------------------------------------------------------------------
    # Stepped motor seats in the rear floor: gearbox pocket, body pad and
    # four through Ø3.5 mm M3 holes per motor.  The pockets are cut
    # first, then the pads are added, so the 2 mm step at x = 40 is clean.
    # The left side is a mirror of the right.
    # ------------------------------------------------------------------
    pad_r, pocket_r, holes_r = motor_seat()
    pad_l = pad_r.mirror(Plane.YZ)
    pocket_l = pocket_r.mirror(Plane.YZ)
    holes_l = [h.mirror(Plane.YZ) for h in holes_r]
    result = result - pocket_r - pocket_l
    result = result + pad_r + pad_l
    for hole in holes_r + holes_l:
        result = result - hole

    # ------------------------------------------------------------------
    # Lower the rear wall for weight.  The rear slab (x -71..71, y 95..100)
    # and the two rear side fins (x ±95..100, y = 95) are cut down to
    # z = 46: the wall is only needed below board height, and the fins are
    # pure ballast.  Nothing above z = 46 at the rear is kept.
    # ------------------------------------------------------------------
    REAR_WALL_Z = 46.0
    rear_slab = block(-71.0, 71.0, 95.0, 100.0, REAR_WALL_Z, BODY_HEIGHT)
    rear_fin_r = block(95.0, 100.0, 92.5, 97.5, REAR_WALL_Z, BODY_HEIGHT)
    rear_fin_l = block(-100.0, -95.0, 92.5, 97.5, REAR_WALL_Z, BODY_HEIGHT)
    result = result - rear_slab - rear_fin_r - rear_fin_l
    print(f"\n  rear wall lowered to z = {REAR_WALL_Z:.0f} "
          f"(slab + 2 fins removed)")

    # ------------------------------------------------------------------
    # Lightweighting cuts (every gram counts).  The bay floor is relieved
    # over the open region (motor seats and perfboard frame stay outside it),
    # leaving a 2.0 mm skin.  The outer walls are skins that are only
    # face-attached to the tub, so a parallel sliver would detach them
    # completely; they are left at 5 mm.
    # ------------------------------------------------------------------
    # keep a 2.0 mm skin above the thinned floor bottom (z = 1.5);
    # literal because FLOOR_BOTTOM_Z was rebound to 0.0 for the nose fix
    result = result - block(-84.0, 84.0, -24.0, 24.0,
                            3.5, MOTOR_BAY_FLOOR_Z)
    print("  lightweight cut: floor relief over the open bay (2.0 mm skin)")

    # ------------------------------------------------------------------
    # Stiffening ribs on the thinned side walls: 2 mm proud webs, full
    # height of the open bay, well below the perfboard frame / windows.
    # ------------------------------------------------------------------
    # y stays forward of the wheel sweep (wheel reaches y = 35)
    rib_r = block(92.6, 95.6, -50.0, 25.0, 5.0, 38.0)
    rib_l = block(-95.6, -92.6, -50.0, 25.0, 5.0, 38.0)
    result = result + rib_r + rib_l
    print("  stiffening ribs added on both inner side walls")

    # ------------------------------------------------------------------
    # Ground-facing infrared sensor slots (vue de dessous): through-holes
    # only, no sensor holders.  31 x 13 mm through-holes (MH board + LED
    # length) cut straight through the floor:
    #   * front-left corner (front-left), front-right corner (front-right)
    #   * one in the floor between the two motors (x = 0)
    # ------------------------------------------------------------------
    def slot_cutter(length, width):
        r = width / 2.0
        s = (length - width) / 2.0
        cut = block(-s, s, -r, r, -0.5, MOTOR_BAY_FLOOR_Z + 0.5)
        cut = cut + Cylinder(r, MOTOR_BAY_FLOOR_Z + 1.0).translate(
            (s, 0.0, MOTOR_BAY_FLOOR_Z / 2.0))
        cut = cut + Cylinder(r, MOTOR_BAY_FLOOR_Z + 1.0).translate(
            (-s, 0.0, MOTOR_BAY_FLOOR_Z / 2.0))
        return cut

    IR_SLOT_L, IR_SLOT_W = 31.0, 13.0
    for px, py in ((-75.0, -78.0), (75.0, -78.0), (0.0, MOTOR_Y)):
        result = result - slot_cutter(IR_SLOT_L, IR_SLOT_W).translate(
            (px, py, 0.0))
    print("  IR ground slots cut (31 x 13 mm): front-left, front-right, between the motors")

    verify(result,
           f"chassis ({BODY_HEIGHT:.0f} mm, perfboard roof + roof seats + motor seats)",
           expected_top=BODY_HEIGHT,
           expected_footprint=98.75, expected_bottom=FLOOR_BOTTOM_Z)

    mp = result.mass_properties(1.0)
    print(f"\n  centre of mass (density 1): "
          f"({mp.center.x:.2f}, {mp.center.y:.2f}, {mp.center.z:.2f}) mm")
    print(f"  PETG mass at 1.27 g/cm^3: {result.volume * 1.27 / 1000.0:.1f} g")

    # ------------------------------------------------------------------
    # Upper platform: a separate printed part, 2 mm lightweight plate,
    # top at z = 67.  Screwed onto the roof ledges with 4 M3 screws;
    # report it, and keep it for the export checks below.
    # ------------------------------------------------------------------
    platform = platform_top()
    report(platform, "platform (upper plate)")
    platform_interference = result.intersect(platform).volume
    print(f"  platform-to-chassis interference: {platform_interference:.6f} mm^3")
    if platform_interference > 1e-6:
        raise RuntimeError("the platform intersects the chassis")
    platform_clearance = platform.distance_to(result)
    print(f"  platform-to-chassis minimum gap: {platform_clearance:.3f} mm")

    # ------------------------------------------------------------------
    # The upper platform is exported as a separate part.  Seats and M3
    # pilots stay in the chassis; the platform is screwed onto them.
    # ------------------------------------------------------------------
    print("\n  upper platform: exported separately (seats and pilots kept in chassis)")

    # ------------------------------------------------------------------
    # Functional checks.
    # ------------------------------------------------------------------
    print("\nFunctional checks:")
    # Existing front/side windows still open, with wall above them.
    front_checks = [(42.5, -35.0, 60.0), (42.5, -35.0, 65.0),
                    (-42.5, -35.0, 60.0), (-42.5, -35.0, 65.0)]
    for p in front_checks:
        if result.contains(Vertex(*p)):
            raise RuntimeError(f"front window blocked at {p}")
    for p in ((42.5, -35.0, 66.5), (42.5, -35.0, 67.5)):
        if not result.contains(Vertex(*p)):
            raise RuntimeError(f"front window rim missing at {p}")
    for p in ((96.9, 0.0, 50.0), (-96.9, 0.0, 50.0)):
        if result.contains(Vertex(*p)):
            raise RuntimeError(f"side window blocked at {p}")
    for p in ((96.9, 0.0, 39.0), (-96.9, 0.0, 39.0),
              (96.9, 0.0, 61.0), (-96.9, 0.0, 61.0)):
        if not result.contains(Vertex(*p)):
            raise RuntimeError(f"side window rim missing at {p}")
    print("  front/side windows: open, rims intact")
    print("  side window sills: z = 40.0, tops z = 60.0 (20 mm high)")

    # Rear window open, wall solid around it.
    for p in ((0.0, 96.9, WINDOW_Z), (12.0, 96.9, WINDOW_Z),
              (-12.0, 96.9, WINDOW_Z)):
        if result.contains(Vertex(*p)):
            raise RuntimeError(f"rear window blocked at {p}")
    for p in ((17.0, 98.0, WINDOW_Z), (-17.0, 98.0, WINDOW_Z)):
        if not result.contains(Vertex(*p)):
            raise RuntimeError(f"rear window edge missing at {p}")
    for p in ((0.0, 96.9, WINDOW_Z - WINDOW_H / 2.0 - 0.2),
              (0.0, 96.9, WINDOW_Z + WINDOW_H / 2.0 + 0.2)):
        if not result.contains(Vertex(*p)):
            raise RuntimeError(f"rear window rim missing at {p}")
    print(f"  rear window: {WINDOW_W:.0f} x {WINDOW_H:.0f} mm open at z = {WINDOW_Z:.1f}")

    # No sensor tray: the floor where the old rails/boss were must be clear.
    for p in ((8.2, 72.5, 7.0), (-8.2, 72.5, 7.0),
              (0.0, 45.0, 7.0), (0.0, 45.0, 5.5),
              (0.0, SENSOR_HOLE_Y, 7.0)):
        if result.contains(Vertex(*p)):
            raise RuntimeError(f"sensor tray material left at {p}")
    print("  rear wall: window only, no sensor tray or M3 boss")

    # Reference board, LEDs and header must clear the chassis.
    board, led_r, led_l, header, sensor = sensor_reference()
    for name, part in (("board", board), ("led_r", led_r),
                       ("led_l", led_l), ("header", header),
                       ("sensor", sensor)):
        inter = result.intersect(part).volume
        if inter > 1e-6:
            raise RuntimeError(f"chassis intersects the {name}: {inter:.6f} mm^3")
    led_clearance = min(led_r.distance_to(result), led_l.distance_to(result))
    print(f"  sensor: no intersection, LED-to-window clearance {led_clearance:.2f} mm")
    if led_clearance < 0.3:
        raise RuntimeError("LED-to-window clearance is too small")

    # The electronics bay is open from the floor all the way up to the
    # perfboard roof: no lower deck, no ribs.
    bay = block(-86.5, 86.5, -24.9, 28.4, 6.0, PERF_BOARD_BOTTOM - 0.1)
    if result.intersect(bay).volume > 1e-6:
        raise RuntimeError("the bay below the perfboard is not clear")
    print(f"  electronics bay: clear, {PERF_BOARD_BOTTOM - 5.0:.1f} mm above "
          "the floor, open to the rear")

    # Perfboard frame, side roof seats and pilots.
    probes = {
        "open bay at deck level (centre)": ((0.0, 0.0, 27.5), False),
        "open bay at deck level (left)": ((-75.0, 20.0, 27.5), False),
        "open bay at deck level (right)": ((75.0, 20.0, 27.5), False),
        "open bay under the board": ((34.0, 15.0, 33.0), False),
        "open bay above the board": ((34.0, 0.0, 50.0), False),
        "open bay at the roof line": ((34.0, 20.0, 63.0), False),
        "left roof seat": ((-91.0, 10.0, 63.0), True),
        "right roof seat": ((91.0, 10.0, 63.0), True),
        "left seat pilot": ((-91.0, 20.0, 63.0), False),
        "right seat pilot": ((91.0, 20.0, 63.0), False),
        # Perfboard frame: side/front ledges, rear beam and rear stop.
        "left frame ledge": ((-91.0, 0.0, 37.0), True),
        "right frame ledge": ((91.0, 0.0, 37.0), True),
        "front frame ledge": ((0.0, -27.0, 37.0), True),
        "rear frame beam": ((0.0, 30.0, 37.0), True),
        "rear board stop": ((0.0, 33.0, 39.2), True),
        "frame opening under the board": ((0.0, 0.0, 37.0), False),
        "board space above the left ledge": ((-91.0, 0.0, 39.0), False),
        "space above the board": ((0.0, 0.0, 40.5), False),
    }
    for label, (point, expected) in probes.items():
        if result.contains(Vertex(*point)) != expected:
            raise RuntimeError(f"{label} is in the wrong state at {point}")
    print("  perfboard frame and side roof seats: present; bay open below")

    # Above the board, the bay must stay open across the width.
    upper_bay = block(-86.0, 86.0, -29.5, 32.0,
                      PERF_BOARD_TOP + 0.5, LEDGE_TOP - 0.1)
    bay_volume = result.intersect(upper_bay).volume
    if bay_volume > 1e-6:
        raise RuntimeError(f"the bay above the board is not open: "
                           f"{bay_volume:.6f} mm^3")
    print(f"  bay above the board: clear from z = {PERF_BOARD_TOP + 0.5:.1f} "
          "to the roof line")

    # Stepped motor seats: pocket floor, pad top, step and M3 holes.
    def floor_z(x, y):
        hits = result.raycast((x, y, BODY_HEIGHT + 20.0), (0, 0, -1))
        return None if not hits else hits[0].point.z

    seat_probes = {
        "gearbox pocket floor": (52.0, MOTOR_Y,
                                 MOTOR_BAY_FLOOR_Z - GEARBOX_POCKET_DEPTH),
        "body pad top": (28.65, MOTOR_Y,
                         MOTOR_BAY_FLOOR_Z + BODY_PAD_HEIGHT),
        "pad at the step": (GEARBOX_POCKET_X0 - 0.1, MOTOR_Y,
                            MOTOR_BAY_FLOOR_Z + BODY_PAD_HEIGHT),
        "pocket past the step": (GEARBOX_POCKET_X0 + 0.1, MOTOR_Y,
                                 MOTOR_BAY_FLOOR_Z - GEARBOX_POCKET_DEPTH),
        "floor outside pocket": (52.0,
                                 MOTOR_Y - GEARBOX_POCKET_W / 2.0 - 0.1,
                                 MOTOR_BAY_FLOOR_Z),
        "pocket inside edge": (52.0,
                               MOTOR_Y - GEARBOX_POCKET_W / 2.0 + 0.1,
                               MOTOR_BAY_FLOOR_Z - GEARBOX_POCKET_DEPTH),
        "floor outside pad": (28.65,
                              MOTOR_Y - BODY_PAD_W / 2.0 - 0.1,
                              MOTOR_BAY_FLOOR_Z),
        "pad inside edge": (28.65,
                            MOTOR_Y - BODY_PAD_W / 2.0 + 0.1,
                            MOTOR_BAY_FLOOR_Z + BODY_PAD_HEIGHT),
    }
    for label, (x, y, expected) in seat_probes.items():
        z = floor_z(x, y)
        if z is None or abs(z - expected) > 1e-3:
            raise RuntimeError(f"{label} is at z={z}, expected {expected}")
    hole_centers = [(MOTOR_HOLE_X + dx, MOTOR_Y + dy)
                    for dx in (-MOTOR_HOLE_DX, MOTOR_HOLE_DX)
                    for dy in (-MOTOR_HOLE_DY, MOTOR_HOLE_DY)]
    for x, y in hole_centers + [(-x, y) for x, y in hole_centers]:
        for z in (1.55, 3.0, 4.5):
            if result.contains(Vertex(x, y, z)):
                raise RuntimeError(f"motor hole at ({x}, {y}, z={z}) is blocked")
        if not result.contains(Vertex(x + 3.0, y, 3.0)):
            raise RuntimeError(f"floor material missing next to motor hole at ({x}, {y})")
    print("  motor seats: gearbox pocket, body pad and step at x = 40 correct")
    print("  motor holes: 8 through Ø3.5 mm, open top to bottom")

    # Perfboard seat and side sensors.  The board is user-supplied; the
    # frame must leave its 190 x 62.5 x 1.6 mm pocket free, and the two
    # HC-SR04 PCBs must stand on it at the 44 x 20 mm side windows.
    perfboard = perfboard_reference()
    board_check = block(PERF_BOARD_X0 + 0.1, PERF_BOARD_X1 - 0.1,
                        PERF_BOARD_Y0 + 0.1, PERF_BOARD_Y1 - 0.1,
                        PERF_BOARD_BOTTOM + 0.1, PERF_BOARD_TOP - 0.1)
    inter = result.intersect(board_check).volume
    if inter > 1e-6:
        raise RuntimeError(f"the perfboard pocket is obstructed: {inter:.6f} mm^3")
    full_inter = result.intersect(perfboard).volume
    if full_inter > 1e-6:
        raise RuntimeError(f"a {PERF_BOARD_LEN:.1f} mm board does not drop in: "
                           f"{full_inter:.6f} mm^3")
    print(f"  perfboard seat: {PERF_BOARD_X1 - PERF_BOARD_X0:.0f} x "
          f"{PERF_BOARD_Y1 - PERF_BOARD_Y0:.1f} x {PERF_BOARD_T:.1f} mm "
          f"at z = {PERF_BOARD_TOP:.1f} (drops in, rests on the frame)")

    pcb_r, eye_r0, eye_r1, sensor_r, sensor_l = hc_sr04_reference()
    for name, part in (("right sensor PCB", pcb_r),
                       ("right sensor eye 1", eye_r0),
                       ("right sensor eye 2", eye_r1),
                       ("left sensor", sensor_l)):
        if result.intersect(part).volume > 1e-6:
            raise RuntimeError(f"chassis intersects the {name}")
    for side, pcb in (("right", pcb_r), ("left", pcb_r.mirror(Plane.YZ))):
        if pcb.distance_to(perfboard) > 1e-6:
            raise RuntimeError(f"the {side} sensor PCB does not sit on the board")
    print("  side sensors: PCBs stand on the board, eyes at z = 50 in the "
          "44 x 20 mm windows")

    # Battery pack and ESP: sit on the perfboard, below the roof line.
    battery = block(BATTERY_X0, BATTERY_X1, BATTERY_Y0, BATTERY_Y1,
                    BATTERY_Z0, BATTERY_Z0 + BATTERY_T)
    esp = block(ESP_X0, ESP_X1, ESP_Y0, ESP_Y1, ESP_Z0, ESP_Z0 + ESP_T)
    for name, part in (("battery pack", battery), ("ESP board", esp)):
        inter = result.intersect(part).volume
        if inter > 1e-6:
            raise RuntimeError(f"chassis intersects the {name}: {inter:.6f} mm^3")
        if part.bbox().max.z >= LEDGE_TOP:
            raise RuntimeError(f"the {name} reaches the roof seats")
    print(f"  battery pack: on the perfboard, top z = {BATTERY_Z0 + BATTERY_T:.1f}, "
          f"{(LEDGE_TOP - BATTERY_Z0 - BATTERY_T):.1f} mm below the roof line")
    print(f"  ESP board   : on the perfboard, top z = {ESP_Z0 + ESP_T:.1f}, "
          f"{(LEDGE_TOP - ESP_Z0 - ESP_T):.1f} mm below the roof line")
    battery_esp_gap = battery.distance_to(esp)
    print(f"  battery-to-ESP gap: {battery_esp_gap:.2f} mm (no centre divider)")
    if battery_esp_gap < 5.0:
        raise RuntimeError("the battery and the ESP are too close together")

    # Reference wheels and motors, seated in the new pockets.
    seated_z = MOTOR_Z - MOTOR_SEAT_DROP
    wheel_r = Cylinder(WHEEL_RADIUS, WHEEL_WIDTH).rotate(Y, 90).translate(
        (WHEEL_X, WHEEL_Y, WHEEL_Z - MOTOR_SEAT_DROP)
    )
    wheel_l = wheel_r.mirror(Plane.YZ)
    motor_r, motor_l = motor_pair(seated_z)
    for name, part in (("wheel_r", wheel_r), ("wheel_l", wheel_l),
                       ("motor_r", motor_r), ("motor_l", motor_l)):
        if result.intersect(part).volume > 1e-6:
            raise RuntimeError(f"chassis intersects the {name}")
    wheel_clearance = result.distance_to(wheel_r)
    print(f"  wheel clearance (seated): {wheel_clearance:.2f} mm")
    if wheel_clearance < 0.3:
        raise RuntimeError("the seated wheel is too close to the chassis")

    # Both the gearbox and the body must touch their seat, so the motor
    # cannot rock on the gearbox shoulder.
    gearbox_seated = Cylinder(GEARBOX_R, GEARBOX_LEN).rotate(Y, 90).translate(
        (52.0, MOTOR_Y, seated_z)
    )
    body_seated = Cylinder(BODY_R, BODY_LEN).rotate(Y, 90).translate(
        (28.65, MOTOR_Y, seated_z)
    )
    for name, part in (("gearbox", gearbox_seated),
                       ("motor body", body_seated)):
        if result.intersect(part).volume > 1e-6:
            raise RuntimeError(f"the seated {name} intersects the chassis")
        if result.distance_to(part) > 1e-6:
            raise RuntimeError(f"the seated {name} does not touch its seat")
    print(f"  motor seat: gearbox and body both in contact at z = {seated_z:.1f}")
    print(f"  motor drop: {MOTOR_SEAT_DROP:.1f} mm")
    print(f"  battery-to-motor clearance: {battery.distance_to(motor_r):.1f} mm")
    print(f"  ESP-to-motor clearance: {esp.distance_to(motor_r):.1f} mm")

    # ------------------------------------------------------------------
    # Export and re-import check.
    # ------------------------------------------------------------------
    print("\nExporting:")
    export_step(result, OUT_STEP)
    print(f"  wrote {OUT_STEP.name} ({OUT_STEP.stat().st_size:,} bytes)")
    export_stl(result, OUT_STL, tolerance=STL_TOLERANCE, angle=STL_ANGLE)
    print(f"  wrote {OUT_STL.name} ({OUT_STL.stat().st_size:,} bytes)")
    export_step(platform, PLATFORM_STEP)
    print(f"  wrote {PLATFORM_STEP.name} ({PLATFORM_STEP.stat().st_size:,} bytes)")
    export_stl(platform, PLATFORM_STL, tolerance=STL_TOLERANCE, angle=STL_ANGLE)
    print(f"  wrote {PLATFORM_STL.name} ({PLATFORM_STL.stat().st_size:,} bytes)")

    print("\nRe-importing STEP:")
    reimported = read_step(OUT_STEP).bodies.one()
    verify(reimported, "re-imported chassis", expected_top=BODY_HEIGHT,
           expected_footprint=98.75, expected_bottom=FLOOR_BOTTOM_Z)
    dv = abs(reimported.volume - result.volume)
    print(f"  volume delta vs built solid: {dv:.9f} mm^3")
    if dv > 1e-3:
        raise RuntimeError("re-imported STEP volume differs from the built solid")
    platform_reimported = read_step(PLATFORM_STEP).bodies.one()
    dv = abs(platform_reimported.volume - platform.volume)
    print(f"  platform volume delta vs built solid: {dv:.9f} mm^3")
    if dv > 1e-3:
        raise RuntimeError("re-imported platform STEP volume differs from the built solid")

    print("\nChecking STL:")
    for name, path, built in (("chassis", OUT_STL, result),
                              ("platform", PLATFORM_STL, platform)):
        info = stl_stats(path)
        stl_err = abs(info["volume"] - built.volume) / built.volume * 100.0
        print(f"  {name}: {info['triangles']:,} triangles, "
              f"open edges {info['open_edges']}, volume error {stl_err:.4f} %")
        if info["open_edges"]:
            raise RuntimeError(f"{name} STL is not watertight")
        if stl_err > 0.05:
            raise RuntimeError(f"{name} STL volume error is too large")
    print("  all STL files watertight")

    # ------------------------------------------------------------------
    # Renders (reference wheels/motors are visual only; the rear hole is
    # shown empty because the sensor holder/case is not part of the model).
    # ------------------------------------------------------------------
    RENDER_DIR.mkdir(exist_ok=True)
    chassis_col = result.colored("#c2410c")
    wheel_col = wheel_r.colored("#111111")
    motor_col = motor_r.colored("#3f3f46")
    battery_col = battery.colored("#2563eb")
    esp_col = esp.colored("#16a34a")
    platform_col = platform.colored("#b45309")
    perfboard_col = perfboard.colored("#d8c48f")
    sensor_col = sensor_r.colored("#0f766e")

    print("\nRendering:")
    render(chassis_col,
           wheel_col, wheel_col.mirror(Plane.YZ),
           motor_col, motor_col.mirror(Plane.YZ),
           path=str(RENDER_DIR / "option1_iso.png"),
           view="iso", size=(1200, 1000), edges=True, background="white")
    render(chassis_col, platform_col,
           path=str(RENDER_DIR / "option1_platform_iso.png"),
           view="iso", size=(1200, 1000), edges=True, background="white")
    render(chassis_col, platform_col,
           path=str(RENDER_DIR / "option1_platform_top.png"),
           view="top", size=(1200, 800), edges=True, background="white")
    render(chassis_col, platform_col,
           path=str(RENDER_DIR / "option1_platform_side.png"),
           view="right", size=(1000, 600),
           section=Plane(origin=(91.0, 0.0, 0.0), z_dir=(1.0, 0.0, 0.0)),
           edges=True, background="white")
    render(chassis_col, platform_col, battery_col, esp_col,
           path=str(RENDER_DIR / "option1_platform_loaded.png"),
           view="iso", size=(1200, 900), edges=True, background="white",
           xray=True)
    render(chassis_col, platform_col,
           path=str(RENDER_DIR / "option1_top.png"),
           view="top", size=(1200, 1000), edges=True, background="white")
    render(chassis_col,
           path=str(RENDER_DIR / "option1_motor_seats.png"),
           view="top", size=(1200, 900), zoom=2.0,
           look_at=(38.0, MOTOR_Y, MOTOR_BAY_FLOOR_Z),
           edges=True, background="white")
    render(chassis_col, perfboard_col, battery_col,
           path=str(RENDER_DIR / "option1_section_battery.png"),
           view="right", size=(1200, 900),
           section=Plane(origin=(BATTERY_X0 + 20.0, 0, 0), z_dir=(1, 0, 0)),
           zoom=1.65, look_at=(0, 2, 45), edges=True, background="white")
    render(chassis_col, perfboard_col, esp_col,
           path=str(RENDER_DIR / "option1_section_esp.png"),
           view="right", size=(1200, 900),
           section=Plane(origin=(ESP_X0 + 14.0, 0, 0), z_dir=(1, 0, 0)),
           zoom=2.2, look_at=(0, 2, 45), edges=True, background="white")
    render(chassis_col, wheel_col, wheel_col.mirror(Plane.YZ),
           motor_col, motor_col.mirror(Plane.YZ),
           path=str(RENDER_DIR / "option1_back.png"),
           view="back", size=(1200, 900), edges=True, background="white")
    render(chassis_col,
           path=str(RENDER_DIR / "option1_window.png"),
           view="back", size=(1100, 850), zoom=4.2,
           look_at=(0.0, 97.0, WINDOW_Z),
           edges=True, background="white")
    render(chassis_col,
           path=str(RENDER_DIR / "option1_front.png"),
           view="front", size=(1200, 900), edges=True, background="white")
    render(chassis_col, perfboard_col,
           path=str(RENDER_DIR / "option1_frame_section.png"),
           view="right", size=(1200, 900),
           section=Plane(origin=(0.0, 0.0, 0.0), z_dir=(1.0, 0.0, 0.0)),
           zoom=2.0, look_at=(0.0, 0.0, 38.0),
           edges=True, background="white")
    render(chassis_col, perfboard_col,
           sensor_col, sensor_col.mirror(Plane.YZ),
           path=str(RENDER_DIR / "option1_sensor_section.png"),
           view="front", size=(1200, 900),
           section=Plane.XZ, zoom=1.4, look_at=(0.0, 0.0, 40.0),
           edges=True, background="white")
    render(chassis_col, perfboard_col,
           sensor_col, sensor_col.mirror(Plane.YZ), esp_col,
           path=str(RENDER_DIR / "option1_open_bay.png"),
           view={"azimuth": 35, "elevation": 25}, size=(1200, 900),
           section=Plane.XZ, zoom=1.5, look_at=(0.0, 0.0, 30.0),
           edges=True, background="white")
    render(chassis_col, perfboard_col,
           sensor_col, sensor_col.mirror(Plane.YZ), esp_col,
           path=str(RENDER_DIR / "option1_perfboard_iso.png"),
           view="iso", size=(1200, 1000), edges=True, background="white")
    render(chassis_col, perfboard_col,
           sensor_col, sensor_col.mirror(Plane.YZ), esp_col,
           path=str(RENDER_DIR / "option1_perfboard.png"),
           view="top", size=(1200, 1000), zoom=1.4,
           edges=True, background="white")
    render(chassis_col, perfboard_col,
           sensor_col, sensor_col.mirror(Plane.YZ), esp_col, battery_col,
           path=str(RENDER_DIR / "option1_perfboard_loaded.png"),
           view="top", size=(1200, 1000), zoom=1.4,
           edges=True, background="white")
    print(f"  wrote option1_*.png in {RENDER_DIR.name}/")

    print("\nWriting viewer:")
    show(chassis_col, platform_col, perfboard_col,
         sensor_col, sensor_col.mirror(Plane.YZ), esp_col,
         path=str(VIEWER_HTML),
         title=("Sumo chassis 68 mm: upper platform screwed to the roof seats, "
                "perfboard frame at the side-window sill "
                "(sensors and ESP on the board)"),
         open_browser=False)
    print(f"  wrote {VIEWER_HTML.name}")

    print("\nDone.")
    print(f"  body height: {BODY_HEIGHT:.1f} mm (was {SOURCE_HEIGHT:.1f} mm)")
    wheel_drop = WHEEL_RADIUS - (WHEEL_Z - MOTOR_SEAT_DROP)
    print(f"  motor seat drop: {MOTOR_SEAT_DROP:.1f} mm")
    print(f"  overall height with the Ø65 wheels: "
          f"{BODY_HEIGHT + wheel_drop:.1f} mm")
    print(f"  ride height (wheel bottom below the chassis floor): "
          f"{wheel_drop:.1f} mm (nose lip; 12.0 mm under the thinned floor)")
    print("  footprint: 197.5 x 198.75 mm (walls/rear -1.25, nose restored)")
    print(f"  perfboard frame: {PERF_BOARD_X1 - PERF_BOARD_X0:.0f} x "
          f"{PERF_BOARD_Y1 - PERF_BOARD_Y0:.1f} mm board, top z = "
          f"{PERF_BOARD_TOP:.1f} mm (side-window sill)")
    print("  board: user-supplied, not exported")
    print(f"  platform: {(94.75 - (-94.75)):.1f} x {(33.5 - (-29.0)):.1f} mm, "
          f"2.0 mm thick, hollowed middle + spine, "
          f"M3 countersunk through-holes at (±91, ±20)")
    print(f"  platform mass at 1.27 g/cm^3: {platform.volume * 1.27 / 1000.0:.1f} g")
    print("  parts to print: chassis + platform")


if __name__ == "__main__":
    main()
