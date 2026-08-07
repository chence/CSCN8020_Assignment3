from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "assets/g1_fixed_base/scene_29dof_fixed_base.xml"

ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
)

# A visible, comfortably reachable pose for the fixed-base G1.
REST_POSE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
TOUCH_POSE = np.array([-1.2, 0.3, 0.0, 1.2], dtype=np.float64)


@dataclass(frozen=True)
class Phase:
    name: str
    duration: float
    start: np.ndarray
    end: np.ndarray


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class ReachTouchDemo:
    def __init__(self, scene: Path, target_radius: float = 0.045) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        self.data = mujoco.MjData(self.model)
        self.target_radius = target_radius

        self.qpos_indices: list[int] = []
        self.qvel_indices: list[int] = []
        self.actuator_indices: list[int] = []
        for joint_name in ARM_JOINTS:
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_name = joint_name.removesuffix("_joint")
            actuator_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            self.qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            self.qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))
            self.actuator_indices.append(actuator_id)

        self.hand_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
        )
        self.hand_offset = np.array([0.08, 0.0, 0.0], dtype=np.float64)
        self.all_actuator_mapping = self._build_actuator_mapping()
        self.target_position = self._position_for_pose(TOUCH_POSE)
        self.reset()

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object was not found: {name}")
        return object_id

    def _build_actuator_mapping(self) -> list[tuple[int, int, int]]:
        mapping = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            mapping.append(
                (
                    actuator_id,
                    int(self.model.jnt_qposadr[joint_id]),
                    int(self.model.jnt_dofadr[joint_id]),
                )
            )
        return mapping

    def _position_for_pose(self, pose: np.ndarray) -> np.ndarray:
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qpos_indices] = pose
        mujoco.mj_forward(self.model, self.data)
        position = self.hand_position().copy()
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        return position

    def hand_position(self) -> np.ndarray:
        rotation = self.data.xmat[self.hand_body_id].reshape(3, 3)
        return (
            self.data.xpos[self.hand_body_id]
            + rotation @ self.hand_offset
        )

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.hold_pose = self.data.qpos.copy()

    def apply_pd(self, arm_target: np.ndarray) -> None:
        arm_actuators = set(self.actuator_indices)
        target_by_actuator = dict(zip(self.actuator_indices, arm_target))
        for actuator_id, qpos_index, qvel_index in self.all_actuator_mapping:
            if actuator_id in arm_actuators:
                target = float(target_by_actuator[actuator_id])
                kp, kd = 30.0, 3.0
            else:
                target = float(self.hold_pose[qpos_index])
                kp, kd = 30.0, 3.0
            torque = (
                kp * (target - float(self.data.qpos[qpos_index]))
                - kd * float(self.data.qvel[qvel_index])
                + float(self.data.qfrc_bias[qvel_index])
            )
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(torque, low, high)

    def add_target_to_viewer(self, viewer: object, touched: bool) -> None:
        scene = viewer.user_scn
        scene.ngeom = 1
        color = (
            np.array([0.1, 0.9, 0.2, 0.75], dtype=np.float32)
            if touched
            else np.array([0.95, 0.15, 0.1, 0.75], dtype=np.float32)
        )
        mujoco.mjv_initGeom(
            scene.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, self.target_radius),
            self.target_position,
            np.eye(3).reshape(-1),
            color,
        )


def run(args: argparse.Namespace) -> int:
    demo = ReachTouchDemo(args.scene, args.target_radius)
    phases = (
        Phase("REST", 1.0, REST_POSE, REST_POSE),
        Phase("REACH", 3.0, REST_POSE, TOUCH_POSE),
        Phase("TOUCH/HOLD", 2.0, TOUCH_POSE, TOUCH_POSE),
        Phase("RETURN", 3.0, TOUCH_POSE, REST_POSE),
    )
    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(demo.model, demo.data)
        viewer.cam.lookat[:] = np.array([0.05, 0.05, 0.9])
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 150.0
        viewer.cam.elevation = -10.0

    rows: list[dict[str, object]] = []
    touch_steps = 0
    minimum_distance = float("inf")
    start_wall = time.perf_counter()
    simulation_start = float(demo.data.time)
    last_print_second = -1

    print("DEMO TYPE: scripted PD baseline (not a trained RL policy)")
    print(f"target_xyz={np.round(demo.target_position, 4).tolist()}")
    try:
        for phase in phases:
            phase_start = float(demo.data.time)
            print(f"\nPHASE {phase.name}")
            while float(demo.data.time) - phase_start < phase.duration:
                elapsed = float(demo.data.time) - phase_start
                blend = smoothstep(elapsed / phase.duration)
                arm_target = phase.start + blend * (phase.end - phase.start)
                demo.apply_pd(arm_target)
                mujoco.mj_step(demo.model, demo.data)

                hand = demo.hand_position()
                distance = float(np.linalg.norm(hand - demo.target_position))
                touched = distance <= args.target_radius
                minimum_distance = min(minimum_distance, distance)
                if phase.name == "TOUCH/HOLD" and touched:
                    touch_steps += 1

                rows.append(
                    {
                        "time": float(demo.data.time) - simulation_start,
                        "phase": phase.name,
                        "hand_x": hand[0],
                        "hand_y": hand[1],
                        "hand_z": hand[2],
                        "target_x": demo.target_position[0],
                        "target_y": demo.target_position[1],
                        "target_z": demo.target_position[2],
                        "distance": distance,
                        "touched": int(touched),
                    }
                )

                whole_second = int(float(demo.data.time) - simulation_start)
                if whole_second != last_print_second:
                    print(
                        f"t={whole_second:2d}s phase={phase.name:<10} "
                        f"distance={distance:.4f}m touched={touched}"
                    )
                    last_print_second = whole_second

                if viewer is not None:
                    if not viewer.is_running():
                        print("Viewer closed before the sequence completed.")
                        return 2
                    demo.add_target_to_viewer(viewer, touched)
                    viewer.sync()
                    target_wall = start_wall + (
                        float(demo.data.time) - simulation_start
                    )
                    time.sleep(max(0.0, target_wall - time.perf_counter()))
    finally:
        if viewer is not None:
            viewer.close()

    hold_duration = touch_steps * demo.model.opt.timestep
    success = hold_duration >= args.required_hold
    print("\nDEMO RESULT")
    print(f"success={success}")
    print(f"minimum_distance={minimum_distance:.4f}m")
    print(f"touch_hold_duration={hold_duration:.3f}s")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv={args.csv}")
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scripted Unitree G1 reach-touch-hold-return demo."
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--target-radius", type=float, default=0.045)
    parser.add_argument("--required-hold", type=float, default=1.5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if not args.scene.is_file():
        parser.error(f"scene does not exist: {args.scene}")
    if args.target_radius <= 0:
        parser.error("--target-radius must be positive")
    if args.required_hold <= 0:
        parser.error("--required-hold must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
