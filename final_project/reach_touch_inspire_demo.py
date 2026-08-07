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
DEFAULT_MODEL = (
    ROOT
    / "assets/g1_inspire_fixed_base/g1_29dof_inspire_DFQ.urdf"
)

ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

LEFT_HAND_JOINTS = (
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
)

REST_ARM = np.array(
    # Elbow bends the arm down; wrist yaw compensates for the Inspire mount so
    # the relaxed hand continues along the forearm without palm-up roll.
    [0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 1.5], dtype=np.float64
)
TOUCH_ARM = np.array(
    # Wrist yaw compensates for the Inspire mounting-frame rotation so the
    # palm and index finger continue naturally along the forearm direction.
    [-1.2, 0.3, 0.0, 1.2, 0.0, 0.0, 1.5], dtype=np.float64
)

# The URDF zero pose is a mechanically flat calibration pose, not a natural
# resting hand. A small curl makes the default hand look relaxed.
RELAXED_HAND = np.array(
    [0.95, 0.22, 0.35, 0.53, 0.35, 0.35, 0.48, 0.48,
     0.55, 0.55, 0.62, 0.62],
    dtype=np.float64,
)

# Natural resting pose for the inactive right hand: thumb tucked toward the
# palm and progressively more curl from index to little finger.
RIGHT_NATURAL_HAND = np.array(
    [1.25, 0.30, 0.48, 0.72, 0.20, 0.20, 0.25, 0.25,
     0.30, 0.30, 0.35, 0.35],
    dtype=np.float64,
)

# Thumb and three non-pointing fingers curl; both index joints stay almost
# straight. Moderate flexion looks more like pointing than a tightly clenched
# fist.
POINTING_HAND = np.array(
    [1.20, 0.50, 0.80, 1.20, 0.03, 0.03, 1.40, 1.40,
     1.40, 1.40, 1.40, 1.40],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Phase:
    name: str
    duration: float
    arm_start: np.ndarray
    arm_end: np.ndarray
    hand_start: np.ndarray
    hand_end: np.ndarray


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def actuator_limit(joint_name: str) -> float:
    if joint_name.startswith(("L_", "R_")):
        return 1.0
    if "knee" in joint_name:
        return 139.0
    if any(part in joint_name for part in ("hip", "waist")):
        return 88.0
    if "wrist" in joint_name:
        return 5.0
    return 25.0


def build_model(urdf: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(urdf.resolve()))

    # The right hand is inactive in this task. Use the original G1 human-shaped
    # rubber-hand visual there, while retaining the articulated Inspire model
    # on the animated left side.
    rubber_hand_path = (
        ROOT / "assets/g1_fixed_base/meshes/right_rubber_hand.STL"
    )
    spec.add_mesh(
        name="right_rubber_hand_display",
        file=str(rubber_hand_path.resolve()),
    )
    spec.body("right_wrist_yaw_link").add_geom(
        name="right_rubber_hand_display",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        pos=[0.0415, 0.003, 0.0],
        meshname="right_rubber_hand_display",
        contype=0,
        conaffinity=0,
        rgba=[0.7, 0.7, 0.7, 1.0],
    )

    # Recreate the original demo's blue checkerboard environment. A URDF only
    # describes the robot, so scene assets must be added separately.
    spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    spec.add_material(
        name="groundplane_material",
        textures=["groundplane"],
        texuniform=True,
        texrepeat=[5.0, 5.0],
        reflectance=0.2,
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        material="groundplane_material",
    )
    # Explicit geometry lines guarantee a visible grid even on renderers that
    # do not display the URDF-added procedural checker texture correctly.
    for grid_index, coordinate in enumerate(np.linspace(-2.0, 2.0, 17)):
        spec.worldbody.add_geom(
            name=f"grid_x_{grid_index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[coordinate, 0.0, 0.002],
            size=[0.003, 2.0, 0.001],
            rgba=[0.18, 0.32, 0.42, 0.75],
        )
        spec.worldbody.add_geom(
            name=f"grid_y_{grid_index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, coordinate, 0.002],
            size=[2.0, 0.003, 0.001],
            rgba=[0.18, 0.32, 0.42, 0.75],
        )
    spec.worldbody.add_light(
        name="overhead",
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    index_body = spec.body("L_index_intermediate")
    index_body.add_site(
        name="left_index_tip",
        pos=[0.0, -0.041, 0.0],
        size=[0.006],
        rgba=[1.0, 0.2, 0.1, 1.0],
    )

    joint_model = spec.compile()
    joint_names = [
        mujoco.mj_id2name(
            joint_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        for joint_id in range(joint_model.njnt)
    ]
    for joint_name in joint_names:
        limit = actuator_limit(joint_name)
        actuator = spec.add_actuator(
            name=f"{joint_name}_motor",
            target=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=True,
            ctrlrange=[-limit, limit],
        )
        actuator.set_to_motor()

    model = spec.compile()

    # Hide only the articulated right Inspire meshes. Their joints remain in
    # the model for dimensional consistency, but the audience sees the clean
    # inactive G1 hand visual added above.
    for geom_id in range(model.ngeom):
        mesh_id = int(model.geom_dataid[geom_id])
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
        )
        if mesh_name == "R_hand_base_link" or (
            mesh_name is not None and mesh_name.endswith("_R")
        ):
            model.geom_rgba[geom_id, 3] = 0.0

    # A URDF root link is fused into MuJoCo's world body. Lift every direct
    # child and fused world geometry together to make the model fixed-base.
    for body_id in range(1, model.nbody):
        if model.body_parentid[body_id] == 0:
            model.body_pos[body_id, 2] += 0.80
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        is_scene_geom = geom_name == "floor" or (
            geom_name is not None and geom_name.startswith("grid_")
        )
        if model.geom_bodyid[geom_id] == 0 and not is_scene_geom:
            model.geom_pos[geom_id, 2] += 0.80

    # This is a kinematic-control baseline. Disabling contact prevents the
    # imported detailed hand collision meshes from fighting the pointing pose.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.opt.gravity[:] = 0.0
    return model


class InspireReachTouchDemo:
    def __init__(
        self,
        urdf: Path,
        target_radius: float = 0.045,
        target_visual_radius: float = 0.018,
    ) -> None:
        self.model = build_model(urdf)
        self.data = mujoco.MjData(self.model)
        self.target_radius = target_radius
        self.target_visual_radius = target_visual_radius
        self.arm_ids = self._joint_actuator_ids(ARM_JOINTS)
        self.hand_ids = self._joint_actuator_ids(LEFT_HAND_JOINTS)
        self.controlled_actuators = {
            actuator_id
            for _, _, actuator_id in self.arm_ids + self.hand_ids
        }
        self.all_actuators = self._all_actuator_mapping()
        self.hand_actuator_ids = {
            actuator_id
            for actuator_id in range(self.model.nu)
            if self.model.actuator(actuator_id).name.startswith(("L_", "R_"))
        }
        self.tip_site_id = self._id(
            mujoco.mjtObj.mjOBJ_SITE, "left_index_tip"
        )
        self.reset()
        self.target_position = self._position_for_pose(
            TOUCH_ARM, POINTING_HAND
        )

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object was not found: {name}")
        return object_id

    def _joint_actuator_ids(
        self, joint_names: tuple[str, ...]
    ) -> list[tuple[int, int, int]]:
        result = []
        for joint_name in joint_names:
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = self._id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_motor"
            )
            result.append(
                (
                    int(self.model.jnt_qposadr[joint_id]),
                    int(self.model.jnt_dofadr[joint_id]),
                    actuator_id,
                )
            )
        return result

    def _all_actuator_mapping(self) -> list[tuple[int, int, int]]:
        result = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            result.append(
                (
                    actuator_id,
                    int(self.model.jnt_qposadr[joint_id]),
                    int(self.model.jnt_dofadr[joint_id]),
                )
            )
        return result

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        # Put both arms and both hands in a presentation-friendly neutral pose
        # instead of the official URDF's mechanical zero configuration.
        for target, (qpos_id, _, _) in zip(REST_ARM, self.arm_ids):
            self.data.qpos[qpos_id] = target
        right_neutral = {
            "right_elbow_joint": 1.2,
            # With the hand pointing down, zero roll makes the right palm face
            # inward toward the left hand. Yaw compensates only for the Inspire
            # mounting-frame rotation; pitch stays zero to avoid wrist flexion.
            # Turn the palm away from the audience and slightly toward the
            # thigh while keeping the hand aligned with the forearm.
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }
        for joint_name, target in right_neutral.items():
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = target
        # Give only the inactive right fingers a natural resting curl. Right
        # arm/wrist posture and all animated left-hand settings are unchanged.
        for left_name, target in zip(LEFT_HAND_JOINTS, RIGHT_NATURAL_HAND):
            right_name = left_name.replace("L_", "R_", 1)
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, right_name)
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = target
        mujoco.mj_forward(self.model, self.data)
        self.hold_pose = self.data.qpos.copy()

    def fingertip_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site_id].copy()

    def _position_for_pose(
        self, arm_pose: np.ndarray, hand_pose: np.ndarray
    ) -> np.ndarray:
        saved = self.data.qpos.copy()
        for target, (qpos_id, _, _) in zip(arm_pose, self.arm_ids):
            self.data.qpos[qpos_id] = target
        for target, (qpos_id, _, _) in zip(hand_pose, self.hand_ids):
            self.data.qpos[qpos_id] = target
        mujoco.mj_forward(self.model, self.data)
        position = self.fingertip_position()
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return position

    def apply_control(
        self, arm_target: np.ndarray, hand_target: np.ndarray
    ) -> None:
        targets = {}
        for target, (_, _, actuator_id) in zip(arm_target, self.arm_ids):
            targets[actuator_id] = float(target)
        for target, (_, _, actuator_id) in zip(hand_target, self.hand_ids):
            targets[actuator_id] = float(target)

        for actuator_id, qpos_id, qvel_id in self.all_actuators:
            target = targets.get(actuator_id, float(self.hold_pose[qpos_id]))
            # This is a scripted joint-position baseline. Directly hold every
            # non-animated joint at its reset pose so the inactive right wrist
            # and fingers cannot drift during the left-arm sequence.
            self.data.qpos[qpos_id] = target
            self.data.qvel[qvel_id] = 0.0
            self.data.ctrl[actuator_id] = 0.0

    def add_target_to_viewer(self, viewer: object, touched: bool) -> None:
        scene = viewer.user_scn
        scene.ngeom = 2
        solid_color = (
            np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
            if touched
            else np.array([0.1, 0.65, 1.0, 0.9], dtype=np.float32)
        )
        mujoco.mjv_initGeom(
            scene.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, self.target_visual_radius),
            self.target_position,
            np.eye(3).reshape(-1),
            solid_color,
        )
        mujoco.mjv_initGeom(
            scene.geoms[1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, self.target_radius),
            self.target_position,
            np.eye(3).reshape(-1),
            np.array([0.1, 0.65, 1.0, 0.08], dtype=np.float32),
        )


def run(args: argparse.Namespace) -> int:
    demo = InspireReachTouchDemo(
        args.model, args.target_radius, args.target_visual_radius
    )
    phases = (
        Phase(
            "REST", 1.0, REST_ARM, REST_ARM, RELAXED_HAND, RELAXED_HAND
        ),
        Phase(
            "POINT",
            1.5,
            REST_ARM,
            REST_ARM,
            RELAXED_HAND,
            POINTING_HAND,
        ),
        Phase(
            "REACH",
            3.0,
            REST_ARM,
            TOUCH_ARM,
            POINTING_HAND,
            POINTING_HAND,
        ),
        Phase(
            "TOUCH/HOLD",
            2.0,
            TOUCH_ARM,
            TOUCH_ARM,
            POINTING_HAND,
            POINTING_HAND,
        ),
        Phase(
            "RETURN",
            3.0,
            TOUCH_ARM,
            REST_ARM,
            POINTING_HAND,
            POINTING_HAND,
        ),
        Phase(
            "RELAX",
            1.5,
            REST_ARM,
            REST_ARM,
            POINTING_HAND,
            RELAXED_HAND,
        ),
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

    print("DEMO TYPE: scripted joint-position baseline (not trained RL)")
    print("HAND: Inspire DFQ five-finger hand")
    print("ENDPOINT: actuated left index fingertip site")
    print(f"model_nq={demo.model.nq} model_nu={demo.model.nu}")
    print(f"target_xyz={np.round(demo.target_position, 4).tolist()}")
    try:
        for phase in phases:
            phase_start = float(demo.data.time)
            print(f"\nPHASE {phase.name}")
            while float(demo.data.time) - phase_start < phase.duration:
                elapsed = float(demo.data.time) - phase_start
                blend = smoothstep(elapsed / phase.duration)
                arm = phase.arm_start + blend * (
                    phase.arm_end - phase.arm_start
                )
                hand = phase.hand_start + blend * (
                    phase.hand_end - phase.hand_start
                )
                demo.apply_control(arm, hand)
                mujoco.mj_step(demo.model, demo.data)

                fingertip = demo.fingertip_position()
                distance = float(
                    np.linalg.norm(fingertip - demo.target_position)
                )
                touched = distance <= args.target_radius
                minimum_distance = min(minimum_distance, distance)
                if phase.name == "TOUCH/HOLD" and touched:
                    touch_steps += 1

                rows.append(
                    {
                        "time": float(demo.data.time) - simulation_start,
                        "phase": phase.name,
                        "fingertip_x": fingertip[0],
                        "fingertip_y": fingertip[1],
                        "fingertip_z": fingertip[2],
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
        description="G1 Inspire five-finger point-and-touch demo."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--target-radius", type=float, default=0.045)
    parser.add_argument("--target-visual-radius", type=float, default=0.018)
    parser.add_argument("--required-hold", type=float, default=1.5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if args.target_radius <= 0 or args.target_visual_radius <= 0:
        parser.error("target radii must be positive")
    if args.required_hold <= 0:
        parser.error("--required-hold must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
