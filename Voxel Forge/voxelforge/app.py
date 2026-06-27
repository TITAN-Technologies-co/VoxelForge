import io
import hashlib
import math
import os
import re
import shlex
import sqlite3
import sys
import tempfile
import time
from urllib import parse as urlparse
from urllib import request as urlrequest

import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

from .config import DB_PATH, logger
from .editor import VFCodeEditor, VFCodeHighlighter
from .engine2d import Engine2D, Scene2D, Sprite2D, Transform2D, Vec2
from .storage import VoxelForgeStore
from .studio2d import Engine2DStudioDialog
from .worker import VFWorker


class HeightMapViewer(QtWidgets.QMainWindow):
    def __init__(self): 
        super().__init__()
        self.setWindowTitle("Voxel Forge")
        self.setGeometry(80, 80, 1320, 860)

        self.height_data = None
        self.alpha_data = None
        self.image_rgb = None
        self.fg_mask = None
        self.part_labels = None
        self.part_sizes = {}

        self.mesh_item = None
        self.mesh_base_vertices = None
        self.mesh_faces = None
        self.face_part_ids = None
        self.face_source_colors = None
        self.secondary_mesh_item = None
        self.secondary_base_vertices = None
        self.secondary_faces = None
        self.secondary_face_colors = None
        self.secondary_rot = np.zeros(3, dtype=np.float32)
        self.extra_modules = []
        self.selected_target = ("module", "primary")
        self.preview_mesh_item = None

        self.part_colors = {}
        self.part_transforms = {}
        self.base_color = QtGui.QColor(50, 160, 255)
        self.ground_color = QtGui.QColor(128, 128, 128)
        self.sky_color = QtGui.QColor(20, 20, 20)
        self.code_script_text = ""
        self.current_vf_file_path = ""
        self.vf_classes = {}
        self.vf_modules = {}
        self.vf_assets = {"picture": {}, "audio": {}, "video": {}, "model": {}}
        self.vf_timeline_events = []
        self.vf_timeline_start_time = 0.0
        self.store = VoxelForgeStore(DB_PATH)
        self.database_path = str(DB_PATH)
        self.current_user = None
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.english_locale = QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates)
        QtCore.QLocale.setDefault(self.english_locale)
        self.setLocale(self.english_locale)

        self.image_resolution = 96
        self.ground_size = 500
        self.auto_start_animation = False
        self.theme_mode = "Dark"
        self.resume_last_session = True

        self.animation_translation = np.zeros(3, dtype=np.float32)
        self.animation_rotation = np.zeros(3, dtype=np.float32)
        self._last_anim_time = time.perf_counter()
        self.orbit_enabled = False
        self.orbit_module = "Primary"
        self.orbit_radius = 120.0
        self.orbit_speed = 20.0
        self.orbit_plane = "XZ"
        self.orbit_spin_axes = "Y"
        self.orbit_spin_speed = 25.0
        self.orbit_angle = 0.0
        self.orbit_spin_angle = 0.0
        self.target_fps = 240
        self._fps_value = 0.0
        self._fps_frames = 0
        self._fps_last_time = time.perf_counter()
        self._syncing_main_controls = False
        self.pressed_nav_keys = set()
        self.walk_speed = 120.0
        self.turn_speed = 120.0
        self._last_nav_time = time.perf_counter()

        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = 280
        self.view.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setCentralWidget(self.view)
        self.setAcceptDrops(True)
        self.view.installEventFilter(self)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.grid = gl.GLGridItem()
        if hasattr(self.grid, "setSpacing"):
            self.grid.setSpacing(10, 10, 1)
        self.view.addItem(self.grid)
        self._set_ground_size(self.ground_size)
        self._apply_scene_colors()

        axis = gl.GLAxisItem()
        axis.setSize(60, 60, 60)
        self.view.addItem(axis)

        self.fps_label = QtWidgets.QLabel(self.view)
        self.fps_label.setText("FPS: 60")
        self.fps_label.setStyleSheet(
            "QLabel { background: rgba(0,0,0,160); color: #00ff90; padding: 6px; border-radius: 6px; font-weight: 700; }"
        )
        self.fps_label.move(10, 10)
        self.fps_label.resize(180, 28)
        self.fps_label.show()

        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.animation_timer.setInterval(max(1, int(round(1000.0 / float(self.target_fps)))))
        self.animation_timer.timeout.connect(self._tick_animation)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(16)
        self.preview_timer.timeout.connect(self._tick_preview)
        self.nav_timer = QtCore.QTimer(self)
        self.nav_timer.setInterval(16)
        self.nav_timer.timeout.connect(self._tick_navigation)
        self.nav_timer.start()
        self.vf_timeline_timer = QtCore.QTimer(self)
        self.vf_timeline_timer.setInterval(20)
        self.vf_timeline_timer.timeout.connect(self._tick_vf_timeline)
        self.preview_angle = 0.0
        self.preview_timer.start()

        self._build_controls()
        self._build_menu()
        self._load_app_settings()
        self._apply_theme()
        if self.resume_last_session:
            self._resume_last_session_on_startup()
        QtCore.QTimer.singleShot(0, self._ensure_user_session)

    def _build_menu(self):
        open_image = QtWidgets.QAction("Open Image", self)
        open_image.triggered.connect(self.load_image)
        open_second_image = QtWidgets.QAction("Add Second Image", self)
        open_second_image.triggered.connect(self.load_second_image)

        save_model = QtWidgets.QAction("Save Private Model", self)
        save_model.triggered.connect(self.save_private_model)

        load_model = QtWidgets.QAction("Load Private Model", self)
        load_model.triggered.connect(self.load_private_model)
        load_second_model = QtWidgets.QAction("Load Second Model", self)
        load_second_model.triggered.connect(self.load_second_private_model)

        resume_last_action = QtWidgets.QAction("Resume Last Session", self)
        resume_last_action.triggered.connect(self._resume_last_session_manual)

        settings_action = QtWidgets.QAction("Program Settings", self)
        settings_action.triggered.connect(self._open_settings_dialog)

        save_settings_file_action = QtWidgets.QAction("Save Settings to Database", self)
        save_settings_file_action.triggered.connect(self._save_settings_file_dialog)

        load_settings_file_action = QtWidgets.QAction("Reload Settings from Database", self)
        load_settings_file_action.triggered.connect(self._load_settings_file_dialog)

        reset_settings_action = QtWidgets.QAction("Reset Settings", self)
        reset_settings_action.triggered.connect(self._reset_all_settings)

        anim_start_action = QtWidgets.QAction("Start Animation", self)
        anim_start_action.triggered.connect(lambda: self.keep_moving_check.setChecked(True))

        anim_stop_action = QtWidgets.QAction("Stop Animation", self)
        anim_stop_action.triggered.connect(lambda: self.keep_moving_check.setChecked(False))

        anim_reset_action = QtWidgets.QAction("Reset Animation Drift", self)
        anim_reset_action.triggered.connect(self._stop_animation_and_reset_drift)

        code_mode_action = QtWidgets.QAction("Open Code Mode", self)
        code_mode_action.setShortcut(QtGui.QKeySequence("F6"))
        code_mode_action.triggered.connect(self._open_code_mode)

        studio_2d_action = QtWidgets.QAction("Open 2D Coding Studio", self)
        studio_2d_action.setShortcut(QtGui.QKeySequence("F7"))
        studio_2d_action.triggered.connect(self._open_2d_coding_studio)

        fullscreen_action = QtWidgets.QAction("Toggle Full Screen", self)
        fullscreen_action.setShortcut(QtGui.QKeySequence("F11"))
        fullscreen_action.triggered.connect(self._toggle_fullscreen)

        engine_2d_action = QtWidgets.QAction("Open 2D Engine Demo", self)
        engine_2d_action.triggered.connect(self._open_2d_engine_demo)

        account_action = QtWidgets.QAction("Account", self)
        account_action.triggered.connect(lambda: self._open_account_dialog(force=True))

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(open_image)
        file_menu.addAction(open_second_image)
        file_menu.addAction(save_model)
        file_menu.addAction(load_model)
        file_menu.addAction(load_second_model)
        file_menu.addAction(resume_last_action)
        file_menu.addAction(save_settings_file_action)
        file_menu.addAction(load_settings_file_action)

        settings_menu = menubar.addMenu("Settings")
        settings_menu.addAction(settings_action)
        settings_menu.addAction(reset_settings_action)

        account_menu = menubar.addMenu("Account")
        account_menu.addAction(account_action)

        animation_menu = menubar.addMenu("Animation")
        animation_menu.addAction(anim_start_action)
        animation_menu.addAction(anim_stop_action)
        animation_menu.addAction(anim_reset_action)

        code_menu = menubar.addMenu("Code")
        code_menu.addAction(code_mode_action)
        code_menu.addAction(studio_2d_action)

        view_menu = menubar.addMenu("View")
        view_menu.addAction(fullscreen_action)
        view_menu.addAction(engine_2d_action)

    def _open_2d_coding_studio(self):
        dialog = Engine2DStudioDialog(self)
        dialog.exec_()

    def _open_2d_engine_demo(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("VoxelForge 2D Engine Demo")
        dialog.resize(760, 480)
        layout = QtWidgets.QVBoxLayout(dialog)

        scene = Scene2D("Demo")
        scene.camera.viewport = Vec2(720, 380)
        scene.add_layer("background", order=0, parallax=Vec2(0.35, 0.35))
        scene.add_layer("actors", order=10)
        scene.add_sprite(
            "background",
            Sprite2D(
                "grid",
                size=Vec2(720, 380),
                texture="grid",
                metadata={"color": "#16202A"},
            ),
        )
        scene.add_sprite(
            "actors",
            Sprite2D(
                "player",
                size=Vec2(48, 48),
                transform=Transform2D(position=Vec2(40, 120)),
                velocity=Vec2(120, 0),
                texture="player",
                solid=True,
                tags={"actor"},
                metadata={"color": "#43B5FF"},
            ),
        )
        scene.add_sprite(
            "actors",
            Sprite2D(
                "crate",
                size=Vec2(64, 64),
                transform=Transform2D(position=Vec2(350, 120)),
                texture="crate",
                solid=True,
                tags={"actor"},
                metadata={"color": "#F26F4C"},
            ),
        )
        engine = Engine2D()
        engine.load_scene(scene)

        graphics_scene = QtWidgets.QGraphicsScene(dialog)
        graphics_scene.setSceneRect(0, 0, 720, 380)
        graphics_view = QtWidgets.QGraphicsView(graphics_scene)
        graphics_view.setRenderHint(QtGui.QPainter.Antialiasing)
        graphics_view.setMinimumHeight(380)
        graphics_view.setFrameShape(QtWidgets.QFrame.NoFrame)
        layout.addWidget(graphics_view)

        status = QtWidgets.QLabel("2D engine running")
        status.setStyleSheet("QLabel { font-weight: 700; color: #8BC34A; }")
        layout.addWidget(status)

        timer = QtCore.QTimer(dialog)
        timer.setInterval(16)

        def _paint_plan():
            player = scene.sprite("player")
            if player is not None:
                bounds = player.bounds()
                if bounds.right > 680 or bounds.left < 20:
                    player.velocity = Vec2(-player.velocity.x, player.velocity.y)
            engine.update(timer.interval() / 1000.0)
            graphics_scene.clear()
            graphics_scene.setBackgroundBrush(QtGui.QColor("#101418"))
            collisions = scene.collision_pairs(tag="actor")
            colliding = {sprite.name for pair in collisions for sprite in pair}
            for command in engine.render_plan():
                color = command.metadata.get("color", "#43B5FF")
                if command.sprite in colliding:
                    color = "#FFD166"
                rect = command.bounds
                item = graphics_scene.addRect(
                    rect.x,
                    rect.y,
                    rect.width,
                    rect.height,
                    QtGui.QPen(QtGui.QColor("#FFFFFF"), 1),
                    QtGui.QBrush(QtGui.QColor(color)),
                )
                item.setZValue(command.z_index)
                text = graphics_scene.addText(command.sprite)
                text.setDefaultTextColor(QtGui.QColor("#F5F5F5"))
                text.setPos(rect.x + 6, rect.y + 6)
                text.setZValue(command.z_index + 1)
            status.setText(
                f"sprites: {len(engine.render_plan())} | collisions: {len(collisions)} | time: {scene.time:.2f}s"
            )

        timer.timeout.connect(_paint_plan)
        dialog.finished.connect(timer.stop)
        _paint_plan()
        timer.start()
        dialog.exec_()

    def _build_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        controls = QtWidgets.QWidget()
        controls.setLocale(self.english_locale)
        form = QtWidgets.QFormLayout(controls)

        form.addRow(self._make_section_label("Model Tools"))

        self.height_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.height_slider.setRange(2, 80)
        self.height_slider.setValue(25)
        self.height_slider.valueChanged.connect(self._rebuild_mesh)
        form.addRow("Extrude Height", self.height_slider)

        self.remove_bg_check = QtWidgets.QCheckBox("Remove Background")
        self.remove_bg_check.setChecked(True)
        self.remove_bg_check.stateChanged.connect(self._rebuild_mesh)
        form.addRow(self.remove_bg_check)

        self.bg_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bg_threshold_slider.setRange(0, 255)
        self.bg_threshold_slider.setValue(245)
        self.bg_threshold_slider.valueChanged.connect(self._rebuild_mesh)
        form.addRow("BG Threshold", self.bg_threshold_slider)

        self.part_gap_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.part_gap_slider.setRange(0, 50)
        self.part_gap_slider.setValue(0)
        self.part_gap_slider.valueChanged.connect(self._refresh_mesh_visual)
        form.addRow("Parts Separation", self.part_gap_slider)

        self.color_mode = QtWidgets.QComboBox()
        self.color_mode.addItems(["Original Image", "Solid", "Radial Gradient", "Per-Part"])
        self.color_mode.setCurrentText("Original Image")
        self.color_mode.currentIndexChanged.connect(self._refresh_mesh_visual)
        form.addRow("Color Mode", self.color_mode)

        solid_color_btn = QtWidgets.QPushButton("Choose Solid Color")
        solid_color_btn.clicked.connect(self._choose_color)
        form.addRow(solid_color_btn)

        self.parts_list = QtWidgets.QListWidget()
        self.parts_list.setMinimumHeight(150)
        self.parts_list.currentItemChanged.connect(self._on_part_selection_changed)
        form.addRow("Detected Parts", self.parts_list)

        part_color_btn = QtWidgets.QPushButton("Set Selected Part Color")
        part_color_btn.clicked.connect(self._set_selected_part_color)
        form.addRow(part_color_btn)

        self.part_move_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.part_move_x_slider.setRange(-200, 200)
        self.part_move_x_slider.valueChanged.connect(self._update_selected_part_transform)
        form.addRow("Part Move X", self.part_move_x_slider)

        self.part_move_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.part_move_y_slider.setRange(-200, 200)
        self.part_move_y_slider.valueChanged.connect(self._update_selected_part_transform)
        form.addRow("Part Move Y", self.part_move_y_slider)

        self.part_move_z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.part_move_z_slider.setRange(-200, 300)
        self.part_move_z_slider.valueChanged.connect(self._update_selected_part_transform)
        form.addRow("Part Move Z", self.part_move_z_slider)

        reset_part_btn = QtWidgets.QPushButton("Reset Selected Part Transform")
        reset_part_btn.clicked.connect(self._reset_selected_part_transform)
        form.addRow(reset_part_btn)

        form.addRow(self._make_section_label("Scene"))
        self.ground_size_spin = QtWidgets.QSpinBox()
        self.ground_size_spin.setRange(50, 5000)
        self.ground_size_spin.setSingleStep(10)
        self.ground_size_spin.setValue(int(self.ground_size))
        self.ground_size_spin.valueChanged.connect(self._on_ground_size_spin_changed)
        form.addRow("Ground Size", self.ground_size_spin)

        ground_color_btn = QtWidgets.QPushButton("Choose Ground Color")
        ground_color_btn.clicked.connect(self._choose_ground_color)
        form.addRow(ground_color_btn)

        sky_color_btn = QtWidgets.QPushButton("Choose Sky Color")
        sky_color_btn.clicked.connect(self._choose_sky_color)
        form.addRow(sky_color_btn)

        url_image_btn = QtWidgets.QPushButton("Load Image URL")
        url_image_btn.clicked.connect(self._open_url_image_picker)
        form.addRow(url_image_btn)

        code_mode_btn = QtWidgets.QPushButton("Open Code Mode (F6)")
        code_mode_btn.clicked.connect(self._open_code_mode)
        form.addRow(code_mode_btn)

        form.addRow(self._make_section_label("Global Transform"))

        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(20, 300)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self._apply_transform)
        form.addRow("Scale %", self.scale_slider)

        self.move_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.move_x_slider.setRange(-300, 300)
        self.move_x_slider.valueChanged.connect(self._apply_transform)
        form.addRow("Move X", self.move_x_slider)

        self.move_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.move_y_slider.setRange(-300, 300)
        self.move_y_slider.valueChanged.connect(self._apply_transform)
        form.addRow("Move Y", self.move_y_slider)

        self.move_z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.move_z_slider.setRange(0, 300)
        self.move_z_slider.valueChanged.connect(self._apply_transform)
        form.addRow("Move Z", self.move_z_slider)

        self.rot_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rot_x_slider.setRange(-180, 180)
        self.rot_x_slider.valueChanged.connect(self._apply_transform)
        form.addRow("Rotate X", self.rot_x_slider)

        self.rot_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rot_y_slider.setRange(-180, 180)
        self.rot_y_slider.valueChanged.connect(self._apply_transform)
        form.addRow("Rotate Y", self.rot_y_slider)

        self.rot_z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rot_z_slider.setRange(-180, 180)
        self.rot_z_slider.valueChanged.connect(self._apply_transform)
        form.addRow("Rotate Z", self.rot_z_slider)

        form.addRow(self._make_section_label("Animation Tools"))

        self.keep_moving_check = QtWidgets.QCheckBox("Keep Moving (Animation)")
        self.keep_moving_check.stateChanged.connect(self._toggle_animation)
        form.addRow(self.keep_moving_check)

        self.anim_move_x_speed = QtWidgets.QDoubleSpinBox()
        self.anim_move_x_speed.setRange(-200.0, 200.0)
        self.anim_move_x_speed.setDecimals(1)
        self.anim_move_x_speed.setSingleStep(1.0)
        self.anim_move_x_speed.setSuffix(" u/s")
        self.anim_move_x_speed.valueChanged.connect(self._on_animation_params_changed)
        form.addRow("Anim Move X", self.anim_move_x_speed)

        self.anim_move_y_speed = QtWidgets.QDoubleSpinBox()
        self.anim_move_y_speed.setRange(-200.0, 200.0)
        self.anim_move_y_speed.setDecimals(1)
        self.anim_move_y_speed.setSingleStep(1.0)
        self.anim_move_y_speed.setSuffix(" u/s")
        self.anim_move_y_speed.valueChanged.connect(self._on_animation_params_changed)
        form.addRow("Anim Move Y", self.anim_move_y_speed)

        self.anim_move_z_speed = QtWidgets.QDoubleSpinBox()
        self.anim_move_z_speed.setRange(-200.0, 200.0)
        self.anim_move_z_speed.setDecimals(1)
        self.anim_move_z_speed.setSingleStep(1.0)
        self.anim_move_z_speed.setSuffix(" u/s")
        self.anim_move_z_speed.valueChanged.connect(self._on_animation_params_changed)
        form.addRow("Anim Move Z", self.anim_move_z_speed)

        self.anim_rot_x_speed = QtWidgets.QDoubleSpinBox()
        self.anim_rot_x_speed.setRange(-360.0, 360.0)
        self.anim_rot_x_speed.setDecimals(1)
        self.anim_rot_x_speed.setSingleStep(1.0)
        self.anim_rot_x_speed.setSuffix(" deg/s")
        self.anim_rot_x_speed.valueChanged.connect(self._on_animation_params_changed)
        form.addRow("Anim Rotate X", self.anim_rot_x_speed)

        self.anim_rot_y_speed = QtWidgets.QDoubleSpinBox()
        self.anim_rot_y_speed.setRange(-360.0, 360.0)
        self.anim_rot_y_speed.setDecimals(1)
        self.anim_rot_y_speed.setSingleStep(1.0)
        self.anim_rot_y_speed.setSuffix(" deg/s")
        self.anim_rot_y_speed.valueChanged.connect(self._on_animation_params_changed)
        self.anim_rot_y_speed.setValue(30.0)
        form.addRow("Anim Rotate Y", self.anim_rot_y_speed)

        self.anim_rot_z_speed = QtWidgets.QDoubleSpinBox()
        self.anim_rot_z_speed.setRange(-360.0, 360.0)
        self.anim_rot_z_speed.setDecimals(1)
        self.anim_rot_z_speed.setSingleStep(1.0)
        self.anim_rot_z_speed.setSuffix(" deg/s")
        self.anim_rot_z_speed.valueChanged.connect(self._on_animation_params_changed)
        form.addRow("Anim Rotate Z", self.anim_rot_z_speed)

        stop_anim_btn = QtWidgets.QPushButton("Stop Animation + Reset Drift")
        stop_anim_btn.clicked.connect(self._stop_animation_and_reset_drift)
        form.addRow(stop_anim_btn)

        form.addRow(self._make_section_label("Account + Storage"))

        account_btn = QtWidgets.QPushButton("Account")
        account_btn.clicked.connect(lambda: self._open_account_dialog(force=True))
        form.addRow(account_btn)

        self.account_label = QtWidgets.QLabel("Not signed in")
        self.account_label.setWordWrap(True)
        form.addRow("User", self.account_label)

        self.database_label = QtWidgets.QLineEdit(self.database_path)
        self.database_label.setReadOnly(True)
        form.addRow("Database", self.database_label)

        form.addRow(self._make_section_label("Settings"))

        save_settings_btn = QtWidgets.QPushButton("Save Settings to Database")
        save_settings_btn.clicked.connect(self._save_settings_file_dialog)
        form.addRow(save_settings_btn)

        load_settings_btn = QtWidgets.QPushButton("Reload Settings from Database")
        load_settings_btn.clicked.connect(self._load_settings_file_dialog)
        form.addRow(load_settings_btn)

        reset_settings_btn = QtWidgets.QPushButton("Reset All Settings")
        reset_settings_btn.clicked.connect(self._reset_all_settings)
        form.addRow(reset_settings_btn)

        form.addRow(self._make_section_label("Second Module"))

        self.second_scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.second_scale_slider.setRange(20, 300)
        self.second_scale_slider.setValue(100)
        self.second_scale_slider.valueChanged.connect(self._apply_secondary_transform)
        form.addRow("Second Scale %", self.second_scale_slider)

        self.second_move_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.second_move_x_slider.setRange(-600, 600)
        self.second_move_x_slider.setValue(180)
        self.second_move_x_slider.valueChanged.connect(self._apply_secondary_transform)
        form.addRow("Second Move X", self.second_move_x_slider)

        self.second_move_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.second_move_y_slider.setRange(-600, 600)
        self.second_move_y_slider.valueChanged.connect(self._apply_secondary_transform)
        form.addRow("Second Move Y", self.second_move_y_slider)

        self.second_move_z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.second_move_z_slider.setRange(-200, 600)
        self.second_move_z_slider.valueChanged.connect(self._apply_secondary_transform)
        form.addRow("Second Move Z", self.second_move_z_slider)

        clear_second_btn = QtWidgets.QPushButton("Clear Second Module")
        clear_second_btn.clicked.connect(self._clear_secondary_module)
        form.addRow(clear_second_btn)

        self.modules_list = QtWidgets.QListWidget()
        self.modules_list.setMinimumHeight(90)
        self.modules_list.currentItemChanged.connect(self._on_module_selection_changed)
        self.modules_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.modules_list.customContextMenuRequested.connect(self._on_modules_context_menu)
        form.addRow("Modules", self.modules_list)

        apply_main_btn = QtWidgets.QPushButton("Apply Main Controls To Selected Module")
        apply_main_btn.clicked.connect(self._apply_main_controls_to_selected_module)
        form.addRow(apply_main_btn)

        self.link_main_to_selected = QtWidgets.QCheckBox("Link Main Controls To Selected Module")
        form.addRow(self.link_main_to_selected)

        self.parts_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.parts_list.customContextMenuRequested.connect(self._on_parts_context_menu)

        form.addRow(self._make_section_label("Copyable Indicators"))
        self.indicator_move_x = QtWidgets.QLineEdit("0")
        self.indicator_move_y = QtWidgets.QLineEdit("0")
        self.indicator_move_z = QtWidgets.QLineEdit("0")
        self.indicator_rot_x = QtWidgets.QLineEdit("0")
        self.indicator_rot_y = QtWidgets.QLineEdit("0")
        self.indicator_rot_z = QtWidgets.QLineEdit("0")
        self.indicator_scale = QtWidgets.QLineEdit("100")
        form.addRow("Move X Value", self.indicator_move_x)
        form.addRow("Move Y Value", self.indicator_move_y)
        form.addRow("Move Z Value", self.indicator_move_z)
        form.addRow("Rotate X Value", self.indicator_rot_x)
        form.addRow("Rotate Y Value", self.indicator_rot_y)
        form.addRow("Rotate Z Value", self.indicator_rot_z)
        form.addRow("Scale Value", self.indicator_scale)

        fullscreen_btn = QtWidgets.QPushButton("Toggle Full Screen (F11)")
        fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        form.addRow(fullscreen_btn)

        form.addRow(self._make_section_label("Inspector Preview"))
        self.preview_title = QtWidgets.QLabel("Selected: Primary Module")
        form.addRow(self.preview_title)
        self.preview_view = gl.GLViewWidget()
        self.preview_view.setMinimumHeight(200)
        self.preview_view.opts["distance"] = 140
        preview_grid = gl.GLGridItem()
        preview_grid.scale(5, 5, 1)
        preview_grid.setSize(120, 120, 1)
        self.preview_view.addItem(preview_grid)
        form.addRow(self.preview_view)

        reset_btn = QtWidgets.QPushButton("Reset Transform")
        reset_btn.clicked.connect(self._reset_transform)
        form.addRow(reset_btn)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _make_section_label(self, text):
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("font-weight: 700; color: #d4af37; padding-top: 8px;")
        label.setLocale(self.english_locale)
        return label

    def _color_to_hex(self, color):
        return color.name().upper() if isinstance(color, QtGui.QColor) else "#000000"

    def _color_from_value(self, value, fallback):
        if isinstance(value, QtGui.QColor):
            return QtGui.QColor(value)
        c = QtGui.QColor(str(value))
        if c.isValid():
            return c
        return QtGui.QColor(fallback)

    def _set_ground_size(self, value):
        size = max(50, int(value))
        self.ground_size = size
        self.grid.setSize(size, size, 1)
        if hasattr(self, "ground_size_spin"):
            self.ground_size_spin.blockSignals(True)
            self.ground_size_spin.setValue(size)
            self.ground_size_spin.blockSignals(False)

    def _apply_scene_colors(self):
        if hasattr(self.grid, "setColor"):
            self.grid.setColor(self.ground_color)
        if hasattr(self.view, "setBackgroundColor"):
            self.view.setBackgroundColor(self.sky_color)

    def _set_ground_color(self, color):
        c = QtGui.QColor(color)
        if not c.isValid():
            return
        self.ground_color = c
        self._apply_scene_colors()

    def _set_sky_color(self, color):
        c = QtGui.QColor(color)
        if not c.isValid():
            return
        self.sky_color = c
        self._apply_scene_colors()

    def _choose_ground_color(self):
        color = QtWidgets.QColorDialog.getColor(self.ground_color, self, "Choose Ground Color")
        if color.isValid():
            self._set_ground_color(color)

    def _choose_sky_color(self):
        color = QtWidgets.QColorDialog.getColor(self.sky_color, self, "Choose Sky Color")
        if color.isValid():
            self._set_sky_color(color)

    def _open_url_image_picker(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Load Image From URL")
        layout = QtWidgets.QFormLayout(dialog)

        url_edit = QtWidgets.QLineEdit()
        url_edit.setPlaceholderText("https://example.com/image.png")
        target_combo = QtWidgets.QComboBox()
        target_combo.addItems(["auto", "primary", "secondary", "add", "extra"])
        layout.addRow("Image URL", url_edit)
        layout.addRow("Target", target_combo)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        url = url_edit.text().strip()
        if not url:
            return
        target = self._script_parse_target(target_combo.currentText())
        progress = QtWidgets.QProgressDialog("Downloading image...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Load Image URL")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()

        worker = VFWorker(self._download_url_bytes, url, "image")

        def _done(result):
            progress.close()
            data, _ = result
            try:
                message = self._run_web_image_data(url, data, target)
                QtWidgets.QMessageBox.information(self, "Load Image URL", f"Success: {message}")
            except Exception as exc:
                logger.exception("Failed to apply downloaded image")
                QtWidgets.QMessageBox.warning(self, "Load Image URL", f"Failed:\n{exc}")

        def _failed(message):
            progress.close()
            QtWidgets.QMessageBox.warning(self, "Load Image URL", f"Failed:\n{message}")

        worker.signals.finished.connect(_done)
        worker.signals.failed.connect(_failed)
        self.thread_pool.start(worker)

    def _on_ground_size_spin_changed(self, value):
        self._set_ground_size(int(value))

    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        self._load_primary_image_from_path(file_path)

    def load_second_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Add Second Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        self._load_secondary_image_from_path(file_path)

    def _load_primary_image_from_path(self, file_path):
        self._store_local_asset(file_path, "image")
        img = Image.open(file_path).convert("RGBA")
        img = img.resize((self.image_resolution, self.image_resolution))
        self.height_data, self.alpha_data, self.image_rgb = self._rgba_to_height_alpha_rgb(
            np.asarray(img, dtype=np.uint8)
        )
        self._rebuild_mesh()

    def _rgba_to_height_alpha_rgb(self, rgba):
        rgb_u8 = rgba[:, :, :3]
        rgb = rgb_u8.astype(np.float32)
        height = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
        return height, rgba[:, :, 3], (rgb / 255.0).astype(np.float32)

    def _build_mesh_data_from_rgba(self, rgba):
        height, alpha, image_rgb = self._rgba_to_height_alpha_rgb(rgba)

        old_height = self.height_data
        old_alpha = self.alpha_data
        old_rgb = self.image_rgb
        try:
            self.height_data = height
            self.alpha_data = alpha
            self.image_rgb = image_rgb
            mask = self._compute_foreground_mask()
            labels, _ = self._label_connected_parts(mask)
            mesh = self._build_mesh_from_parts(height, mask, labels)
            return mesh
        finally:
            self.height_data = old_height
            self.alpha_data = old_alpha
            self.image_rgb = old_rgb

    def _load_secondary_image_from_path(self, file_path):
        self._store_local_asset(file_path, "image")
        img = Image.open(file_path).convert("RGBA")
        img = img.resize((self.image_resolution, self.image_resolution))
        rgba = np.array(img, dtype=np.uint8)
        mesh = self._build_mesh_data_from_rgba(rgba)
        if mesh[0] is None:
            return
        verts, faces, _, source_colors = mesh
        if self.secondary_mesh_item is None:
            self._set_secondary_mesh(verts, faces, source_colors)
        else:
            self._add_extra_module(verts, faces, source_colors)

    def _store_local_asset(self, file_path, kind):
        try:
            path = Path(file_path)
            if not path.is_file():
                return
            key_hash = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
            self.store.save_blob(
                f"local-{kind}:{key_hash}",
                path.read_bytes(),
                kind,
                user_id=self.current_user["id"] if self.current_user else None,
                content_type=f"{kind}/local",
            )
        except Exception:
            logger.exception("Failed to store local asset %s", file_path)

    def _set_secondary_mesh(self, verts, faces, colors):
        self.secondary_base_vertices = verts.astype(np.float32)
        self.secondary_faces = faces.astype(np.uint32)
        self.secondary_face_colors = colors.astype(np.float32) if colors is not None else np.ones((faces.shape[0], 3), dtype=np.float32)

        if self.secondary_mesh_item is not None:
            self.view.removeItem(self.secondary_mesh_item)
            self.secondary_mesh_item = None

        face_colors = np.ones((self.secondary_faces.shape[0], 4), dtype=np.float32)
        face_colors[:, :3] = self.secondary_face_colors[:, :3]
        self.secondary_mesh_item = gl.GLMeshItem(
            vertexes=self.secondary_base_vertices,
            faces=self.secondary_faces,
            faceColors=face_colors,
            smooth=False,
            drawEdges=False,
            shader="shaded",
        )
        self.secondary_mesh_item.setGLOptions("opaque")
        self.view.addItem(self.secondary_mesh_item)
        self._apply_secondary_transform()
        self.selected_target = ("module", "secondary")
        self._refresh_modules_list()
        self._update_preview_for_selection()

    def _add_extra_module(self, verts, faces, colors):
        face_colors = colors.astype(np.float32) if colors is not None else np.ones((faces.shape[0], 3), dtype=np.float32)
        item = gl.GLMeshItem(
            vertexes=verts.astype(np.float32),
            faces=faces.astype(np.uint32),
            faceColors=np.column_stack([face_colors, np.ones((face_colors.shape[0],), dtype=np.float32)]),
            smooth=False,
            drawEdges=False,
            shader="shaded",
        )
        item.setGLOptions("opaque")
        self.view.addItem(item)
        idx = len(self.extra_modules)
        module = {
            "mesh_item": item,
            "base_vertices": verts.astype(np.float32),
            "faces": faces.astype(np.uint32),
            "face_colors": face_colors,
            "transform": np.array([100.0, 260.0 + 120.0 * idx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        self.extra_modules.append(module)
        self._apply_extra_module_transform(idx)
        self.selected_target = ("module", ("extra", idx))
        self._refresh_modules_list()
        self._update_preview_for_selection()

    def _compute_foreground_mask(self):
        if self.height_data is None:
            return None

        rows, cols = self.height_data.shape
        mask = np.ones((rows, cols), dtype=bool)
        if self.remove_bg_check.isChecked():
            threshold = self.bg_threshold_slider.value()
            luminance = (self.height_data * 255.0).astype(np.uint8)
            color_mask = luminance < threshold
            if self.alpha_data is not None:
                alpha_mask = self.alpha_data > 20
                mask = color_mask & alpha_mask
            else:
                mask = color_mask

        if not mask.any():
            mask[:] = True
        return mask

    def _label_connected_parts(self, mask):
        rows, cols = mask.shape
        labels = np.full((rows, cols), -1, dtype=np.int32)
        part_id = 0
        part_sizes = {}

        for i in range(rows):
            for j in range(cols):
                if not mask[i, j] or labels[i, j] != -1:
                    continue

                stack = [(i, j)]
                labels[i, j] = part_id
                size = 0

                while stack:
                    y, x = stack.pop()
                    size += 1
                    if y > 0 and mask[y - 1, x] and labels[y - 1, x] == -1:
                        labels[y - 1, x] = part_id
                        stack.append((y - 1, x))
                    if y < rows - 1 and mask[y + 1, x] and labels[y + 1, x] == -1:
                        labels[y + 1, x] = part_id
                        stack.append((y + 1, x))
                    if x > 0 and mask[y, x - 1] and labels[y, x - 1] == -1:
                        labels[y, x - 1] = part_id
                        stack.append((y, x - 1))
                    if x < cols - 1 and mask[y, x + 1] and labels[y, x + 1] == -1:
                        labels[y, x + 1] = part_id
                        stack.append((y, x + 1))

                part_sizes[part_id] = size
                part_id += 1

        return labels, part_sizes

    def _default_part_color(self, part_id):
        hue = (part_id * 47) % 360
        return QtGui.QColor.fromHsv(hue, 220, 255)

    def _sync_part_colors(self, part_ids):
        if isinstance(part_ids, int):
            valid_ids = set(range(part_ids))
        else:
            valid_ids = {int(part_id) for part_id in part_ids}

        for part_id in sorted(valid_ids):
            if part_id not in self.part_colors:
                self.part_colors[part_id] = self._default_part_color(part_id)
            if part_id not in self.part_transforms:
                self.part_transforms[part_id] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        stale = [k for k in self.part_colors if k not in valid_ids]
        for key in stale:
            del self.part_colors[key]
        stale_transforms = [k for k in self.part_transforms if k not in valid_ids]
        for key in stale_transforms:
            del self.part_transforms[key]

    def _refresh_parts_list(self):
        self.parts_list.clear()
        for part_id in sorted(self.part_sizes.keys()):
            size = self.part_sizes[part_id]
            item = QtWidgets.QListWidgetItem(f"Part {part_id + 1} ({size} px)")
            item.setData(QtCore.Qt.UserRole, int(part_id))
            self.parts_list.addItem(item)

    def _add_triangle(self, verts, faces, part_ids, src_colors, a, b, c, part_id, src_color):
        idx = len(verts)
        verts.extend([a, b, c])
        faces.append([idx, idx + 1, idx + 2])
        part_ids.append(part_id)
        src_colors.append(src_color)

    def _build_mesh_from_parts(self, data, mask, labels):
        rows, cols = data.shape
        h_strength = float(self.height_slider.value())
        base = 1.0

        cell_count = int(mask.sum())
        if cell_count == 0:
            return None, None, None

        boundary_edges = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
        if rows > 1:
            boundary_edges += int((mask[1:, :] & ~mask[:-1, :]).sum())
            boundary_edges += int((mask[:-1, :] & ~mask[1:, :]).sum())
        if cols > 1:
            boundary_edges += int((mask[:, 1:] & ~mask[:, :-1]).sum())
            boundary_edges += int((mask[:, :-1] & ~mask[:, 1:]).sum())

        face_count = cell_count * 4 + boundary_edges * 2
        verts = np.empty((face_count * 3, 3), dtype=np.float32)
        faces = np.arange(face_count * 3, dtype=np.uint32).reshape(face_count, 3)
        part_ids = np.empty((face_count,), dtype=np.int32)
        src_colors = np.empty((face_count, 3), dtype=np.float32)
        face_idx = 0

        def add_triangle(a, b, c, part_id, src_color):
            nonlocal face_idx
            vert_idx = face_idx * 3
            verts[vert_idx] = a
            verts[vert_idx + 1] = b
            verts[vert_idx + 2] = c
            part_ids[face_idx] = part_id
            src_colors[face_idx] = src_color
            face_idx += 1

        for i in range(rows):
            for j in range(cols):
                if not mask[i, j]:
                    continue

                pid = int(labels[i, j])
                height = base + float(data[i, j]) * h_strength
                src_color = self.image_rgb[i, j] if self.image_rgb is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)

                x0 = float(j - cols / 2.0)
                x1 = x0 + 1.0
                y0 = float(i - rows / 2.0)
                y1 = y0 + 1.0
                z0 = 0.0
                z1 = height

                add_triangle((x0, y0, z1), (x1, y0, z1), (x0, y1, z1), pid, src_color)
                add_triangle((x1, y0, z1), (x1, y1, z1), (x0, y1, z1), pid, src_color)

                add_triangle((x0, y0, z0), (x0, y1, z0), (x1, y0, z0), pid, src_color)
                add_triangle((x1, y0, z0), (x0, y1, z0), (x1, y1, z0), pid, src_color)

                if i == 0 or not mask[i - 1, j]:
                    add_triangle((x0, y0, z0), (x1, y0, z0), (x0, y0, z1), pid, src_color)
                    add_triangle((x1, y0, z0), (x1, y0, z1), (x0, y0, z1), pid, src_color)
                if i == rows - 1 or not mask[i + 1, j]:
                    add_triangle((x0, y1, z0), (x0, y1, z1), (x1, y1, z0), pid, src_color)
                    add_triangle((x1, y1, z0), (x0, y1, z1), (x1, y1, z1), pid, src_color)
                if j == 0 or not mask[i, j - 1]:
                    add_triangle((x0, y0, z0), (x0, y1, z0), (x0, y0, z1), pid, src_color)
                    add_triangle((x0, y1, z0), (x0, y1, z1), (x0, y0, z1), pid, src_color)
                if j == cols - 1 or not mask[i, j + 1]:
                    add_triangle((x1, y0, z0), (x1, y0, z1), (x1, y1, z0), pid, src_color)
                    add_triangle((x1, y1, z0), (x1, y0, z1), (x1, y1, z1), pid, src_color)

        return (
            verts[:face_idx * 3],
            faces[:face_idx],
            part_ids[:face_idx],
            src_colors[:face_idx],
        )

    def _explode_parts(self, base_vertices, face_part_ids):
        gap = float(self.part_gap_slider.value()) * 0.15
        if gap <= 0.0:
            return base_vertices.copy()

        verts = base_vertices.copy()
        tri_count = face_part_ids.shape[0]
        if tri_count == 0:
            return verts

        centers = verts[self.mesh_faces].mean(axis=1)[:, :2]

        global_center = centers.mean(axis=0)
        unique_parts = np.unique(face_part_ids)
        offsets = {}
        for pid in unique_parts:
            part_centers = centers[face_part_ids == pid]
            if part_centers.size == 0:
                offsets[int(pid)] = np.array([0.0, 0.0], dtype=np.float32)
                continue
            c = part_centers.mean(axis=0)
            direction = c - global_center
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                offsets[int(pid)] = np.array([0.0, 0.0], dtype=np.float32)
            else:
                offsets[int(pid)] = direction / norm * gap

        for pid, off in offsets.items():
            if abs(float(off[0])) < 1e-6 and abs(float(off[1])) < 1e-6:
                continue
            idxs = np.unique(self.mesh_faces[face_part_ids == pid].ravel())
            verts[idxs, 0] += off[0]
            verts[idxs, 1] += off[1]

        return verts

    def _apply_part_transforms(self, base_vertices, face_part_ids):
        verts = base_vertices.copy()
        for pid in np.unique(face_part_ids):
            pid = int(pid)
            transform = self.part_transforms.get(pid)
            if transform is None:
                continue
            if float(np.linalg.norm(transform)) < 1e-6:
                continue
            idxs = np.unique(self.mesh_faces[face_part_ids == pid].ravel())
            verts[idxs] += transform
        return verts

    def _create_face_colors(self, vertices, faces, face_part_ids, source_colors):
        face_colors = np.ones((faces.shape[0], 4), dtype=np.float32)
        mode = self.color_mode.currentText()
        base = np.array(
            [self.base_color.redF(), self.base_color.greenF(), self.base_color.blueF()],
            dtype=np.float32,
        )

        if mode == "Original Image":
            if source_colors is not None and source_colors.shape[0] == faces.shape[0]:
                face_colors[:, :3] = source_colors
            else:
                face_colors[:, :3] = base
            return face_colors

        if mode == "Solid":
            face_colors[:, :3] = base
            return face_colors

        if mode == "Per-Part":
            for pid in np.unique(face_part_ids):
                c = self.part_colors.get(int(pid), self.base_color)
                rgb = np.array([c.redF(), c.greenF(), c.blueF()], dtype=np.float32)
                face_colors[face_part_ids == pid, :3] = rgb
            return face_colors

        centers = vertices[faces].mean(axis=1)
        radial = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
        radial_norm = radial / max(radial.max(), 1e-6)
        bright = np.clip(base + 0.35, 0.0, 1.0)
        dark = np.clip(base * 0.35, 0.0, 1.0)
        face_colors[:, :3] = bright * (1.0 - radial_norm[:, None]) + dark * radial_norm[:, None]
        return face_colors

    def _rebuild_mesh(self):
        if self.height_data is None:
            return

        mask = self._compute_foreground_mask()
        labels, part_sizes = self._label_connected_parts(mask)

        self.fg_mask = mask
        self.part_labels = labels
        self.part_sizes = part_sizes
        self._sync_part_colors(part_sizes.keys())
        self._refresh_parts_list()
        if self.parts_list.count() > 0 and self.parts_list.currentItem() is None:
            self.parts_list.setCurrentRow(0)

        mesh = self._build_mesh_from_parts(self.height_data, self.fg_mask, self.part_labels)
        if mesh[0] is None:
            return

        self.mesh_base_vertices, self.mesh_faces, self.face_part_ids, self.face_source_colors = mesh
        self.selected_target = ("module", "primary")
        self._refresh_mesh_visual()

    def _refresh_mesh_visual(self):
        if self.mesh_base_vertices is None or self.mesh_faces is None or self.face_part_ids is None:
            return

        vertices = self._explode_parts(self.mesh_base_vertices, self.face_part_ids)
        vertices = self._apply_part_transforms(vertices, self.face_part_ids)
        colors = self._create_face_colors(
            vertices,
            self.mesh_faces,
            self.face_part_ids,
            self.face_source_colors,
        )

        if self.mesh_item is None:
            self.mesh_item = gl.GLMeshItem(
                vertexes=vertices,
                faces=self.mesh_faces,
                faceColors=colors,
                smooth=False,
                drawEdges=False,
                shader="shaded",
            )
            self.mesh_item.setGLOptions("opaque")
            self.view.addItem(self.mesh_item)
        elif hasattr(self.mesh_item, "setMeshData"):
            self.mesh_item.setMeshData(vertexes=vertices, faces=self.mesh_faces, faceColors=colors)
        else:
            self.view.removeItem(self.mesh_item)
            self.mesh_item = gl.GLMeshItem(
                vertexes=vertices,
                faces=self.mesh_faces,
                faceColors=colors,
                smooth=False,
                drawEdges=False,
                shader="shaded",
            )
            self.mesh_item.setGLOptions("opaque")
            self.view.addItem(self.mesh_item)
        self._apply_transform()
        self._update_preview_for_selection()

    def _apply_transform(self, _value=None, apply_secondary=True):
        if self.mesh_item is None:
            if apply_secondary:
                self._apply_secondary_transform()
            return

        scale = self.scale_slider.value() / 100.0
        move_x = float(self.move_x_slider.value()) + float(self.animation_translation[0])
        move_y = float(self.move_y_slider.value()) + float(self.animation_translation[1])
        move_z = float(self.move_z_slider.value()) + float(self.animation_translation[2])
        rot_x = float(self.rot_x_slider.value()) + float(self.animation_rotation[0])
        rot_y = float(self.rot_y_slider.value()) + float(self.animation_rotation[1])
        rot_z = float(self.rot_z_slider.value()) + float(self.animation_rotation[2])
        if self.orbit_enabled and self.orbit_module == "Primary":
            ox, oy, oz = self._orbit_offset_xyz()
            move_x += ox
            move_y += oy
            move_z += oz
            sx, sy, sz = self._orbit_spin_components()
            rot_x += sx
            rot_y += sy
            rot_z += sz

        self.mesh_item.resetTransform()
        self.mesh_item.scale(scale, scale, scale)
        self.mesh_item.rotate(rot_x, 1, 0, 0)
        self.mesh_item.rotate(rot_y, 0, 1, 0)
        self.mesh_item.rotate(rot_z, 0, 0, 1)
        self.mesh_item.translate(move_x, move_y, move_z)
        self.indicator_move_x.setText(str(int(self.move_x_slider.value())))
        self.indicator_move_y.setText(str(int(self.move_y_slider.value())))
        self.indicator_move_z.setText(str(int(self.move_z_slider.value())))
        self.indicator_rot_x.setText(str(int(self.rot_x_slider.value())))
        self.indicator_rot_y.setText(str(int(self.rot_y_slider.value())))
        self.indicator_rot_z.setText(str(int(self.rot_z_slider.value())))
        self.indicator_scale.setText(str(int(self.scale_slider.value())))
        if self.link_main_to_selected.isChecked() and not self._syncing_main_controls:
            self._apply_main_controls_to_selected_module()
        if apply_secondary:
            self._apply_secondary_transform()

    def _apply_secondary_transform(self, _value=None, apply_extras=True):
        if self.secondary_mesh_item is None:
            if apply_extras:
                self._apply_all_extra_modules_transforms()
            return
        scale = self.second_scale_slider.value() / 100.0
        move_x = float(self.second_move_x_slider.value())
        move_y = float(self.second_move_y_slider.value())
        move_z = float(self.second_move_z_slider.value())
        rot_x = float(self.secondary_rot[0])
        rot_y = float(self.secondary_rot[1])
        rot_z = float(self.secondary_rot[2])
        if self.orbit_enabled and self.orbit_module == "Secondary":
            ox, oy, oz = self._orbit_offset_xyz()
            move_x += ox
            move_y += oy
            move_z += oz
            rot_x, rot_y, rot_z = self._orbit_spin_components()
        self.secondary_mesh_item.resetTransform()
        self.secondary_mesh_item.scale(scale, scale, scale)
        self.secondary_mesh_item.rotate(rot_x, 1, 0, 0)
        self.secondary_mesh_item.rotate(rot_y, 0, 1, 0)
        self.secondary_mesh_item.rotate(rot_z, 0, 0, 1)
        self.secondary_mesh_item.translate(move_x, move_y, move_z)
        if apply_extras:
            self._apply_all_extra_modules_transforms()

    def _clear_secondary_module(self):
        if self.secondary_mesh_item is not None:
            self.view.removeItem(self.secondary_mesh_item)
            self.secondary_mesh_item = None
        self.secondary_base_vertices = None
        self.secondary_faces = None
        self.secondary_face_colors = None
        self.secondary_rot[:] = 0.0
        self._refresh_modules_list()
        self._update_preview_for_selection()

    def _apply_extra_module_transform(self, idx):
        if idx < 0 or idx >= len(self.extra_modules):
            return
        mod = self.extra_modules[idx]
        t = mod["transform"]
        item = mod["mesh_item"]
        item.resetTransform()
        item.scale(float(t[0]) / 100.0, float(t[0]) / 100.0, float(t[0]) / 100.0)
        item.rotate(float(t[4]), 1, 0, 0)
        item.rotate(float(t[5]), 0, 1, 0)
        item.rotate(float(t[6]), 0, 0, 1)
        item.translate(float(t[1]), float(t[2]), float(t[3]))

    def _apply_all_extra_modules_transforms(self):
        for i in range(len(self.extra_modules)):
            self._apply_extra_module_transform(i)

    def _refresh_modules_list(self):
        self.modules_list.blockSignals(True)
        current = self.selected_target
        self.modules_list.clear()
        item = QtWidgets.QListWidgetItem("Primary Module")
        item.setData(QtCore.Qt.UserRole, ("primary", -1))
        self.modules_list.addItem(item)
        if self.secondary_mesh_item is not None:
            item = QtWidgets.QListWidgetItem("Secondary Module")
            item.setData(QtCore.Qt.UserRole, ("secondary", -1))
            self.modules_list.addItem(item)
        for i in range(len(self.extra_modules)):
            item = QtWidgets.QListWidgetItem(f"Module {i + 3}")
            item.setData(QtCore.Qt.UserRole, ("extra", i))
            self.modules_list.addItem(item)
        for i in range(self.modules_list.count()):
            d = self.modules_list.item(i).data(QtCore.Qt.UserRole)
            if current[0] == "module":
                if current[1] == "primary" and d == ("primary", -1):
                    self.modules_list.setCurrentRow(i)
                elif current[1] == "secondary" and d == ("secondary", -1):
                    self.modules_list.setCurrentRow(i)
                elif isinstance(current[1], tuple) and current[1][0] == "extra" and d == ("extra", current[1][1]):
                    self.modules_list.setCurrentRow(i)
        self.modules_list.blockSignals(False)

    def _on_module_selection_changed(self, current, previous):
        if current is None:
            return
        kind, idx = current.data(QtCore.Qt.UserRole)
        if kind == "primary":
            self.selected_target = ("module", "primary")
        elif kind == "secondary":
            self.selected_target = ("module", "secondary")
        else:
            self.selected_target = ("module", ("extra", int(idx)))
        self._update_preview_for_selection()

    def _choose_color(self):
        color = QtWidgets.QColorDialog.getColor(self.base_color, self, "Choose Base Color")
        if color.isValid():
            self.base_color = color
            self._refresh_mesh_visual()

    def _set_selected_part_color(self):
        item = self.parts_list.currentItem()
        if item is None:
            QtWidgets.QMessageBox.information(self, "Part Color", "Select a part first.")
            return

        pid = int(item.data(QtCore.Qt.UserRole))
        initial = self.part_colors.get(pid, self._default_part_color(pid))
        color = QtWidgets.QColorDialog.getColor(initial, self, f"Part {pid + 1} Color")
        if color.isValid():
            self.part_colors[pid] = color
            self.color_mode.setCurrentText("Per-Part")
            self._refresh_mesh_visual()

    def _selected_part_id(self):
        item = self.parts_list.currentItem()
        if item is None:
            return None
        return int(item.data(QtCore.Qt.UserRole))

    def _on_part_selection_changed(self, current, previous):
        if current is None:
            return
        pid = int(current.data(QtCore.Qt.UserRole))
        self.selected_target = ("part", int(pid))
        self._update_preview_for_selection()
        transform = self.part_transforms.get(pid, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.part_move_x_slider.blockSignals(True)
        self.part_move_y_slider.blockSignals(True)
        self.part_move_z_slider.blockSignals(True)
        self.part_move_x_slider.setValue(int(transform[0]))
        self.part_move_y_slider.setValue(int(transform[1]))
        self.part_move_z_slider.setValue(int(transform[2]))
        self.part_move_x_slider.blockSignals(False)
        self.part_move_y_slider.blockSignals(False)
        self.part_move_z_slider.blockSignals(False)

    def _update_selected_part_transform(self):
        pid = self._selected_part_id()
        if pid is None:
            return
        self.part_transforms[pid] = np.array(
            [
                float(self.part_move_x_slider.value()),
                float(self.part_move_y_slider.value()),
                float(self.part_move_z_slider.value()),
            ],
            dtype=np.float32,
        )
        self._refresh_mesh_visual()

    def _reset_selected_part_transform(self):
        pid = self._selected_part_id()
        if pid is None:
            QtWidgets.QMessageBox.information(self, "Part Transform", "Select a part first.")
            return
        self.part_transforms[pid] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._on_part_selection_changed(self.parts_list.currentItem(), None)
        self._refresh_mesh_visual()

    def _reset_transform(self):
        self._stop_animation_and_reset_drift()
        self.move_x_slider.setValue(0)
        self.move_y_slider.setValue(0)
        self.move_z_slider.setValue(0)
        self.rot_x_slider.setValue(0)
        self.rot_y_slider.setValue(0)
        self.rot_z_slider.setValue(0)
        self.scale_slider.setValue(100)

    def _xor_bytes(self, data):
        key = b"VoxelForgePrivateModelKey"
        out = bytearray(len(data))
        key_len = len(key)
        for i, b in enumerate(data):
            out[i] = b ^ key[i % key_len]
        return bytes(out)

    def _build_model_payload(self):
        part_count = int(max(self.part_colors.keys()) + 1) if self.part_colors else 0
        part_color_rgba = np.zeros((part_count, 4), dtype=np.uint8)
        for pid, color in self.part_colors.items():
            part_color_rgba[pid] = [color.red(), color.green(), color.blue(), color.alpha()]

        return {
            "signature": np.array([b"VF3D_PRIVATE_V1"]),
            "vertices": self.mesh_base_vertices,
            "faces": self.mesh_faces,
            "face_part_ids": self.face_part_ids,
            "face_source_colors": self.face_source_colors
            if self.face_source_colors is not None
            else np.zeros((0, 3), dtype=np.float32),
            "part_color_rgba": part_color_rgba,
            "part_transforms": self._serialize_part_transforms(),
            "part_labels": self.part_labels if self.part_labels is not None else np.zeros((0, 0), dtype=np.int32),
            "fg_mask": self.fg_mask.astype(np.uint8) if self.fg_mask is not None else np.zeros((0, 0), dtype=np.uint8),
            "scale": np.array([self.scale_slider.value()], dtype=np.float32),
            "move": np.array(
                [self.move_x_slider.value(), self.move_y_slider.value(), self.move_z_slider.value()],
                dtype=np.float32,
            ),
            "rotation": np.array(
                [self.rot_x_slider.value(), self.rot_y_slider.value(), self.rot_z_slider.value()],
                dtype=np.float32,
            ),
            "animation_translation": self.animation_translation.astype(np.float32),
            "animation_rotation": self.animation_rotation.astype(np.float32),
            "anim_move_speed": self._get_animation_move_speed(),
            "anim_rot_speed": self._get_animation_rot_speed(),
            "anim_enabled": np.array([1 if self.keep_moving_check.isChecked() else 0], dtype=np.uint8),
        }

    def _write_private_model_file(self, file_path):
        encoded = self._private_model_bytes()
        with open(file_path, "wb") as f:
            f.write(encoded)

    def _private_model_bytes(self):
        payload = self._build_model_payload()
        mem = io.BytesIO()
        np.savez_compressed(mem, **payload)
        blob = mem.getvalue()
        return b"VFMODEL1" + self._xor_bytes(blob)

    def _decode_private_model(self, file_path):
        with open(file_path, "rb") as f:
            raw = f.read()
        return self._decode_private_model_bytes(raw)

    def _decode_private_model_bytes(self, raw):
        if not raw.startswith(b"VFMODEL1"):
            return None

        decoded = self._xor_bytes(raw[len(b"VFMODEL1"):])
        mem = io.BytesIO(decoded)
        data = np.load(mem, allow_pickle=False)
        signature = data["signature"][0]
        if signature != b"VF3D_PRIVATE_V1":
            return None

        faces = data["faces"].astype(np.uint32)
        if "face_source_colors" in data:
            source_colors = data["face_source_colors"].astype(np.float32)
        else:
            source_colors = np.ones((faces.shape[0], 3), dtype=np.float32)

        return {
            "vertices": data["vertices"].astype(np.float32),
            "faces": faces,
            "face_source_colors": source_colors,
            "face_part_ids": data["face_part_ids"].astype(np.int32),
            "part_color_rgba": data["part_color_rgba"].astype(np.uint8),
            "part_transforms": data["part_transforms"].astype(np.float32) if "part_transforms" in data else np.zeros((0, 4), dtype=np.float32),
            "part_labels": data["part_labels"].astype(np.int32),
            "fg_mask": data["fg_mask"].astype(bool),
            "scale": data["scale"].astype(np.float32) if "scale" in data else None,
            "move": data["move"].astype(np.float32) if "move" in data else None,
            "rotation": data["rotation"].astype(np.float32) if "rotation" in data else None,
            "animation_translation": data["animation_translation"].astype(np.float32) if "animation_translation" in data else None,
            "animation_rotation": data["animation_rotation"].astype(np.float32) if "animation_rotation" in data else None,
            "anim_move_speed": data["anim_move_speed"].astype(np.float32) if "anim_move_speed" in data else None,
            "anim_rot_speed": data["anim_rot_speed"].astype(np.float32) if "anim_rot_speed" in data else None,
            "anim_enabled": data["anim_enabled"].astype(np.uint8) if "anim_enabled" in data else None,
        }

    def _apply_decoded_private_model(self, decoded):
        self.mesh_base_vertices = decoded["vertices"]
        self.mesh_faces = decoded["faces"]
        self.face_part_ids = decoded["face_part_ids"]
        self.face_source_colors = decoded["face_source_colors"]

        part_color_rgba = decoded["part_color_rgba"]
        self.part_colors = {}
        for pid in range(part_color_rgba.shape[0]):
            r, g, b, a = [int(v) for v in part_color_rgba[pid]]
            self.part_colors[pid] = QtGui.QColor(r, g, b, a)
        self.part_transforms = self._deserialize_part_transforms(decoded["part_transforms"])

        self.part_labels = decoded["part_labels"]
        self.fg_mask = decoded["fg_mask"]

        if self.face_part_ids.size:
            unique, counts = np.unique(self.face_part_ids, return_counts=True)
            self.part_sizes = {int(u): int(c) for u, c in zip(unique, counts)}
        else:
            self.part_sizes = {}

        if decoded["scale"] is not None:
            self.scale_slider.setValue(int(float(decoded["scale"][0])))
        if decoded["move"] is not None:
            move = decoded["move"]
            self.move_x_slider.setValue(int(float(move[0])))
            self.move_y_slider.setValue(int(float(move[1])))
            self.move_z_slider.setValue(int(float(move[2])))
        if decoded["rotation"] is not None:
            rot = decoded["rotation"]
            self.rot_x_slider.setValue(int(float(rot[0])))
            self.rot_y_slider.setValue(int(float(rot[1])))
            self.rot_z_slider.setValue(int(float(rot[2])))
        if decoded["animation_translation"] is not None:
            self.animation_translation = decoded["animation_translation"]
        if decoded["animation_rotation"] is not None:
            self.animation_rotation = decoded["animation_rotation"]
        if decoded["anim_move_speed"] is not None:
            s = decoded["anim_move_speed"]
            self.anim_move_x_speed.setValue(float(s[0]))
            self.anim_move_y_speed.setValue(float(s[1]))
            self.anim_move_z_speed.setValue(float(s[2]))
        if decoded["anim_rot_speed"] is not None:
            s = decoded["anim_rot_speed"]
            self.anim_rot_x_speed.setValue(float(s[0]))
            self.anim_rot_y_speed.setValue(float(s[1]))
            self.anim_rot_z_speed.setValue(float(s[2]))
        if decoded["anim_enabled"] is not None:
            self.keep_moving_check.setChecked(bool(int(decoded["anim_enabled"][0])))

        self._refresh_parts_list()
        if self.parts_list.count() > 0:
            self.parts_list.setCurrentRow(0)
        self._refresh_mesh_visual()

    def _read_private_model_bytes(self, raw, show_feedback=True):
        try:
            decoded = self._decode_private_model_bytes(raw)
            if decoded is None:
                if show_feedback:
                    QtWidgets.QMessageBox.warning(self, "Load Model", "Invalid private model file.")
                return False
            self._apply_decoded_private_model(decoded)
            if show_feedback:
                QtWidgets.QMessageBox.information(self, "Load Model", "Loaded model data.")
            return True
        except Exception as exc:
            logger.exception("Failed to load private model")
            if show_feedback:
                QtWidgets.QMessageBox.warning(self, "Load Model", f"Failed to load model: {exc}")
            return False

    def _read_private_model_file(self, file_path, show_feedback=True):
        try:
            decoded = self._decode_private_model(file_path)
            if decoded is None:
                if show_feedback:
                    QtWidgets.QMessageBox.warning(self, "Load Model", "Invalid private model file.")
                return False
            self._apply_decoded_private_model(decoded)
            if show_feedback:
                QtWidgets.QMessageBox.information(self, "Load Model", f"Loaded: {file_path}")
            return True
        except Exception as exc:
            logger.exception("Failed to load private model file")
            if show_feedback:
                QtWidgets.QMessageBox.warning(self, "Load Model", f"Failed to load model: {exc}")
            return False

    def save_private_model(self):
        if self.mesh_base_vertices is None or self.mesh_faces is None:
            QtWidgets.QMessageBox.warning(self, "Save Model", "No model to save.")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Private Model", "", "VoxelForge Model (*.vf3d)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".vf3d"):
            file_path += ".vf3d"

        encoded = self._private_model_bytes()
        with open(file_path, "wb") as f:
            f.write(encoded)
        self.store.save_blob(
            f"exported-model:{hashlib.sha256(str(file_path).encode('utf-8')).hexdigest()[:16]}",
            encoded,
            "model",
            user_id=self.current_user["id"] if self.current_user else None,
            content_type="application/x-voxelforge-model",
        )
        QtWidgets.QMessageBox.information(self, "Save Model", f"Saved: {file_path}")

    def load_private_model(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Private Model", "", "VoxelForge Model (*.vf3d)"
        )
        if not file_path:
            return
        self._store_local_asset(file_path, "model")
        self._read_private_model_file(file_path, show_feedback=True)

    def load_second_private_model(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Second Model", "", "VoxelForge Model (*.vf3d)"
        )
        if not file_path:
            return
        self._store_local_asset(file_path, "model")
        model = self._decode_private_model(file_path)
        if model is None:
            QtWidgets.QMessageBox.warning(self, "Load Second Model", "Failed to load second model.")
            return
        colors = model["face_source_colors"]
        if self.secondary_mesh_item is None:
            self._set_secondary_mesh(model["vertices"], model["faces"], colors)
        else:
            self._add_extra_module(model["vertices"], model["faces"], colors)
        self._refresh_modules_list()

    def _serialize_part_transforms(self):
        if not self.part_transforms:
            return np.zeros((0, 4), dtype=np.float32)
        rows = []
        for pid in sorted(self.part_transforms.keys()):
            t = self.part_transforms[pid]
            rows.append([float(pid), float(t[0]), float(t[1]), float(t[2])])
        return np.asarray(rows, dtype=np.float32)

    def _deserialize_part_transforms(self, arr):
        out = {}
        if arr is None or arr.size == 0:
            return out
        for row in arr:
            pid = int(row[0])
            out[pid] = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=np.float32)
        return out

    def _duplicate_selected_part(self):
        item = self.parts_list.currentItem()
        if item is None or self.mesh_faces is None:
            return
        pid = int(item.data(QtCore.Qt.UserRole))
        mask = self.face_part_ids == pid
        if not np.any(mask):
            return
        part_faces = self.mesh_faces[mask]
        part_colors = self.face_source_colors[mask] if self.face_source_colors is not None else np.ones((part_faces.shape[0], 3), dtype=np.float32)
        old_to_new = {}
        new_verts = []
        new_faces = []
        for face in part_faces:
            nf = []
            for idx in face:
                idx = int(idx)
                if idx not in old_to_new:
                    old_to_new[idx] = len(new_verts)
                    new_verts.append(self.mesh_base_vertices[idx])
                nf.append(old_to_new[idx])
            new_faces.append(nf)
        if len(new_faces) == 0:
            return
        new_verts = np.asarray(new_verts, dtype=np.float32)
        new_verts[:, 0] += 12.0
        face_offset = self.mesh_base_vertices.shape[0]
        new_faces = np.asarray(new_faces, dtype=np.uint32) + face_offset
        new_pid = int(self.face_part_ids.max()) + 1 if self.face_part_ids.size else 0
        self.mesh_base_vertices = np.vstack([self.mesh_base_vertices, new_verts]).astype(np.float32)
        self.mesh_faces = np.vstack([self.mesh_faces, new_faces]).astype(np.uint32)
        self.face_part_ids = np.concatenate([self.face_part_ids, np.full((new_faces.shape[0],), new_pid, dtype=np.int32)])
        self.face_source_colors = np.vstack([self.face_source_colors, part_colors]).astype(np.float32)
        base_color = self.part_colors.get(pid, self.base_color)
        self.part_colors[new_pid] = QtGui.QColor(base_color)
        self.part_transforms[new_pid] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._recompute_part_sizes_from_faces()
        self._refresh_parts_list()
        self._refresh_mesh_visual()

    def _delete_selected_part(self):
        item = self.parts_list.currentItem()
        if item is None or self.mesh_faces is None:
            return
        pid = int(item.data(QtCore.Qt.UserRole))
        keep = self.face_part_ids != pid
        if not np.any(keep):
            QtWidgets.QMessageBox.warning(self, "Delete Part", "Cannot delete all parts.")
            return
        self.mesh_faces = self.mesh_faces[keep]
        self.face_part_ids = self.face_part_ids[keep]
        self.face_source_colors = self.face_source_colors[keep]
        self.part_colors.pop(pid, None)
        self.part_transforms.pop(pid, None)
        self._recompute_part_sizes_from_faces()
        self._refresh_parts_list()
        self._refresh_mesh_visual()

    def _recompute_part_sizes_from_faces(self):
        if self.face_part_ids is None or self.face_part_ids.size == 0:
            self.part_sizes = {}
            return
        unique, counts = np.unique(self.face_part_ids, return_counts=True)
        self.part_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

    def _on_parts_context_menu(self, pos):
        item = self.parts_list.itemAt(pos)
        if item is None:
            return
        self.parts_list.setCurrentItem(item)
        menu = QtWidgets.QMenu(self)
        a_dup = menu.addAction("Duplicate Part")
        a_del = menu.addAction("Delete Part")
        act = menu.exec_(self.parts_list.mapToGlobal(pos))
        if act == a_dup:
            self._duplicate_selected_part()
        elif act == a_del:
            self._delete_selected_part()

    def _on_modules_context_menu(self, pos):
        item = self.modules_list.itemAt(pos)
        if item is None:
            return
        self.modules_list.setCurrentItem(item)
        menu = QtWidgets.QMenu(self)
        a_dup = menu.addAction("Duplicate Module")
        a_del = menu.addAction("Delete Module")
        act = menu.exec_(self.modules_list.mapToGlobal(pos))
        if act == a_dup:
            self._duplicate_selected_module()
        elif act == a_del:
            self._delete_selected_module()

    def _duplicate_selected_module(self):
        target = self.selected_target
        if target != ("module", "primary") and target != ("module", "secondary") and not (
            isinstance(target[1], tuple) and target[1][0] == "extra"
        ):
            return
        if target == ("module", "primary"):
            verts = self.mesh_base_vertices
            faces = self.mesh_faces
            colors = self.face_source_colors
        elif target == ("module", "secondary"):
            verts = self.secondary_base_vertices
            faces = self.secondary_faces
            colors = self.secondary_face_colors
        else:
            idx = int(target[1][1])
            mod = self.extra_modules[idx]
            verts = mod["base_vertices"]
            faces = mod["faces"]
            colors = mod["face_colors"]
        if verts is None or faces is None:
            return
        self._add_extra_module(verts.copy(), faces.copy(), colors.copy() if colors is not None else None)

    def _delete_selected_module(self):
        target = self.selected_target
        if target == ("module", "primary"):
            QtWidgets.QMessageBox.warning(self, "Delete Module", "Primary module cannot be deleted.")
            return
        if target == ("module", "secondary"):
            self._clear_secondary_module()
            return
        if isinstance(target[1], tuple) and target[1][0] == "extra":
            idx = int(target[1][1])
            if 0 <= idx < len(self.extra_modules):
                self.view.removeItem(self.extra_modules[idx]["mesh_item"])
                del self.extra_modules[idx]
                self.selected_target = ("module", "primary")
                self._refresh_modules_list()
                self._update_preview_for_selection()

    def _apply_main_controls_to_selected_module(self):
        target = self.selected_target
        if target == ("module", "primary"):
            return
        scale = float(self.scale_slider.value())
        mx = float(self.move_x_slider.value())
        my = float(self.move_y_slider.value())
        mz = float(self.move_z_slider.value())
        rx = float(self.rot_x_slider.value())
        ry = float(self.rot_y_slider.value())
        rz = float(self.rot_z_slider.value())
        if target == ("module", "secondary"):
            self._syncing_main_controls = True
            try:
                self.second_scale_slider.setValue(int(scale))
                self.second_move_x_slider.setValue(int(mx))
                self.second_move_y_slider.setValue(int(my))
                self.second_move_z_slider.setValue(int(mz))
            finally:
                self._syncing_main_controls = False
            self.secondary_rot[:] = [rx, ry, rz]
            self._apply_secondary_transform()
            return
        if isinstance(target[1], tuple) and target[1][0] == "extra":
            idx = int(target[1][1])
            if 0 <= idx < len(self.extra_modules):
                self.extra_modules[idx]["transform"][:] = [scale, mx, my, mz, rx, ry, rz]
                self._apply_extra_module_transform(idx)
                self._update_preview_for_selection()

    def _get_animation_move_speed(self):
        return np.array(
            [
                float(self.anim_move_x_speed.value()),
                float(self.anim_move_y_speed.value()),
                float(self.anim_move_z_speed.value()),
            ],
            dtype=np.float32,
        )

    def _get_animation_rot_speed(self):
        return np.array(
            [
                float(self.anim_rot_x_speed.value()),
                float(self.anim_rot_y_speed.value()),
                float(self.anim_rot_z_speed.value()),
            ],
            dtype=np.float32,
        )

    def _toggle_animation(self):
        if self.keep_moving_check.isChecked():
            self._last_anim_time = time.perf_counter()
            self._fps_last_time = self._last_anim_time
            self._fps_frames = 0
            self.animation_timer.start()
        else:
            self.animation_timer.stop()
            self._fps_value = 0.0
            self._update_fps_label()
        self._apply_transform()

    def _on_animation_params_changed(self):
        if self.keep_moving_check.isChecked():
            self._last_anim_time = time.perf_counter()
        self._apply_transform()

    def _tick_animation(self):
        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_anim_time))
        self._last_anim_time = now
        if dt <= 0.0:
            return

        self.animation_translation += self._get_animation_move_speed() * dt
        self.animation_rotation += self._get_animation_rot_speed() * dt
        self.animation_rotation = np.mod(self.animation_rotation, 360.0)
        self.orbit_angle = float((self.orbit_angle + self.orbit_speed * dt) % 360.0)
        self.orbit_spin_angle = float((self.orbit_spin_angle + self.orbit_spin_speed * dt) % 360.0)
        self._fps_frames += 1
        fps_dt = now - self._fps_last_time
        if fps_dt >= 0.25:
            self._fps_value = self._fps_frames / fps_dt
            self._fps_frames = 0
            self._fps_last_time = now
            self._update_fps_label()
        if self.orbit_enabled and self.orbit_module == "Secondary":
            self._apply_secondary_transform(apply_extras=False)
        else:
            self._apply_transform(apply_secondary=False)

    def _update_fps_label(self):
        self.fps_label.setText(f"FPS: {self._fps_value:.1f} / {self.target_fps}")

    def _orbit_offset_xyz(self):
        a = np.deg2rad(self.orbit_angle)
        r = float(self.orbit_radius)
        if self.orbit_plane == "XY":
            return r * np.cos(a), r * np.sin(a), 0.0
        if self.orbit_plane == "YZ":
            return 0.0, r * np.cos(a), r * np.sin(a)
        return r * np.cos(a), 0.0, r * np.sin(a)

    def _orbit_spin_components(self):
        ang = float(self.orbit_spin_angle)
        axes = self.orbit_spin_axes
        return (
            ang if "X" in axes else 0.0,
            ang if "Y" in axes else 0.0,
            ang if "Z" in axes else 0.0,
        )

    def _tick_preview(self):
        if self.preview_mesh_item is None:
            return
        self.preview_angle = float((self.preview_angle + 1.2) % 360.0)
        self.preview_mesh_item.resetTransform()
        self.preview_mesh_item.rotate(self.preview_angle, 0, 1, 0)

    def _text_input_has_focus(self):
        fw = QtWidgets.QApplication.focusWidget()
        return isinstance(
            fw,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QSpinBox,
                QtWidgets.QDoubleSpinBox,
                QtWidgets.QComboBox,
            ),
        )

    def _handle_nav_key(self, event, pressed):
        if self._text_input_has_focus():
            return False
        key_map = {
            QtCore.Qt.Key_W: "w",
            QtCore.Qt.Key_A: "a",
            QtCore.Qt.Key_S: "s",
            QtCore.Qt.Key_D: "d",
            QtCore.Qt.Key_Space: "space",
            QtCore.Qt.Key_Shift: "shift",
            QtCore.Qt.Key_Q: "turn_left",
            QtCore.Qt.Key_E: "turn_right",
            QtCore.Qt.Key_Left: "turn_left",
            QtCore.Qt.Key_Right: "turn_right",
        }
        key = key_map.get(event.key())
        if key is None:
            return False
        if pressed:
            self.pressed_nav_keys.add(key)
        else:
            self.pressed_nav_keys.discard(key)
        event.accept()
        return True

    def _tick_navigation(self):
        now = time.perf_counter()
        dt = max(0.0, min(0.05, now - self._last_nav_time))
        self._last_nav_time = now
        if dt <= 0.0 or not self.pressed_nav_keys:
            return
        forward, right = self._camera_planar_vectors()
        direction = np.zeros(3, dtype=np.float32)
        if "w" in self.pressed_nav_keys:
            direction += forward
        if "s" in self.pressed_nav_keys:
            direction -= forward
        if "d" in self.pressed_nav_keys:
            direction += right
        if "a" in self.pressed_nav_keys:
            direction -= right
        if "space" in self.pressed_nav_keys:
            direction[2] += 1.0
        if "shift" in self.pressed_nav_keys:
            direction[2] -= 1.0
        turn_dir = 0.0
        if "turn_left" in self.pressed_nav_keys:
            turn_dir += 1.0
        if "turn_right" in self.pressed_nav_keys:
            turn_dir -= 1.0
        if abs(turn_dir) > 1e-6:
            self.view.opts["azimuth"] = float(self.view.opts.get("azimuth", 0.0)) + turn_dir * self.turn_speed * dt
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            self.view.update()
            return
        direction /= norm
        step = direction * float(self.walk_speed) * dt
        center = self.view.opts.get("center", QtGui.QVector3D(0.0, 0.0, 0.0))
        self.view.opts["center"] = QtGui.QVector3D(
            float(center.x() + step[0]),
            float(center.y() + step[1]),
            float(center.z() + step[2]),
        )
        self.view.update()

    def _camera_planar_vectors(self):
        center = self.view.opts.get("center", QtGui.QVector3D(0.0, 0.0, 0.0))
        fx = 0.0
        fy = 0.0
        try:
            cam = self.view.cameraPosition()
            fx = float(center.x() - cam.x())
            fy = float(center.y() - cam.y())
        except Exception:
            pass
        norm = math.hypot(fx, fy)
        if norm < 1e-6:
            az = math.radians(float(self.view.opts.get("azimuth", 0.0)))
            fx = -math.cos(az)
            fy = -math.sin(az)
            norm = math.hypot(fx, fy)
        fx /= norm
        fy /= norm
        forward = np.array([fx, fy, 0.0], dtype=np.float32)
        right = np.array([fy, -fx, 0.0], dtype=np.float32)
        return forward, right

    def _stop_animation_and_reset_drift(self):
        self.keep_moving_check.setChecked(False)
        self.animation_translation[:] = 0.0
        self.animation_rotation[:] = 0.0
        self._apply_transform()

    def _update_account_label(self):
        if not hasattr(self, "account_label"):
            return
        if self.current_user:
            name = self.current_user.get("display_name") or self.current_user.get("email")
            provider = self.current_user.get("auth_provider", "email")
            self.account_label.setText(f"{name} ({provider})")
        else:
            self.account_label.setText("Not signed in")

    def _set_current_user(self, user):
        self.current_user = user
        self.store.save_state("current_user_id", user["id"] if user else None)
        self._update_account_label()
        self._save_app_settings()

    def _ensure_user_session(self):
        self._update_account_label()
        if self.current_user is None:
            self._open_account_dialog(force=False)

    def _open_account_dialog(self, force=False):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("VoxelForge Account")
        dialog.resize(460, 320)
        layout = QtWidgets.QVBoxLayout(dialog)

        current = QtWidgets.QLabel()
        current.setWordWrap(True)
        if self.current_user:
            current.setText(f"Signed in as {self.current_user.get('email')}")
        else:
            current.setText("Sign in or create an account. Account data is stored in voxelforge.db.")
        layout.addWidget(current)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        sign_in = QtWidgets.QWidget()
        sign_in_form = QtWidgets.QFormLayout(sign_in)
        login_email = QtWidgets.QLineEdit()
        login_password = QtWidgets.QLineEdit()
        login_password.setEchoMode(QtWidgets.QLineEdit.Password)
        sign_in_form.addRow("Email", login_email)
        sign_in_form.addRow("Password", login_password)
        tabs.addTab(sign_in, "Sign In")

        sign_up = QtWidgets.QWidget()
        sign_up_form = QtWidgets.QFormLayout(sign_up)
        signup_name = QtWidgets.QLineEdit()
        signup_email = QtWidgets.QLineEdit()
        signup_password = QtWidgets.QLineEdit()
        signup_password.setEchoMode(QtWidgets.QLineEdit.Password)
        sign_up_form.addRow("Name", signup_name)
        sign_up_form.addRow("Email", signup_email)
        sign_up_form.addRow("Password", signup_password)
        tabs.addTab(sign_up, "Sign Up")

        google = QtWidgets.QWidget()
        google_form = QtWidgets.QFormLayout(google)
        google_name = QtWidgets.QLineEdit()
        google_email = QtWidgets.QLineEdit()
        google_sub = QtWidgets.QLineEdit()
        google_sub.setPlaceholderText("Google subject/account id; optional until OAuth is configured")
        google_note = QtWidgets.QLabel(
            "Google OAuth needs a Google client ID. This stores the Google account identity in SQLite so OAuth can be connected without a schema change."
        )
        google_note.setWordWrap(True)
        google_form.addRow(google_note)
        google_form.addRow("Name", google_name)
        google_form.addRow("Google Email", google_email)
        google_form.addRow("Google ID", google_sub)
        tabs.addTab(google, "Google")

        status = QtWidgets.QLabel("")
        status.setWordWrap(True)
        status.setStyleSheet("QLabel { color: #EF5350; }")
        layout.addWidget(status)

        buttons = QtWidgets.QDialogButtonBox()
        login_btn = buttons.addButton("Sign In", QtWidgets.QDialogButtonBox.AcceptRole)
        signup_btn = buttons.addButton("Create Account", QtWidgets.QDialogButtonBox.ActionRole)
        google_btn = buttons.addButton("Use Google Account", QtWidgets.QDialogButtonBox.ActionRole)
        signout_btn = buttons.addButton("Sign Out", QtWidgets.QDialogButtonBox.DestructiveRole)
        close_btn = buttons.addButton(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(buttons)

        def _show_error(exc):
            status.setText(str(exc))

        def _sign_in():
            try:
                user = self.store.authenticate_email_user(login_email.text(), login_password.text())
                self._set_current_user(user)
                dialog.accept()
            except Exception as exc:
                _show_error(exc)

        def _sign_up():
            try:
                user = self.store.create_email_user(signup_email.text(), signup_password.text(), signup_name.text())
                self._set_current_user(user)
                dialog.accept()
            except sqlite3.IntegrityError:
                _show_error("An account with this email already exists.")
            except Exception as exc:
                _show_error(exc)

        def _google_sign_in():
            try:
                user = self.store.create_or_update_google_user(
                    google_email.text(),
                    google_sub.text(),
                    google_name.text(),
                )
                self._set_current_user(user)
                dialog.accept()
            except sqlite3.IntegrityError:
                _show_error("This Google identity is already linked to another account.")
            except Exception as exc:
                _show_error(exc)

        def _sign_out():
            self._set_current_user(None)
            current.setText("Signed out.")
            if not force:
                dialog.accept()

        login_btn.clicked.connect(_sign_in)
        signup_btn.clicked.connect(_sign_up)
        google_btn.clicked.connect(_google_sign_in)
        signout_btn.clicked.connect(_sign_out)
        close_btn.clicked.connect(dialog.reject)

        dialog.exec_()

    def _open_settings_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Program Settings")
        layout = QtWidgets.QFormLayout(dialog)

        version_label = QtWidgets.QLabel("v1.0")
        version_label.setStyleSheet("font-weight: 700;")
        layout.addRow("Version", version_label)

        controls_label = QtWidgets.QLabel(
            "W: Forward | S: Backward | A: Left | D: Right | Space: Fly Up | Shift: Land Down | Q/E or Arrow Left/Right: Turn"
        )
        controls_label.setWordWrap(True)
        layout.addRow("Movement Controls", controls_label)

        image_res = QtWidgets.QSpinBox()
        image_res.setRange(64, 512)
        image_res.setValue(int(self.image_resolution))
        layout.addRow("Image Resolution", image_res)

        ground_size = QtWidgets.QSpinBox()
        ground_size.setRange(50, 5000)
        ground_size.setValue(int(self.ground_size))
        layout.addRow("Ground Size", ground_size)

        dialog_ground_color = QtGui.QColor(self.ground_color)
        dialog_sky_color = QtGui.QColor(self.sky_color)
        ground_color_btn = QtWidgets.QPushButton(f"Ground Color: {self._color_to_hex(dialog_ground_color)}")
        sky_color_btn = QtWidgets.QPushButton(f"Sky Color: {self._color_to_hex(dialog_sky_color)}")
        layout.addRow(ground_color_btn)
        layout.addRow(sky_color_btn)

        def _pick_ground():
            nonlocal dialog_ground_color
            color = QtWidgets.QColorDialog.getColor(dialog_ground_color, dialog, "Choose Ground Color")
            if color.isValid():
                dialog_ground_color = color
                ground_color_btn.setText(f"Ground Color: {self._color_to_hex(color)}")

        def _pick_sky():
            nonlocal dialog_sky_color
            color = QtWidgets.QColorDialog.getColor(dialog_sky_color, dialog, "Choose Sky Color")
            if color.isValid():
                dialog_sky_color = color
                sky_color_btn.setText(f"Sky Color: {self._color_to_hex(color)}")

        ground_color_btn.clicked.connect(_pick_ground)
        sky_color_btn.clicked.connect(_pick_sky)

        auto_anim = QtWidgets.QCheckBox("Auto Start Animation")
        auto_anim.setChecked(bool(self.auto_start_animation))
        layout.addRow(auto_anim)

        theme_combo = QtWidgets.QComboBox()
        theme_combo.addItems(["Dark", "Light"])
        theme_combo.setCurrentText(self.theme_mode if self.theme_mode in ("Dark", "Light") else "Dark")
        layout.addRow("Theme", theme_combo)

        resume_session_check = QtWidgets.QCheckBox("Resume Last Session on Startup")
        resume_session_check.setChecked(bool(self.resume_last_session))
        layout.addRow(resume_session_check)

        orbit_enable_check = QtWidgets.QCheckBox("Enable Orbit System")
        orbit_enable_check.setChecked(bool(self.orbit_enabled))
        layout.addRow(orbit_enable_check)

        orbit_module_combo = QtWidgets.QComboBox()
        orbit_module_combo.addItems(["Primary", "Secondary"])
        orbit_module_combo.setCurrentText(self.orbit_module if self.orbit_module in ("Primary", "Secondary") else "Primary")
        layout.addRow("Orbit Module", orbit_module_combo)

        orbit_plane_combo = QtWidgets.QComboBox()
        orbit_plane_combo.addItems(["XZ", "XY", "YZ"])
        orbit_plane_combo.setCurrentText(self.orbit_plane if self.orbit_plane in ("XZ", "XY", "YZ") else "XZ")
        layout.addRow("Orbit Plane", orbit_plane_combo)

        orbit_radius_spin = QtWidgets.QDoubleSpinBox()
        orbit_radius_spin.setRange(0.0, 5000.0)
        orbit_radius_spin.setDecimals(1)
        orbit_radius_spin.setSingleStep(5.0)
        orbit_radius_spin.setValue(float(self.orbit_radius))
        layout.addRow("Orbit Radius", orbit_radius_spin)

        orbit_speed_spin = QtWidgets.QDoubleSpinBox()
        orbit_speed_spin.setRange(-720.0, 720.0)
        orbit_speed_spin.setDecimals(1)
        orbit_speed_spin.setSingleStep(1.0)
        orbit_speed_spin.setValue(float(self.orbit_speed))
        layout.addRow("Orbit Speed (deg/s)", orbit_speed_spin)

        orbit_axes_combo = QtWidgets.QComboBox()
        orbit_axes_combo.addItems(["X", "Y", "Z", "XY", "YZ", "XZ", "XYZ"])
        orbit_axes_combo.setCurrentText(self.orbit_spin_axes if self.orbit_spin_axes in ("X", "Y", "Z", "XY", "YZ", "XZ", "XYZ") else "Y")
        layout.addRow("Self Rotation Axes", orbit_axes_combo)

        orbit_spin_speed_spin = QtWidgets.QDoubleSpinBox()
        orbit_spin_speed_spin.setRange(-720.0, 720.0)
        orbit_spin_speed_spin.setDecimals(1)
        orbit_spin_speed_spin.setSingleStep(1.0)
        orbit_spin_speed_spin.setValue(float(self.orbit_spin_speed))
        layout.addRow("Self Rotation Speed", orbit_spin_speed_spin)

        db_label = QtWidgets.QLineEdit(self.database_path)
        db_label.setReadOnly(True)
        layout.addRow("Database", db_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.image_resolution = int(image_res.value())
        self._set_ground_size(int(ground_size.value()))
        self._set_ground_color(dialog_ground_color)
        self._set_sky_color(dialog_sky_color)
        self.auto_start_animation = bool(auto_anim.isChecked())
        self.theme_mode = str(theme_combo.currentText())
        self.resume_last_session = bool(resume_session_check.isChecked())
        self.orbit_enabled = bool(orbit_enable_check.isChecked())
        self.orbit_module = str(orbit_module_combo.currentText())
        self.orbit_plane = str(orbit_plane_combo.currentText())
        self.orbit_radius = float(orbit_radius_spin.value())
        self.orbit_speed = float(orbit_speed_spin.value())
        self.orbit_spin_axes = str(orbit_axes_combo.currentText())
        self.orbit_spin_speed = float(orbit_spin_speed_spin.value())
        self._apply_theme()
        if self.orbit_enabled and not self.keep_moving_check.isChecked():
            self.keep_moving_check.setChecked(True)
        self._apply_transform()
        self._save_app_settings()

    def _save_app_settings(self):
        self.store.save_state("settings", self._collect_settings_dict())
        self.store.save_state("current_user_id", self.current_user["id"] if self.current_user else None)
        if self.code_script_text:
            self.store.save_state("current_code_script", self.code_script_text)

    def _load_app_settings(self):
        user_id = self.store.load_state("current_user_id")
        self.current_user = self.store.get_user(user_id)
        payload = self.store.load_state("settings", {})
        if payload:
            self._apply_settings_dict(payload)
        else:
            self.code_script_text = self.store.load_state("current_code_script", self.code_script_text) or self.code_script_text
        self._update_account_label()

    def _collect_settings_dict(self):
        return {
            "image_resolution": int(self.image_resolution),
            "ground_size": int(self.ground_size),
            "ground_color": self._color_to_hex(self.ground_color),
            "sky_color": self._color_to_hex(self.sky_color),
            "code_script": str(self.code_script_text),
            "current_vf_file_path": str(self.current_vf_file_path),
            "auto_start_animation": bool(self.auto_start_animation),
            "theme_mode": str(self.theme_mode),
            "resume_last_session": bool(self.resume_last_session),
            "orbit_enabled": bool(self.orbit_enabled),
            "orbit_module": str(self.orbit_module),
            "orbit_plane": str(self.orbit_plane),
            "orbit_radius": float(self.orbit_radius),
            "orbit_speed": float(self.orbit_speed),
            "orbit_spin_axes": str(self.orbit_spin_axes),
            "orbit_spin_speed": float(self.orbit_spin_speed),
            "remove_background": bool(self.remove_bg_check.isChecked()),
            "bg_threshold": int(self.bg_threshold_slider.value()),
            "extrude_height": int(self.height_slider.value()),
            "parts_separation": int(self.part_gap_slider.value()),
            "color_mode": str(self.color_mode.currentText()),
            "global_scale": int(self.scale_slider.value()),
            "global_move_x": int(self.move_x_slider.value()),
            "global_move_y": int(self.move_y_slider.value()),
            "global_move_z": int(self.move_z_slider.value()),
            "global_rot_x": int(self.rot_x_slider.value()),
            "global_rot_y": int(self.rot_y_slider.value()),
            "global_rot_z": int(self.rot_z_slider.value()),
            "animation_enabled": bool(self.keep_moving_check.isChecked()),
            "anim_move_x": float(self.anim_move_x_speed.value()),
            "anim_move_y": float(self.anim_move_y_speed.value()),
            "anim_move_z": float(self.anim_move_z_speed.value()),
            "anim_rot_x": float(self.anim_rot_x_speed.value()),
            "anim_rot_y": float(self.anim_rot_y_speed.value()),
            "anim_rot_z": float(self.anim_rot_z_speed.value()),
        }

    def _to_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _apply_settings_dict(self, payload):
        if not payload:
            return

        self.image_resolution = int(payload.get("image_resolution", self.image_resolution))
        self._set_ground_size(int(payload.get("ground_size", self.ground_size)))
        self._set_ground_color(self._color_from_value(payload.get("ground_color", self._color_to_hex(self.ground_color)), self.ground_color))
        self._set_sky_color(self._color_from_value(payload.get("sky_color", self._color_to_hex(self.sky_color)), self.sky_color))
        self.code_script_text = str(payload.get("code_script", self.code_script_text))
        self.current_vf_file_path = str(payload.get("current_vf_file_path", self.current_vf_file_path))
        self.auto_start_animation = self._to_bool(payload.get("auto_start_animation", self.auto_start_animation))
        self.theme_mode = str(payload.get("theme_mode", self.theme_mode))
        self.resume_last_session = self._to_bool(payload.get("resume_last_session", self.resume_last_session))
        self.orbit_enabled = self._to_bool(payload.get("orbit_enabled", self.orbit_enabled))
        self.orbit_module = str(payload.get("orbit_module", self.orbit_module))
        self.orbit_plane = str(payload.get("orbit_plane", self.orbit_plane))
        self.orbit_radius = float(payload.get("orbit_radius", self.orbit_radius))
        self.orbit_speed = float(payload.get("orbit_speed", self.orbit_speed))
        self.orbit_spin_axes = str(payload.get("orbit_spin_axes", self.orbit_spin_axes))
        self.orbit_spin_speed = float(payload.get("orbit_spin_speed", self.orbit_spin_speed))
        self._apply_theme()

        self.remove_bg_check.setChecked(self._to_bool(payload.get("remove_background", self.remove_bg_check.isChecked())))
        self.bg_threshold_slider.setValue(int(payload.get("bg_threshold", self.bg_threshold_slider.value())))
        self.height_slider.setValue(int(payload.get("extrude_height", self.height_slider.value())))
        self.part_gap_slider.setValue(int(payload.get("parts_separation", self.part_gap_slider.value())))

        mode = str(payload.get("color_mode", self.color_mode.currentText()))
        idx = self.color_mode.findText(mode)
        if idx >= 0:
            self.color_mode.setCurrentIndex(idx)

        self.scale_slider.setValue(int(payload.get("global_scale", self.scale_slider.value())))
        self.move_x_slider.setValue(int(payload.get("global_move_x", self.move_x_slider.value())))
        self.move_y_slider.setValue(int(payload.get("global_move_y", self.move_y_slider.value())))
        self.move_z_slider.setValue(int(payload.get("global_move_z", self.move_z_slider.value())))
        self.rot_x_slider.setValue(int(payload.get("global_rot_x", self.rot_x_slider.value())))
        self.rot_y_slider.setValue(int(payload.get("global_rot_y", self.rot_y_slider.value())))
        self.rot_z_slider.setValue(int(payload.get("global_rot_z", self.rot_z_slider.value())))

        self.anim_move_x_speed.setValue(float(payload.get("anim_move_x", self.anim_move_x_speed.value())))
        self.anim_move_y_speed.setValue(float(payload.get("anim_move_y", self.anim_move_y_speed.value())))
        self.anim_move_z_speed.setValue(float(payload.get("anim_move_z", self.anim_move_z_speed.value())))
        self.anim_rot_x_speed.setValue(float(payload.get("anim_rot_x", self.anim_rot_x_speed.value())))
        self.anim_rot_y_speed.setValue(float(payload.get("anim_rot_y", self.anim_rot_y_speed.value())))
        self.anim_rot_z_speed.setValue(float(payload.get("anim_rot_z", self.anim_rot_z_speed.value())))

        anim_on = self._to_bool(payload.get("animation_enabled", self.auto_start_animation))
        if self.orbit_enabled:
            anim_on = True
        self.keep_moving_check.setChecked(anim_on)
        self._apply_transform()

    def _save_settings_to_file(self, path=None):
        self._save_app_settings()

    def _load_settings_from_file(self, path=None, silent=False):
        try:
            payload = self.store.load_state("settings", {})
            self._apply_settings_dict(payload)
            return True
        except Exception as exc:
            logger.exception("Failed to load settings from database")
            if not silent:
                QtWidgets.QMessageBox.warning(self, "Settings", f"Failed to load settings from database:\n{exc}")
            return False

    def _save_settings_file_dialog(self):
        try:
            self._save_app_settings()
            QtWidgets.QMessageBox.information(self, "Settings", f"Saved settings to database:\n{self.database_path}")
        except Exception as exc:
            logger.exception("Failed to save settings to database")
            QtWidgets.QMessageBox.warning(self, "Settings", f"Failed to save settings to database:\n{exc}")

    def _load_settings_file_dialog(self):
        if self._load_settings_from_file(None):
            QtWidgets.QMessageBox.information(self, "Settings", f"Reloaded settings from database:\n{self.database_path}")

    def _reset_all_settings(self):
        defaults = {
            "image_resolution": 96,
            "ground_size": 500,
            "ground_color": "#808080",
            "sky_color": "#141414",
            "code_script": "",
            "current_vf_file_path": "",
            "auto_start_animation": False,
            "theme_mode": "Dark",
            "resume_last_session": True,
            "orbit_enabled": False,
            "orbit_module": "Primary",
            "orbit_plane": "XZ",
            "orbit_radius": 120.0,
            "orbit_speed": 20.0,
            "orbit_spin_axes": "Y",
            "orbit_spin_speed": 25.0,
            "remove_background": True,
            "bg_threshold": 245,
            "extrude_height": 25,
            "parts_separation": 0,
            "color_mode": "Original Image",
            "global_scale": 100,
            "global_move_x": 0,
            "global_move_y": 0,
            "global_move_z": 0,
            "global_rot_x": 0,
            "global_rot_y": 0,
            "global_rot_z": 0,
            "animation_enabled": False,
            "anim_move_x": 0.0,
            "anim_move_y": 0.0,
            "anim_move_z": 0.0,
            "anim_rot_x": 0.0,
            "anim_rot_y": 30.0,
            "anim_rot_z": 0.0,
        }
        self.animation_translation[:] = 0.0
        self.animation_rotation[:] = 0.0
        self._apply_settings_dict(defaults)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _apply_theme(self):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return

        mode = str(self.theme_mode).strip().lower()
        if mode == "light":
            app.setStyleSheet(
                """
                QWidget { background-color: #f3f3f3; color: #121212; }
                QMainWindow, QDockWidget { background-color: #efefef; }
                QMenuBar, QMenu, QToolTip { background-color: #ffffff; color: #101010; }
                QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QListWidget {
                    background-color: #ffffff; color: #101010; border: 1px solid #bbbbbb; border-radius: 4px; padding: 4px;
                }
                QSlider::groove:horizontal { background: #cccccc; height: 6px; border-radius: 3px; }
                QSlider::handle:horizontal { background: #2f7ed8; width: 14px; margin: -4px 0; border-radius: 7px; }
                """
            )
        else:
            app.setStyleSheet(
                """
                QWidget { background-color: #181818; color: #f0f0f0; }
                QMainWindow, QDockWidget { background-color: #141414; }
                QMenuBar, QMenu, QToolTip { background-color: #1f1f1f; color: #f0f0f0; }
                QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QListWidget {
                    background-color: #242424; color: #f0f0f0; border: 1px solid #444444; border-radius: 4px; padding: 4px;
                }
                QSlider::groove:horizontal { background: #444444; height: 6px; border-radius: 3px; }
                QSlider::handle:horizontal { background: #d4af37; width: 14px; margin: -4px 0; border-radius: 7px; }
                """
            )

    def _default_code_script(self):
        return (
            "# VF language (.vf)\n"
            "# Scene\n"
            "GROUND_SIZE 900\n"
            "GROUND_COLOR #606060\n"
            "SKY_COLOR #1A2230\n"
            "\n"
            "# Define reusable class defaults\n"
            "CLASS Crate type=box width=80 depth=80 height=80 color=#43B5FF\n"
            "NEW crateA class=Crate tx=0 ty=0 tz=40\n"
            "APPLY crateA target=primary\n"
            "\n"
            "# Add more modules with custom overrides\n"
            "NEW ballA type=sphere radius=28 segments=26 rings=18 color=#F26F4C tx=130 tz=28\n"
            "APPLY ballA target=add\n"
            "\n"
            "# Multi-face image module (4 side faces + top/bottom)\n"
            "FACEMODULE cubeA faces=4 radius=45 height=70 front=https://example.com/front.png right=https://example.com/right.png back=https://example.com/back.png left=https://example.com/left.png top=https://example.com/top.png bottom=https://example.com/bottom.png target=add\n"
            "\n"
            "# URL validation + media assets\n"
            "URLCHECK https://example.com/picture.png type=image\n"
            "PICTURE poster https://example.com/picture.png mode=asset\n"
            "AUDIO theme https://example.com/theme.mp3 volume=80 loop=true\n"
            "VIDEO intro https://example.com/intro.mp4 volume=60\n"
            "\n"
            "# Timeline example\n"
            "AT 0\n"
            "ROTATE axes=XY speed=40 duration=10\n"
            "WAIT 5\n"
            "SETCOLOR color=#FF3030 mode=Solid\n"
        )

    def _open_code_mode(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("VoxelForge VF Editor")
        dialog.resize(1040, 690)
        layout = QtWidgets.QVBoxLayout(dialog)

        toolbar = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton("Open .vf")
        save_btn = QtWidgets.QPushButton("Save .vf")
        save_as_btn = QtWidgets.QPushButton("Save As .vf")
        run_btn = QtWidgets.QPushButton("Run VF")
        live_check = QtWidgets.QCheckBox("Live Update")
        live_check.setChecked(True)
        reset_btn = QtWidgets.QPushButton("Insert Template")
        close_btn = QtWidgets.QPushButton("Close")
        toolbar.addWidget(open_btn)
        toolbar.addWidget(save_btn)
        toolbar.addWidget(save_as_btn)
        toolbar.addWidget(run_btn)
        toolbar.addWidget(live_check)
        toolbar.addWidget(reset_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(close_btn)
        layout.addLayout(toolbar)

        editor = VFCodeEditor()
        editor.setFont(QtGui.QFont("Consolas", 11))
        editor.setStyleSheet(
            "QPlainTextEdit { background: #1E1E1E; color: #D4D4D4; border: 1px solid #3C3C3C; }"
        )
        editor.setPlaceholderText(self._default_code_script())
        editor.setPlainText(self.code_script_text.strip() or self._default_code_script())
        editor._highlighter = VFCodeHighlighter(editor.document())
        layout.addWidget(editor)

        status = QtWidgets.QLabel("VF ready")
        status.setStyleSheet("QLabel { color: #8BC34A; font-weight: 600; }")
        layout.addWidget(status)
        last_live_text = {"value": editor.toPlainText()}
        live_timer = QtCore.QTimer(dialog)
        live_timer.setSingleShot(True)
        live_timer.setInterval(350)

        def _update_cursor_status():
            c = editor.textCursor()
            status.setText(f"VF ready | Line {c.blockNumber() + 1}, Col {c.positionInBlock() + 1}")

        editor.cursorPositionChanged.connect(_update_cursor_status)
        _update_cursor_status()

        def _open_vf_file():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog,
                "Open VF File",
                self.current_vf_file_path or os.path.dirname(os.path.abspath(__file__)),
                "VF Script (*.vf);;Text Files (*.txt);;All Files (*)",
            )
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                editor.setPlainText(content)
                self.current_vf_file_path = path
                self.code_script_text = content
                self.store.save_script(
                    os.path.basename(path),
                    content,
                    user_id=self.current_user["id"] if self.current_user else None,
                    path=path,
                )
                status.setText(f"Opened: {os.path.basename(path)}")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dialog, "VF Editor", f"Failed to open file:\n{exc}")

        def _save_vf_file(save_as=False):
            path = self.current_vf_file_path
            if save_as or not path:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    dialog,
                    "Save VF File",
                    path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "script.vf"),
                    "VF Script (*.vf)",
                )
                if not path:
                    return False
                if not path.lower().endswith(".vf"):
                    path += ".vf"
            try:
                text = editor.toPlainText()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                self.current_vf_file_path = path
                self.code_script_text = text
                self.store.save_script(
                    os.path.basename(path),
                    text,
                    user_id=self.current_user["id"] if self.current_user else None,
                    path=path,
                )
                status.setText(f"Saved: {os.path.basename(path)}")
                return True
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dialog, "VF Editor", f"Failed to save file:\n{exc}")
                return False

        def _run_from_editor(show_popup=False):
            text = editor.toPlainText()
            ok, summary, err = self._run_code_script_internal(text)
            if ok:
                if show_popup:
                    details = "\n".join(summary[:20])
                    if len(summary) > 20:
                        details += f"\n...and {len(summary) - 20} more actions."
                    QtWidgets.QMessageBox.information(self, "Code Mode", f"Script executed.\n{details}")
                else:
                    status.setText(f"Live Update OK | {len(summary)} actions")
                    status.setStyleSheet("QLabel { color: #8BC34A; font-weight: 600; }")
            else:
                if show_popup:
                    QtWidgets.QMessageBox.warning(self, "Code Mode", f"Script failed:\n{err}")
                else:
                    status.setText(f"Live Update Error | {err}")
                    status.setStyleSheet("QLabel { color: #EF5350; font-weight: 600; }")
            return ok

        def _schedule_live_update():
            if not live_check.isChecked():
                return
            text = editor.toPlainText()
            if text == last_live_text["value"]:
                return
            live_timer.start()

        def _on_live_timeout():
            if not live_check.isChecked():
                return
            text = editor.toPlainText()
            if text == last_live_text["value"]:
                return
            last_live_text["value"] = text
            _run_from_editor(show_popup=False)

        live_timer.timeout.connect(_on_live_timeout)
        editor.textChanged.connect(_schedule_live_update)

        open_btn.clicked.connect(_open_vf_file)
        save_btn.clicked.connect(lambda: _save_vf_file(save_as=False))
        save_as_btn.clicked.connect(lambda: _save_vf_file(save_as=True))
        run_btn.clicked.connect(lambda: _run_from_editor(show_popup=True))
        reset_btn.clicked.connect(lambda: editor.setPlainText(self._default_code_script()))
        close_btn.clicked.connect(dialog.close)

        dialog.exec_()

    def _run_code_script(self, script):
        ok, summary, err = self._run_code_script_internal(script)
        if ok:
            details = "\n".join(summary[:20])
            if len(summary) > 20:
                details += f"\n...and {len(summary) - 20} more actions."
            QtWidgets.QMessageBox.information(self, "Code Mode", f"Script executed.\n{details}")
        else:
            QtWidgets.QMessageBox.warning(self, "Code Mode", f"Script failed:\n{err}")

    def _run_code_script_internal(self, script):
        self.code_script_text = str(script)
        try:
            summary = self._execute_code_script(script)
            self.store.save_script(
                os.path.basename(self.current_vf_file_path) if self.current_vf_file_path else "current.vf",
                str(script),
                user_id=self.current_user["id"] if self.current_user else None,
                path=self.current_vf_file_path,
            )
            self._save_app_settings()
            return True, summary, None
        except Exception as exc:
            return False, [], str(exc)

    def _split_script_args(self, tokens):
        pos = []
        kwargs = {}
        key_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        for token in tokens:
            if "=" in token:
                k, v = token.split("=", 1)
                k = k.strip()
                if key_re.match(k):
                    kwargs[k.lower()] = v.strip()
                    continue
            pos.append(token)
        return pos, kwargs

    def _script_parse_color(self, value):
        text = str(value).strip()
        if "," in text:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Invalid RGB color '{value}'")
            r, g, b = [max(0, min(255, int(float(x)))) for x in parts]
            return QtGui.QColor(r, g, b)
        c = QtGui.QColor(text)
        if not c.isValid():
            raise ValueError(f"Invalid color '{value}'")
        return c

    def _script_parse_target(self, value):
        t = str(value).strip().lower()
        if t in ("auto", "primary", "secondary", "add", "extra"):
            return t
        raise ValueError(f"Invalid target '{value}'")

    def _script_to_bool(self, value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _vf_parse_axes(self, text):
        axes = str(text).strip().upper()
        if axes in ("X", "Y", "Z", "XY", "YZ", "XZ", "XYZ"):
            return axes
        raise ValueError("axes must be one of X,Y,Z,XY,YZ,XZ,XYZ")

    def _vf_set_rotation_speeds(self, axes, speed):
        speed = float(speed)
        self.anim_rot_x_speed.setValue(speed if "X" in axes else 0.0)
        self.anim_rot_y_speed.setValue(speed if "Y" in axes else 0.0)
        self.anim_rot_z_speed.setValue(speed if "Z" in axes else 0.0)
        if not self.keep_moving_check.isChecked():
            self.keep_moving_check.setChecked(True)

    def _vf_stop_rotation_speeds(self):
        self.anim_rot_x_speed.setValue(0.0)
        self.anim_rot_y_speed.setValue(0.0)
        self.anim_rot_z_speed.setValue(0.0)

    def _vf_set_mesh_color(self, color, mode):
        self.base_color = self._script_parse_color(color)
        mode_text = str(mode).strip().lower()
        if mode_text in ("solid", "solid color"):
            self.color_mode.setCurrentText("Solid")
        elif mode_text in ("original", "original image"):
            self.color_mode.setCurrentText("Original Image")
        elif mode_text in ("per-part", "perpart"):
            self.color_mode.setCurrentText("Per-Part")
        elif mode_text in ("radial", "radial gradient"):
            self.color_mode.setCurrentText("Radial Gradient")
        self._refresh_mesh_visual()

    def _vf_schedule_event(self, at_sec, callback, label):
        self.vf_timeline_events.append(
            {"at": max(0.0, float(at_sec)), "callback": callback, "label": str(label), "done": False}
        )

    def _vf_start_timeline_if_needed(self):
        if not self.vf_timeline_events:
            return
        self.vf_timeline_events.sort(key=lambda e: e["at"])
        self.vf_timeline_start_time = time.perf_counter()
        self.vf_timeline_timer.start()

    def _tick_vf_timeline(self):
        if not self.vf_timeline_events:
            self.vf_timeline_timer.stop()
            return
        elapsed = time.perf_counter() - self.vf_timeline_start_time
        pending = 0
        for ev in self.vf_timeline_events:
            if ev["done"]:
                continue
            if elapsed >= ev["at"]:
                try:
                    ev["callback"]()
                except Exception:
                    pass
                ev["done"] = True
            else:
                pending += 1
        if pending == 0:
            self.vf_timeline_timer.stop()

    def _download_url_bytes(self, url, expected_kind="any"):
        parsed = urlparse.urlparse(str(url))
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http/https URLs are allowed.")
        req = urlrequest.Request(str(url), headers={"User-Agent": "VoxelForge/1.0"})
        max_bytes = 50 * 1024 * 1024
        with urlrequest.urlopen(req, timeout=25) as resp:
            status = int(getattr(resp, "status", 200))
            if status < 200 or status >= 400:
                raise ValueError(f"URL returned HTTP {status}.")
            content_type = str(resp.headers.get("Content-Type", "")).lower()
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError("URL content is too large. Maximum allowed download is 50 MB.")
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("URL content is too large. Maximum allowed download is 50 MB.")

        if not data:
            raise ValueError("URL returned empty content.")

        if expected_kind == "image":
            if "image" not in content_type:
                try:
                    Image.open(io.BytesIO(data)).verify()
                except Exception as exc:
                    raise ValueError(f"URL is not a valid image: {exc}") from exc
        elif expected_kind == "audio":
            if "audio" not in content_type and not any(
                str(parsed.path).lower().endswith(ext) for ext in (".mp3", ".wav", ".ogg", ".m4a", ".flac")
            ):
                raise ValueError(f"URL does not look like audio. Content-Type: {content_type or 'unknown'}")
        elif expected_kind == "video":
            if "video" not in content_type and not any(
                str(parsed.path).lower().endswith(ext) for ext in (".mp4", ".webm", ".mov", ".mkv", ".avi")
            ):
                raise ValueError(f"URL does not look like video. Content-Type: {content_type or 'unknown'}")
        elif expected_kind == "model":
            if len(data) < 8:
                raise ValueError("Model URL returned invalid data.")

        return data, content_type

    def _vf_class_defaults(self):
        return {
            "type": "box",
            "width": 50.0,
            "depth": 50.0,
            "height": 50.0,
            "radius": 25.0,
            "segments": 24,
            "rings": 16,
            "color": "#43B5FF",
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
            "rx": 0.0,
            "ry": 0.0,
            "rz": 0.0,
            "scale": 1.0,
        }

    def _vf_merge_props(self, base, updates):
        out = dict(base)
        for k, v in updates.items():
            out[str(k).lower()] = v
        return out

    def _vf_parse_props(self, kwargs):
        props = {}
        for k, v in kwargs.items():
            key = str(k).lower()
            if key in ("type", "class", "color", "name", "target", "mode", "url"):
                props[key] = str(v)
            elif key in ("segments", "rings"):
                props[key] = min(256, max(3, int(float(v))))
            else:
                props[key] = float(v)
        return props

    def _vf_geometry_from_props(self, props):
        p = self._vf_merge_props(self._vf_class_defaults(), props)
        shape_type = str(p.get("type", "box")).strip().lower()
        color = self._script_parse_color(p.get("color", "#43B5FF"))
        if shape_type == "box":
            verts, faces, colors = self._build_box_mesh(float(p["width"]), float(p["depth"]), float(p["height"]), color)
        elif shape_type == "sphere":
            verts, faces, colors = self._build_sphere_mesh(float(p["radius"]), int(p["segments"]), int(p["rings"]), color)
        elif shape_type == "cylinder":
            verts, faces, colors = self._build_cylinder_mesh(float(p["radius"]), float(p["height"]), int(p["segments"]), color)
        else:
            raise ValueError(f"Unsupported type '{shape_type}'. Use box/sphere/cylinder.")
        return self._vf_apply_geom_transform(verts, p), faces, colors

    def _vf_apply_geom_transform(self, verts, props):
        out = np.asarray(verts, dtype=np.float32).copy()
        s = float(props.get("scale", 1.0))
        if abs(s - 1.0) > 1e-6:
            out *= s
        m = QtGui.QMatrix4x4()
        m.rotate(float(props.get("rx", 0.0)), 1, 0, 0)
        m.rotate(float(props.get("ry", 0.0)), 0, 1, 0)
        m.rotate(float(props.get("rz", 0.0)), 0, 0, 1)
        tx = float(props.get("tx", 0.0))
        ty = float(props.get("ty", 0.0))
        tz = float(props.get("tz", 0.0))
        if abs(tx) > 1e-6 or abs(ty) > 1e-6 or abs(tz) > 1e-6:
            m.translate(tx, ty, tz)
        for i in range(out.shape[0]):
            v = m.map(QtGui.QVector3D(float(out[i, 0]), float(out[i, 1]), float(out[i, 2])))
            out[i, 0], out[i, 1], out[i, 2] = v.x(), v.y(), v.z()
        return out

    def _avg_color_from_image_url(self, url):
        data, _ = self._download_url_bytes(url, expected_kind="image")
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        avg = arr.mean(axis=(0, 1))
        return np.array([avg[0] / 255.0, avg[1] / 255.0, avg[2] / 255.0], dtype=np.float32)

    def _build_image_face_module(self, side_count, radius, height, side_colors, top_color, bottom_color):
        n = max(3, int(side_count))
        r = max(0.1, float(radius))
        h = max(0.1, float(height))
        z0 = -h * 0.5
        z1 = h * 0.5

        verts = []
        faces = []
        tri_colors = []

        ring_bottom = []
        ring_top = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            x = r * math.cos(a)
            y = r * math.sin(a)
            ring_bottom.append(np.array([x, y, z0], dtype=np.float32))
            ring_top.append(np.array([x, y, z1], dtype=np.float32))

        def add_tri(a, b, c, color_rgb):
            idx = len(verts)
            verts.extend([a.tolist(), b.tolist(), c.tolist()])
            faces.append([idx, idx + 1, idx + 2])
            tri_colors.append(color_rgb.tolist())

        for i in range(n):
            j = (i + 1) % n
            c = side_colors[i % len(side_colors)]
            b0, b1 = ring_bottom[i], ring_bottom[j]
            t0, t1 = ring_top[i], ring_top[j]
            add_tri(b0, b1, t0, c)
            add_tri(b1, t1, t0, c)

        top_center = np.array([0.0, 0.0, z1], dtype=np.float32)
        bottom_center = np.array([0.0, 0.0, z0], dtype=np.float32)
        for i in range(n):
            j = (i + 1) % n
            add_tri(top_center, ring_top[i], ring_top[j], top_color)
            add_tri(bottom_center, ring_bottom[j], ring_bottom[i], bottom_color)

        return (
            np.asarray(verts, dtype=np.float32),
            np.asarray(faces, dtype=np.uint32),
            np.asarray(tri_colors, dtype=np.float32),
        )

    def _set_primary_mesh_from_geometry(self, verts, faces, colors):
        self.height_data = None
        self.alpha_data = None
        self.image_rgb = None
        self.mesh_base_vertices = verts.astype(np.float32)
        self.mesh_faces = faces.astype(np.uint32)
        self.face_part_ids = np.zeros((self.mesh_faces.shape[0],), dtype=np.int32)
        self.face_source_colors = colors.astype(np.float32)
        self.part_sizes = {0: int(self.mesh_faces.shape[0])} if self.mesh_faces.shape[0] else {}
        self._sync_part_colors(1 if self.mesh_faces.shape[0] else 0)
        self._refresh_parts_list()
        if self.parts_list.count() > 0:
            self.parts_list.setCurrentRow(0)
        self.selected_target = ("module", "primary")
        self._refresh_mesh_visual()

    def _attach_script_mesh(self, verts, faces, colors, target):
        if target == "primary" or (target == "auto" and self.mesh_item is None):
            self._set_primary_mesh_from_geometry(verts, faces, colors)
            return "mesh attached to primary"
        if target == "secondary":
            if self.secondary_mesh_item is None:
                self._set_secondary_mesh(verts, faces, colors)
                return "mesh attached to secondary"
            self._add_extra_module(verts, faces, colors)
            return "mesh attached as extra module"
        if self.secondary_mesh_item is None:
            self._set_secondary_mesh(verts, faces, colors)
            return "mesh attached to secondary"
        self._add_extra_module(verts, faces, colors)
        return "mesh attached as extra module"

    def _clear_scene(self):
        self.height_data = None
        self.alpha_data = None
        self.image_rgb = None
        if self.mesh_item is not None:
            self.view.removeItem(self.mesh_item)
            self.mesh_item = None
        self.mesh_base_vertices = None
        self.mesh_faces = None
        self.face_part_ids = None
        self.face_source_colors = None
        self.part_sizes = {}
        self.part_labels = None
        self.fg_mask = None
        self.parts_list.clear()

        if self.secondary_mesh_item is not None:
            self.view.removeItem(self.secondary_mesh_item)
            self.secondary_mesh_item = None
        self.secondary_base_vertices = None
        self.secondary_faces = None
        self.secondary_face_colors = None
        self.secondary_rot[:] = 0.0

        for module in self.extra_modules:
            item = module.get("mesh_item")
            if item is not None:
                self.view.removeItem(item)
        self.extra_modules = []
        self.selected_target = ("module", "primary")
        self._refresh_modules_list()
        self._update_preview_for_selection()

    def _build_box_mesh(self, width, depth, height, color):
        hx = max(0.1, float(width)) * 0.5
        hy = max(0.1, float(depth)) * 0.5
        hz = max(0.1, float(height)) * 0.5
        verts = np.array(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 6, 5], [4, 7, 6],
                [0, 4, 5], [0, 5, 1],
                [1, 5, 6], [1, 6, 2],
                [2, 6, 7], [2, 7, 3],
                [3, 7, 4], [3, 4, 0],
            ],
            dtype=np.uint32,
        )
        rgb = np.array([[color.redF(), color.greenF(), color.blueF()]], dtype=np.float32)
        return verts, faces, np.repeat(rgb, faces.shape[0], axis=0)

    def _build_sphere_mesh(self, radius, segments, rings, color):
        r = max(0.1, float(radius))
        seg = min(256, max(8, int(segments)))
        ring = min(128, max(6, int(rings)))
        verts = []
        faces = []
        for i in range(ring + 1):
            phi = math.pi * i / ring
            z = r * math.cos(phi)
            s = r * math.sin(phi)
            for j in range(seg):
                theta = 2.0 * math.pi * j / seg
                x = s * math.cos(theta)
                y = s * math.sin(theta)
                verts.append([x, y, z])
        for i in range(ring):
            for j in range(seg):
                a = i * seg + j
                b = i * seg + ((j + 1) % seg)
                c = (i + 1) * seg + j
                d = (i + 1) * seg + ((j + 1) % seg)
                faces.append([a, b, c])
                faces.append([b, d, c])
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)
        rgb = np.array([[color.redF(), color.greenF(), color.blueF()]], dtype=np.float32)
        return verts, faces, np.repeat(rgb, faces.shape[0], axis=0)

    def _build_cylinder_mesh(self, radius, height, segments, color):
        r = max(0.1, float(radius))
        h = max(0.1, float(height))
        seg = min(256, max(8, int(segments)))
        half_h = h * 0.5
        verts = []
        faces = []
        for j in range(seg):
            theta = 2.0 * math.pi * j / seg
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            verts.append([x, y, -half_h])
            verts.append([x, y, half_h])
        bottom_center = len(verts)
        verts.append([0.0, 0.0, -half_h])
        top_center = len(verts)
        verts.append([0.0, 0.0, half_h])

        for j in range(seg):
            nj = (j + 1) % seg
            b0 = 2 * j
            t0 = b0 + 1
            b1 = 2 * nj
            t1 = b1 + 1
            faces.append([b0, b1, t0])
            faces.append([b1, t1, t0])
            faces.append([bottom_center, b1, b0])
            faces.append([top_center, t0, t1])
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)
        rgb = np.array([[color.redF(), color.greenF(), color.blueF()]], dtype=np.float32)
        return verts, faces, np.repeat(rgb, faces.shape[0], axis=0)

    def _run_web_image(self, url, target):
        data, _ = self._download_url_bytes(url, expected_kind="image")
        return self._run_web_image_data(url, data, target)

    def _run_web_image_data(self, url, data, target):
        asset_key = hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:16]
        self.store.save_blob(
            f"url-image:{asset_key}",
            data,
            "image",
            user_id=self.current_user["id"] if self.current_user else None,
            content_type="image/unknown",
        )
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        img = img.resize((self.image_resolution, self.image_resolution))
        rgba = np.asarray(img, dtype=np.uint8)

        if target == "primary" or (target == "auto" and self.mesh_item is None):
            self.height_data, self.alpha_data, self.image_rgb = self._rgba_to_height_alpha_rgb(rgba)
            self._rebuild_mesh()
            return "image loaded to primary"

        mesh = self._build_mesh_data_from_rgba(rgba)
        if mesh[0] is None:
            raise ValueError("Image did not produce a valid mesh.")
        verts, faces, _, colors = mesh
        return self._attach_script_mesh(verts, faces, colors, target)

    def _run_web_model(self, url, target):
        data, _ = self._download_url_bytes(url, expected_kind="model")
        asset_key = hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:16]
        self.store.save_blob(
            f"url-model:{asset_key}",
            data,
            "model",
            user_id=self.current_user["id"] if self.current_user else None,
            content_type="application/x-voxelforge-model",
        )
        suffix = os.path.splitext(urlparse.urlparse(url).path)[1] or ".vf3d"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            if target == "primary" or (target == "auto" and self.mesh_item is None):
                if not self._read_private_model_file(tmp_path, show_feedback=False):
                    raise ValueError("Downloaded model is not a valid .vf3d file.")
                return "model loaded to primary"
            model = self._decode_private_model(tmp_path)
            if model is None:
                raise ValueError("Downloaded model is not a valid .vf3d file.")
            return self._attach_script_mesh(model["vertices"], model["faces"], model["face_source_colors"], target)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _execute_code_script(self, script):
        self.vf_timeline_timer.stop()
        self.vf_timeline_events = []
        actions = []
        timeline_cursor = 0.0
        for line_no, raw in enumerate(str(script).splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tokens = shlex.split(line)
                if not tokens:
                    continue
                cmd = tokens[0].upper()
                pos, kwargs = self._split_script_args(tokens[1:])

                if cmd == "GROUND_SIZE":
                    if not pos:
                        raise ValueError("GROUND_SIZE requires one number.")
                    self._set_ground_size(int(float(pos[0])))
                    actions.append(f"L{line_no}: ground size updated")
                    continue
                if cmd == "AT":
                    if not pos:
                        raise ValueError("AT requires seconds.")
                    timeline_cursor = max(0.0, float(pos[0]))
                    actions.append(f"L{line_no}: timeline cursor set to {timeline_cursor:.2f}s")
                    continue
                if cmd == "WAIT":
                    if not pos:
                        raise ValueError("WAIT requires seconds.")
                    timeline_cursor += max(0.0, float(pos[0]))
                    actions.append(f"L{line_no}: timeline cursor moved to {timeline_cursor:.2f}s")
                    continue
                if cmd == "ROTATE":
                    axes = self._vf_parse_axes(kwargs.get("axes", "Y"))
                    speed = float(kwargs.get("speed", 30.0))
                    duration = max(0.0, float(kwargs.get("duration", 1.0)))
                    start_at = timeline_cursor
                    stop_at = timeline_cursor + duration
                    self._vf_schedule_event(
                        start_at,
                        lambda a=axes, s=speed: self._vf_set_rotation_speeds(a, s),
                        f"rotate start {axes}",
                    )
                    self._vf_schedule_event(
                        stop_at,
                        self._vf_stop_rotation_speeds,
                        "rotate stop",
                    )
                    actions.append(
                        f"L{line_no}: scheduled rotation axes={axes} speed={speed:.1f} for {duration:.2f}s at {start_at:.2f}s"
                    )
                    continue
                if cmd == "SETCOLOR":
                    color = kwargs.get("color", pos[0] if pos else None)
                    if color is None:
                        raise ValueError("SETCOLOR requires color.")
                    mode = kwargs.get("mode", "Solid")
                    at_sec = timeline_cursor
                    self._vf_schedule_event(
                        at_sec,
                        lambda c=color, m=mode: self._vf_set_mesh_color(c, m),
                        f"set color {color}",
                    )
                    actions.append(f"L{line_no}: scheduled color change at {at_sec:.2f}s")
                    continue
                if cmd == "GROUND_COLOR":
                    if not pos:
                        raise ValueError("GROUND_COLOR requires a color.")
                    self._set_ground_color(self._script_parse_color(pos[0]))
                    actions.append(f"L{line_no}: ground color updated")
                    continue
                if cmd == "SKY_COLOR":
                    if not pos:
                        raise ValueError("SKY_COLOR requires a color.")
                    self._set_sky_color(self._script_parse_color(pos[0]))
                    actions.append(f"L{line_no}: sky color updated")
                    continue
                if cmd == "CLEAR":
                    self._clear_scene()
                    actions.append(f"L{line_no}: scene cleared")
                    continue
                if cmd == "IMAGE":
                    if not pos:
                        raise ValueError("IMAGE requires a URL.")
                    target = self._script_parse_target(kwargs.get("target", pos[1] if len(pos) > 1 else "auto"))
                    actions.append(f"L{line_no}: {self._run_web_image(pos[0], target)}")
                    continue
                if cmd == "PICTURE":
                    if len(pos) < 2:
                        raise ValueError("PICTURE requires name and URL.")
                    name, url = str(pos[0]), str(pos[1])
                    mode = str(kwargs.get("mode", "asset")).strip().lower()
                    data, content_type = self._download_url_bytes(url, expected_kind="image")
                    self.store.save_blob(
                        f"asset:picture:{name}",
                        data,
                        "picture",
                        user_id=self.current_user["id"] if self.current_user else None,
                        content_type=content_type or "image/unknown",
                    )
                    self.vf_assets["picture"][name] = {
                        "url": url,
                        "bytes": len(data),
                        "content_type": content_type,
                        "mode": mode,
                    }
                    if mode == "mesh":
                        target = self._script_parse_target(kwargs.get("target", "auto"))
                        actions.append(f"L{line_no}: {self._run_web_image(url, target)}")
                    else:
                        actions.append(f"L{line_no}: picture '{name}' validated and stored")
                    continue
                if cmd == "MODEL":
                    if not pos:
                        raise ValueError("MODEL requires a URL.")
                    target = self._script_parse_target(kwargs.get("target", pos[1] if len(pos) > 1 else "auto"))
                    actions.append(f"L{line_no}: {self._run_web_model(pos[0], target)}")
                    continue
                if cmd == "AUDIO":
                    if len(pos) < 2:
                        raise ValueError("AUDIO requires name and URL.")
                    name, url = str(pos[0]), str(pos[1])
                    data, content_type = self._download_url_bytes(url, expected_kind="audio")
                    self.store.save_blob(
                        f"asset:audio:{name}",
                        data,
                        "audio",
                        user_id=self.current_user["id"] if self.current_user else None,
                        content_type=content_type or "audio/unknown",
                    )
                    self.vf_assets["audio"][name] = {
                        "url": url,
                        "volume": int(float(kwargs.get("volume", 100))),
                        "loop": self._script_to_bool(kwargs.get("loop", False)),
                        "content_type": content_type,
                    }
                    actions.append(f"L{line_no}: audio '{name}' validated and stored")
                    continue
                if cmd == "VIDEO":
                    if len(pos) < 2:
                        raise ValueError("VIDEO requires name and URL.")
                    name, url = str(pos[0]), str(pos[1])
                    data, content_type = self._download_url_bytes(url, expected_kind="video")
                    self.store.save_blob(
                        f"asset:video:{name}",
                        data,
                        "video",
                        user_id=self.current_user["id"] if self.current_user else None,
                        content_type=content_type or "video/unknown",
                    )
                    self.vf_assets["video"][name] = {
                        "url": url,
                        "volume": int(float(kwargs.get("volume", 100))),
                        "loop": self._script_to_bool(kwargs.get("loop", False)),
                        "content_type": content_type,
                    }
                    actions.append(f"L{line_no}: video '{name}' validated and stored")
                    continue
                if cmd == "URLCHECK":
                    if not pos:
                        raise ValueError("URLCHECK requires URL.")
                    check_type = str(kwargs.get("type", "any")).strip().lower()
                    expected = "any"
                    if check_type in ("image", "picture"):
                        expected = "image"
                    elif check_type in ("audio",):
                        expected = "audio"
                    elif check_type in ("video",):
                        expected = "video"
                    elif check_type in ("model", "vf3d"):
                        expected = "model"
                    data, content_type = self._download_url_bytes(pos[0], expected_kind=expected)
                    actions.append(f"L{line_no}: URL OK ({len(data)} bytes, content-type={content_type or 'unknown'})")
                    continue
                if cmd == "FACEMODULE":
                    if not pos:
                        raise ValueError("FACEMODULE requires module name.")
                    module_name = str(pos[0])
                    side_count = int(float(kwargs.get("faces", 4)))
                    radius = float(kwargs.get("radius", 40))
                    height = float(kwargs.get("height", kwargs.get("depth", 60)))
                    target = self._script_parse_target(kwargs.get("target", "auto"))
                    default_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)

                    side_colors = []
                    side_url_map = []
                    for i in range(1, side_count + 1):
                        key = f"face{i}"
                        if key in kwargs:
                            url = kwargs[key]
                            side_url_map.append((i, url))
                            side_colors.append(self._avg_color_from_image_url(url))
                    for named_key in ("front", "right", "back", "left"):
                        if named_key in kwargs:
                            url = kwargs[named_key]
                            side_url_map.append((named_key, url))
                            side_colors.append(self._avg_color_from_image_url(url))
                    if not side_colors:
                        side_colors = [default_color]

                    top_color = default_color
                    bottom_color = default_color
                    if "top" in kwargs:
                        top_color = self._avg_color_from_image_url(kwargs["top"])
                    if "bottom" in kwargs:
                        bottom_color = self._avg_color_from_image_url(kwargs["bottom"])

                    verts, faces, colors = self._build_image_face_module(
                        side_count=side_count,
                        radius=radius,
                        height=height,
                        side_colors=side_colors,
                        top_color=top_color,
                        bottom_color=bottom_color,
                    )
                    attach_info = self._attach_script_mesh(verts, faces, colors, target)
                    self.vf_modules[module_name] = {
                        "type": "facemodule",
                        "faces": side_count,
                        "radius": radius,
                        "height": height,
                        "target": target,
                        "side_images": side_url_map,
                        "top": kwargs.get("top", ""),
                        "bottom": kwargs.get("bottom", ""),
                    }
                    actions.append(
                        f"L{line_no}: module '{module_name}' built with {side_count} side faces ({len(side_url_map)} images) -> {attach_info}"
                    )
                    continue
                if cmd == "BOX":
                    if len(pos) < 3:
                        raise ValueError("BOX requires width depth height.")
                    color_text = kwargs.get("color", pos[3] if len(pos) > 3 else "#43B5FF")
                    target = self._script_parse_target(kwargs.get("target", pos[4] if len(pos) > 4 else "auto"))
                    verts, faces, colors = self._build_box_mesh(float(pos[0]), float(pos[1]), float(pos[2]), self._script_parse_color(color_text))
                    actions.append(f"L{line_no}: {self._attach_script_mesh(verts, faces, colors, target)}")
                    continue
                if cmd == "SPHERE":
                    if not pos:
                        raise ValueError("SPHERE requires radius.")
                    radius = float(pos[0])
                    segments = int(kwargs.get("segments", pos[1] if len(pos) > 1 else 24))
                    rings = int(kwargs.get("rings", pos[2] if len(pos) > 2 else 16))
                    color_text = kwargs.get("color", pos[3] if len(pos) > 3 else "#F26F4C")
                    target = self._script_parse_target(kwargs.get("target", pos[4] if len(pos) > 4 else "auto"))
                    verts, faces, colors = self._build_sphere_mesh(radius, segments, rings, self._script_parse_color(color_text))
                    actions.append(f"L{line_no}: {self._attach_script_mesh(verts, faces, colors, target)}")
                    continue
                if cmd == "CYLINDER":
                    if len(pos) < 2:
                        raise ValueError("CYLINDER requires radius and height.")
                    radius = float(pos[0])
                    height = float(pos[1])
                    segments = int(kwargs.get("segments", pos[2] if len(pos) > 2 else 20))
                    color_text = kwargs.get("color", pos[3] if len(pos) > 3 else "#70D090")
                    target = self._script_parse_target(kwargs.get("target", pos[4] if len(pos) > 4 else "auto"))
                    verts, faces, colors = self._build_cylinder_mesh(radius, height, segments, self._script_parse_color(color_text))
                    actions.append(f"L{line_no}: {self._attach_script_mesh(verts, faces, colors, target)}")
                    continue
                if cmd == "CLASS":
                    if not pos:
                        raise ValueError("CLASS requires class name.")
                    class_name = str(pos[0])
                    props = self._vf_parse_props(kwargs)
                    self.vf_classes[class_name] = self._vf_merge_props(self._vf_class_defaults(), props)
                    actions.append(f"L{line_no}: class '{class_name}' defined")
                    continue
                if cmd == "NEW":
                    if not pos:
                        raise ValueError("NEW requires module name.")
                    module_name = str(pos[0])
                    class_name = str(kwargs.get("class", "")).strip()
                    props = self._vf_parse_props(kwargs)
                    base = self._vf_class_defaults()
                    if class_name:
                        if class_name not in self.vf_classes:
                            raise ValueError(f"Unknown class '{class_name}'.")
                        base = self._vf_merge_props(base, self.vf_classes[class_name])
                    for key in ("class", "target"):
                        props.pop(key, None)
                    self.vf_modules[module_name] = self._vf_merge_props(base, props)
                    actions.append(f"L{line_no}: module '{module_name}' created")
                    continue
                if cmd == "SET":
                    if not pos:
                        raise ValueError("SET requires module name.")
                    module_name = str(pos[0])
                    if module_name not in self.vf_modules:
                        raise ValueError(f"Unknown module '{module_name}'.")
                    props = self._vf_parse_props(kwargs)
                    for key in ("class", "target"):
                        props.pop(key, None)
                    self.vf_modules[module_name] = self._vf_merge_props(self.vf_modules[module_name], props)
                    actions.append(f"L{line_no}: module '{module_name}' updated")
                    continue
                if cmd == "APPLY":
                    if not pos:
                        raise ValueError("APPLY requires module name.")
                    module_name = str(pos[0])
                    if module_name not in self.vf_modules:
                        raise ValueError(f"Unknown module '{module_name}'.")
                    target = self._script_parse_target(kwargs.get("target", "auto"))
                    verts, faces, colors = self._vf_geometry_from_props(self.vf_modules[module_name])
                    actions.append(f"L{line_no}: module '{module_name}' -> {self._attach_script_mesh(verts, faces, colors, target)}")
                    continue
                if cmd == "DELETE":
                    if not pos:
                        raise ValueError("DELETE requires module name.")
                    module_name = str(pos[0])
                    if module_name in self.vf_modules:
                        del self.vf_modules[module_name]
                        actions.append(f"L{line_no}: module '{module_name}' deleted")
                    else:
                        actions.append(f"L{line_no}: module '{module_name}' not found")
                    continue
                if cmd == "SAVEVF":
                    path = str(pos[0]) if pos else self.current_vf_file_path
                    if not path:
                        raise ValueError("SAVEVF requires a path or existing open .vf file.")
                    if not path.lower().endswith(".vf"):
                        path += ".vf"
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(str(script))
                    self.current_vf_file_path = path
                    actions.append(f"L{line_no}: script saved to {path}")
                    continue
                if cmd == "LOADVF":
                    if not pos:
                        raise ValueError("LOADVF requires path.")
                    path = str(pos[0])
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = f.read()
                    self.current_vf_file_path = path
                    self.code_script_text = loaded
                    actions.append(f"L{line_no}: script loaded from {path}")
                    continue
                if cmd == "HELP":
                    actions.append(
                        "VF commands: GROUND_SIZE, GROUND_COLOR, SKY_COLOR, CLEAR, IMAGE, PICTURE, MODEL, AUDIO, VIDEO, URLCHECK, "
                        "FACEMODULE, AT, WAIT, ROTATE, SETCOLOR, BOX, SPHERE, CYLINDER, CLASS, NEW, SET, APPLY, DELETE, SAVEVF, LOADVF."
                    )
                    continue
                raise ValueError(
                    f"Unknown command '{cmd}'. Use HELP for command list."
                )
            except Exception as exc:
                raise ValueError(f"Line {line_no}: {exc}") from exc

        if not actions:
            actions.append("No commands executed.")
        self._vf_start_timeline_if_needed()
        self._apply_transform()
        return actions

    def _is_image_file(self, path):
        return os.path.splitext(path)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp")

    def _is_model_file(self, path):
        return os.path.splitext(path)[1].lower() == ".vf3d"

    def _build_main_transform_matrix(self):
        m = QtGui.QMatrix4x4()
        scale = self.scale_slider.value() / 100.0
        move_x = float(self.move_x_slider.value()) + float(self.animation_translation[0])
        move_y = float(self.move_y_slider.value()) + float(self.animation_translation[1])
        move_z = float(self.move_z_slider.value()) + float(self.animation_translation[2])
        rot_x = float(self.rot_x_slider.value()) + float(self.animation_rotation[0])
        rot_y = float(self.rot_y_slider.value()) + float(self.animation_rotation[1])
        rot_z = float(self.rot_z_slider.value()) + float(self.animation_rotation[2])
        if self.orbit_enabled and self.orbit_module == "Primary":
            ox, oy, oz = self._orbit_offset_xyz()
            sx, sy, sz = self._orbit_spin_components()
            move_x += ox
            move_y += oy
            move_z += oz
            rot_x += sx
            rot_y += sy
            rot_z += sz
        m.scale(scale, scale, scale)
        m.rotate(rot_x, 1, 0, 0)
        m.rotate(rot_y, 0, 1, 0)
        m.rotate(rot_z, 0, 0, 1)
        m.translate(move_x, move_y, move_z)
        return m

    def _build_secondary_transform_matrix(self):
        m = QtGui.QMatrix4x4()
        scale = self.second_scale_slider.value() / 100.0
        move_x = float(self.second_move_x_slider.value())
        move_y = float(self.second_move_y_slider.value())
        move_z = float(self.second_move_z_slider.value())
        rot_x = float(self.secondary_rot[0])
        rot_y = float(self.secondary_rot[1])
        rot_z = float(self.secondary_rot[2])
        if self.orbit_enabled and self.orbit_module == "Secondary":
            ox, oy, oz = self._orbit_offset_xyz()
            sx, sy, sz = self._orbit_spin_components()
            move_x += ox
            move_y += oy
            move_z += oz
            rot_x = sx
            rot_y = sy
            rot_z = sz
        m.scale(scale, scale, scale)
        m.rotate(rot_x, 1, 0, 0)
        m.rotate(rot_y, 0, 1, 0)
        m.rotate(rot_z, 0, 0, 1)
        m.translate(move_x, move_y, move_z)
        return m

    def _map_world_to_screen(self, world):
        if self.view.width() <= 0 or self.view.height() <= 0:
            return None
        vec = QtGui.QVector4D(float(world[0]), float(world[1]), float(world[2]), 1.0)
        try:
            proj = self.view.projectionMatrix()
        except TypeError:
            region = (0, 0, int(self.view.width()), int(self.view.height()))
            viewport = (0, 0, int(self.view.width()), int(self.view.height()))
            proj = self.view.projectionMatrix(region, viewport)
        clip = proj * self.view.viewMatrix() * vec
        w = clip.w()
        if abs(w) < 1e-6:
            return None
        ndc_x = clip.x() / w
        ndc_y = clip.y() / w
        sx = (ndc_x + 1.0) * 0.5 * self.view.width()
        sy = (1.0 - (ndc_y + 1.0) * 0.5) * self.view.height()
        return np.array([sx, sy], dtype=np.float32)

    def _update_preview_mesh(self, verts, faces, colors, title):
        self.preview_title.setText(title)
        if self.preview_mesh_item is not None:
            self.preview_view.removeItem(self.preview_mesh_item)
            self.preview_mesh_item = None
        if verts is None or faces is None or len(faces) == 0:
            return
        face_colors = np.ones((faces.shape[0], 4), dtype=np.float32)
        if colors is not None and colors.shape[0] == faces.shape[0]:
            face_colors[:, :3] = colors[:, :3]
        self.preview_mesh_item = gl.GLMeshItem(
            vertexes=verts.astype(np.float32),
            faces=faces.astype(np.uint32),
            faceColors=face_colors,
            smooth=False,
            drawEdges=False,
            shader="shaded",
        )
        self.preview_mesh_item.setGLOptions("opaque")
        self.preview_view.addItem(self.preview_mesh_item)
        self.preview_angle = 0.0

    def _get_part_preview_geometry(self, pid):
        if self.mesh_base_vertices is None or self.mesh_faces is None or self.face_part_ids is None:
            return None, None, None
        mask = self.face_part_ids == int(pid)
        if not np.any(mask):
            return None, None, None
        part_faces = self.mesh_faces[mask]
        part_colors = self.face_source_colors[mask] if self.face_source_colors is not None else None
        old_to_new = {}
        new_vertices = []
        new_faces = []
        for face in part_faces:
            nf = []
            for idx in face:
                idx = int(idx)
                if idx not in old_to_new:
                    old_to_new[idx] = len(new_vertices)
                    new_vertices.append(self.mesh_base_vertices[idx])
                nf.append(old_to_new[idx])
            new_faces.append(nf)
        return np.asarray(new_vertices, dtype=np.float32), np.asarray(new_faces, dtype=np.uint32), part_colors

    def _update_preview_for_selection(self):
        kind, value = self.selected_target
        if kind == "module" and value == "primary":
            self._update_preview_mesh(
                self.mesh_base_vertices,
                self.mesh_faces,
                self.face_source_colors,
                "Selected: Primary Module",
            )
            return
        if kind == "module" and value == "secondary":
            self._update_preview_mesh(
                self.secondary_base_vertices,
                self.secondary_faces,
                self.secondary_face_colors,
                "Selected: Secondary Module",
            )
            return
        if kind == "module" and isinstance(value, tuple) and value[0] == "extra":
            idx = int(value[1])
            if 0 <= idx < len(self.extra_modules):
                mod = self.extra_modules[idx]
                self._update_preview_mesh(
                    mod["base_vertices"],
                    mod["faces"],
                    mod["face_colors"],
                    f"Selected: Module {idx + 3}",
                )
                return
        if kind == "part":
            verts, faces, colors = self._get_part_preview_geometry(int(value))
            self._update_preview_mesh(verts, faces, colors, f"Selected: Part {int(value) + 1}")
            return
        self._update_preview_mesh(None, None, None, "Selected: None")

    def _pick_scene_target(self, pos):
        candidates = []
        p = np.array([float(pos.x()), float(pos.y())], dtype=np.float32)

        if self.mesh_base_vertices is not None and self.mesh_base_vertices.size:
            main_center_local = self.mesh_base_vertices.mean(axis=0)
            main_center_world = self._build_main_transform_matrix().map(
                QtGui.QVector3D(float(main_center_local[0]), float(main_center_local[1]), float(main_center_local[2]))
            )
            sp = self._map_world_to_screen((main_center_world.x(), main_center_world.y(), main_center_world.z()))
            if sp is not None:
                candidates.append((np.linalg.norm(sp - p), ("module", "primary")))

            if self.face_part_ids is not None:
                for pid in np.unique(self.face_part_ids):
                    mask = self.face_part_ids == int(pid)
                    if not np.any(mask):
                        continue
                    idxs = self.mesh_faces[mask].ravel()
                    center_local = self.mesh_base_vertices[idxs].mean(axis=0)
                    part_t = self.part_transforms.get(int(pid), np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    center_local = center_local + part_t
                    cw = self._build_main_transform_matrix().map(
                        QtGui.QVector3D(float(center_local[0]), float(center_local[1]), float(center_local[2]))
                    )
                    sp_part = self._map_world_to_screen((cw.x(), cw.y(), cw.z()))
                    if sp_part is not None:
                        candidates.append((np.linalg.norm(sp_part - p), ("part", int(pid))))

        if self.secondary_base_vertices is not None and self.secondary_base_vertices.size:
            sec_center_local = self.secondary_base_vertices.mean(axis=0)
            sec_center_world = self._build_secondary_transform_matrix().map(
                QtGui.QVector3D(float(sec_center_local[0]), float(sec_center_local[1]), float(sec_center_local[2]))
            )
            sp2 = self._map_world_to_screen((sec_center_world.x(), sec_center_world.y(), sec_center_world.z()))
            if sp2 is not None:
                candidates.append((np.linalg.norm(sp2 - p), ("module", "secondary")))

        for i, mod in enumerate(self.extra_modules):
            verts = mod["base_vertices"]
            if verts is None or verts.size == 0:
                continue
            c = verts.mean(axis=0)
            t = mod["transform"]
            m = QtGui.QMatrix4x4()
            m.scale(float(t[0]) / 100.0, float(t[0]) / 100.0, float(t[0]) / 100.0)
            m.rotate(float(t[4]), 1, 0, 0)
            m.rotate(float(t[5]), 0, 1, 0)
            m.rotate(float(t[6]), 0, 0, 1)
            m.translate(float(t[1]), float(t[2]), float(t[3]))
            cw = m.map(QtGui.QVector3D(float(c[0]), float(c[1]), float(c[2])))
            spx = self._map_world_to_screen((cw.x(), cw.y(), cw.z()))
            if spx is not None:
                candidates.append((np.linalg.norm(spx - p), ("module", ("extra", i))))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        if candidates[0][0] > 70.0:
            return None
        return candidates[0][1]

    def _handle_drop_file(self, path):
        if not os.path.isfile(path):
            return
        if self._is_image_file(path):
            if self.mesh_item is None:
                self._load_primary_image_from_path(path)
            elif self.secondary_mesh_item is None:
                self._load_secondary_image_from_path(path)
            else:
                self._load_secondary_image_from_path(path)
            return
        if self._is_model_file(path):
            if self.mesh_item is None:
                self._read_private_model_file(path, show_feedback=False)
            elif self.secondary_mesh_item is None:
                model = self._decode_private_model(path)
                if model is not None:
                    self._set_secondary_mesh(model["vertices"], model["faces"], model["face_source_colors"])
            else:
                model = self._decode_private_model(path)
                if model is not None:
                    self._add_extra_module(model["vertices"], model["faces"], model["face_source_colors"])

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and self.isActiveWindow():
            if self._handle_nav_key(event, True):
                return True
        if event.type() == QtCore.QEvent.KeyRelease and self.isActiveWindow():
            if self._handle_nav_key(event, False):
                return True
        if obj is self.view and event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton:
                self.view.setFocus()
                try:
                    target = self._pick_scene_target(event.pos())
                except Exception:
                    target = None
                if target is not None:
                    self.selected_target = target
                    if target[0] == "part":
                        pid = int(target[1])
                        for i in range(self.parts_list.count()):
                            item = self.parts_list.item(i)
                            if int(item.data(QtCore.Qt.UserRole)) == pid:
                                self.parts_list.setCurrentRow(i)
                                break
                    else:
                        self._update_preview_for_selection()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if self._handle_nav_key(event, True):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._handle_nav_key(event, False):
            return
        super().keyReleaseEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and (self._is_image_file(path) or self._is_model_file(path)):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path:
                continue
            self._handle_drop_file(path)
        event.acceptProposedAction()

    def _save_last_session_snapshot(self):
        if self.mesh_base_vertices is None or self.mesh_faces is None:
            return
        try:
            self.store.save_blob(
                "last_session_model",
                self._private_model_bytes(),
                "vf3d-session",
                user_id=self.current_user["id"] if self.current_user else None,
                content_type="application/x-voxelforge-model",
            )
        except Exception:
            logger.exception("Failed to save last session snapshot")

    def _resume_last_session_on_startup(self):
        raw = self.store.load_blob("last_session_model")
        if raw is None:
            return
        self._read_private_model_bytes(raw, show_feedback=False)

    def _resume_last_session_manual(self):
        raw = self.store.load_blob("last_session_model")
        if raw is None:
            QtWidgets.QMessageBox.information(self, "Resume Session", "No saved last session found yet.")
            return
        ok = self._read_private_model_bytes(raw, show_feedback=False)
        if ok:
            QtWidgets.QMessageBox.information(self, "Resume Session", "Resumed from your last session.")
        else:
            QtWidgets.QMessageBox.warning(self, "Resume Session", "Failed to resume last session.")

    def closeEvent(self, event):
        self._save_app_settings()
        self._save_last_session_snapshot()
        self.store.close()
        super().closeEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = HeightMapViewer()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
