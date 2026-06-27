from __future__ import annotations

from PyQt5 import QtGui, QtWidgets
from pyqtgraph.Qt import QtCore

from .editor import VFCodeEditor, VFCodeHighlighter
from .engine2d import Engine2D, Scene2D, Vec2
from .engine2d_script import DEFAULT_2D_SCRIPT, SCRIPT_HELP, parse_2d_script


class Engine2DStudioDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial_script: str = ""):
        super().__init__(parent)
        self.setWindowTitle("VoxelForge 2D Coding Studio")
        self.resize(1220, 760)
        self.engine = Engine2D()
        self.engine.load_scene(Scene2D("Empty"))
        self._last_script = ""
        self._items = []

        root = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run")
        self.template_btn = QtWidgets.QPushButton("Template")
        self.live_check = QtWidgets.QCheckBox("Live")
        self.live_check.setChecked(True)
        self.pause_check = QtWidgets.QCheckBox("Pause")
        toolbar.addWidget(self.run_btn)
        toolbar.addWidget(self.template_btn)
        toolbar.addWidget(self.live_check)
        toolbar.addWidget(self.pause_check)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.editor = VFCodeEditor()
        self.editor.setFont(QtGui.QFont("Consolas", 11))
        self.editor.setStyleSheet(
            "QPlainTextEdit { background: #1E1E1E; color: #D4D4D4; border: 1px solid #3C3C3C; }"
        )
        self.editor._highlighter = VFCodeHighlighter(self.editor.document())
        self.editor.setPlainText(initial_script.strip() or DEFAULT_2D_SCRIPT)
        left_layout.addWidget(self.editor, 1)
        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.graphics_scene = QtWidgets.QGraphicsScene(self)
        self.graphics_view = QtWidgets.QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.graphics_view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.graphics_view.setMinimumHeight(420)
        right_layout.addWidget(self.graphics_view, 1)

        lower = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(500)
        self.console.setStyleSheet(
            "QPlainTextEdit { background: #111111; color: #D7D7D7; border: 1px solid #333333; }"
        )
        self.reference = QtWidgets.QPlainTextEdit(SCRIPT_HELP)
        self.reference.setReadOnly(True)
        self.reference.setStyleSheet(
            "QPlainTextEdit { background: #171A1F; color: #B9D6FF; border: 1px solid #333333; }"
        )
        lower.addWidget(self.console)
        lower.addWidget(self.reference)
        lower.setSizes([420, 320])
        right_layout.addWidget(lower, 0)
        splitter.addWidget(right)
        splitter.setSizes([520, 700])

        self.status = QtWidgets.QLabel("Ready")
        self.status.setStyleSheet("QLabel { color: #8BC34A; font-weight: 700; }")
        root.addWidget(self.status)

        self.live_timer = QtCore.QTimer(self)
        self.live_timer.setSingleShot(True)
        self.live_timer.setInterval(300)
        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.setInterval(16)

        self.run_btn.clicked.connect(self.run_script)
        self.template_btn.clicked.connect(lambda: self.editor.setPlainText(DEFAULT_2D_SCRIPT))
        self.editor.textChanged.connect(self.schedule_live_run)
        self.live_timer.timeout.connect(self.run_script)
        self.frame_timer.timeout.connect(self.tick)

        self.run_script()
        self.frame_timer.start()

    def schedule_live_run(self):
        if self.live_check.isChecked():
            self.live_timer.start()

    def run_script(self):
        text = self.editor.toPlainText()
        if text == self._last_script and self.engine.scene.layers:
            return
        result = parse_2d_script(text)
        if result.ok:
            self.engine.load_scene(result.scene)
            self._last_script = text
            self.console.setPlainText("\n".join(result.actions[-80:]))
            self.status.setStyleSheet("QLabel { color: #8BC34A; font-weight: 700; }")
            self.status.setText(f"Running scene: {result.scene.name}")
            self.paint_scene()
            return

        messages = [f"Line {item.line}: {item.message}" for item in result.diagnostics]
        self.console.setPlainText("\n".join(messages))
        self.status.setStyleSheet("QLabel { color: #EF5350; font-weight: 700; }")
        self.status.setText(f"{len(messages)} script error(s)")

    def tick(self):
        if self.pause_check.isChecked():
            return
        scene = self.engine.scene
        for layer in scene.layers:
            for sprite in layer.sprites:
                bounds = sprite.bounds()
                if sprite.velocity.x and (bounds.right > scene.camera.viewport.x - 20 or bounds.left < 20):
                    sprite.velocity = Vec2(-sprite.velocity.x, sprite.velocity.y)
                if sprite.velocity.y and (bounds.bottom > scene.camera.viewport.y - 20 or bounds.top < 20):
                    sprite.velocity = Vec2(sprite.velocity.x, -sprite.velocity.y)
        self.engine.update(self.frame_timer.interval() / 1000.0)
        self.paint_scene()

    def paint_scene(self):
        scene = self.engine.scene
        viewport = scene.camera.viewport
        self.graphics_scene.setSceneRect(0, 0, viewport.x, viewport.y)
        self.graphics_scene.clear()
        self.graphics_scene.setBackgroundBrush(QtGui.QColor("#0F1218"))
        collisions = scene.collision_pairs()
        colliding = {sprite.name for pair in collisions for sprite in pair}

        for command in self.engine.render_plan():
            rect = command.bounds
            color = command.metadata.get("color", "#43B5FF")
            if command.sprite in colliding:
                color = "#FFD166"
            pen = QtGui.QPen(QtGui.QColor("#F5F7FA"), 1 if command.sprite in colliding else 0)
            brush = QtGui.QBrush(QtGui.QColor(str(color)))
            item = self.graphics_scene.addRect(rect.x, rect.y, rect.width, rect.height, pen, brush)
            item.setZValue(command.z_index)
            label = self.graphics_scene.addText(command.sprite)
            label.setDefaultTextColor(QtGui.QColor("#FFFFFF"))
            label.setPos(rect.x + 6, rect.y + 5)
            label.setZValue(command.z_index + 0.1)

        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.status.setText(
            f"Running scene: {scene.name} | sprites: {len(self.engine.render_plan())} | "
            f"collisions: {len(collisions)} | time: {scene.time:.2f}s"
        )
