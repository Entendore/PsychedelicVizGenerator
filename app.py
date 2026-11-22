import sys
import numpy as np
import random
import sounddevice as sd
from collections import deque

# GUI Imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QPushButton, QComboBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

# Performance Imports
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found. Expert GPU visualizers will be disabled.")

# --- Expert Algorithm Classes (Music) ---

class LSystemRhythm:
    def __init__(self, axiom="0", rules={"0": "1[0]0", "1": "11"}, iterations=4):
        self.pattern = self._generate(axiom, rules, iterations)
    
    def _generate(self, axiom, rules, iterations):
        current = axiom
        for _ in range(iterations):
            next_str = ""
            for char in current:
                next_str += rules.get(char, char)
            current = next_str
        return [1 if c == '1' else 0 for c in current]

class MarkovMelody:
    def __init__(self, scale):
        self.scale = scale
        self.current_state = 0
        size = len(scale)
        self.matrix = self._build_matrix(size)
        
    def _build_matrix(self, size):
        mat = np.random.random((size, size)) + 0.1
        for i in range(size):
            if i > 0: mat[i][i-1] += 2.0
            if i < size-1: mat[i][i+1] += 2.0
            mat[i][0] += 1.0
            mat[i][4 % size] += 1.5
            if i == (size-1): mat[i][0] += 10.0
        row_sums = mat.sum(axis=1)
        return mat / row_sums[:, np.newaxis]

    def next_note(self):
        probs = self.matrix[self.current_state]
        next_state = np.random.choice(len(probs), p=probs)
        self.current_state = next_state
        return self.scale[next_state]

# --- Audio Engine ---

class AudioEngine(QThread):
    data_ready = Signal(float)

    def __init__(self):
        super().__init__()
        self.sample_rate = 44100
        self.running = False
        self.volume = 0.5
        
        # Composition
        self.steps_rhythm = 16
        self.steps_melody = 19 
        self.rhythm_gen = LSystemRhythm(axiom="0", rules={"0": "01", "1": "10"}, iterations=6)
        self.rhythm_pattern = self.rhythm_gen.pattern[:self.steps_rhythm]
        if sum(self.rhythm_pattern) < 3:
            self.rhythm_pattern = [1 if i % 4 == 0 else 0 for i in range(self.steps_rhythm)]

        self.scale = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25]
        self.melody_gen = MarkovMelody(self.scale)
        self.melody_pattern = [self.melody_gen.next_note() for _ in range(self.steps_melody)]
        
        # Synthesis State
        self.tick = 0
        self.ticks_per_step = 2200
        self.phase_kick = 0.0; self.phase_hat = 0.0; self.phase_bass = 0.0
        self.phase_lead = 0.0; self.phase_fb = 0.0
        self.filter_delay = np.zeros(4)
        self.delay_buffer = np.zeros(int(self.sample_rate * 0.5))
        self.delay_idx = 0

        if NUMBA_AVAILABLE:
            print("Compiling Expert Numba Kernels...")
            # FIX: Corrected argument count and types to match _numba_render_block definition
            # 1. outdata (array)
            # 2. frames (int)
            # 3. sr (int)
            # 4-8: phases (floats)
            # 9-10: freqs (floats)
            # 11-12: triggers (bool/int)
            # 13: delay_buf (array)
            # 14: delay_idx (int)
            # 15: filt_del (array)
            # 16: time_val (float)
            dummy_out = np.zeros(512, dtype=np.float32)
            dummy_delay = np.zeros(10, dtype=np.float32)
            dummy_filt = np.zeros(4, dtype=np.float32)
            
            self._numba_render_block(
                dummy_out, 512, 44100, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                55.0, 440.0, 
                False, False, 
                dummy_delay, 0, dummy_filt, 0.0
            )

    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_render_block(outdata, frames, sr, 
                             p_kick, p_hat, p_bass, p_lead, p_fb,
                             f_bass, f_lead, 
                             kick_trig, hat_trig,
                             delay_buf, delay_idx, filt_del, time_val):
        d1, d2, d3, d4 = filt_del
        for i in range(frames):
            # 1. KICK
            kick_out = 0.0
            if kick_trig:
                env = 1.0 - (i / frames)
                freq = 50.0 * (1.0 + env * 5.0)
                p_kick += (2 * np.pi * freq) / sr
                kick_out = np.sin(p_kick) * env
            
            # 2. HI-HAT
            hat_out = 0.0
            if hat_trig:
                noise = np.random.rand() * 2.0 - 1.0
                hat_out = noise * (1.0 - (i/frames)*2.0) * 0.2

            # 3. BASS
            p_bass += (2 * np.pi * f_bass) / sr
            bass_out = np.sin(p_bass) * 0.4
            
            # 4. LEAD: Feedback FM
            fb_amt = 1.0 + 0.5 * np.sin(time_val * 0.0001)
            modulator = np.sin(p_fb)
            p_lead += (2 * np.pi * f_lead) / sr
            lead_raw = np.sin(p_lead + fb_amt * modulator)
            p_fb += (2 * np.pi * (f_lead * 0.5)) / sr 
            lead_out = lead_raw * 0.3
            
            # 5. FILTER
            sig = (kick_out + hat_out + bass_out + lead_out)
            cutoff = 0.2
            d1 = d1 + cutoff * (sig - d1 + 0.5 * (d1 - d2))
            d2 = d2 + cutoff * (d1 - d2)
            d3 = d3 + cutoff * (d2 - d3)
            d4 = d4 + cutoff * (d3 - d4)
            filtered_sig = d4
            
            # 6. DELAY
            d_len = len(delay_buf)
            r_idx = (delay_idx - int(sr * 0.3)) % d_len
            delayed = delay_buf[r_idx] * 0.5
            delay_buf[delay_idx] = filtered_sig + delayed * 0.6
            delay_idx = (delay_idx + 1) % d_len
            
            outdata[i] = (filtered_sig * 0.8 + delayed * 0.4)
            
        return p_kick, p_hat, p_bass, p_lead, p_fb, delay_idx, d1, d2, d3, d4

    def callback(self, outdata, frames, time_info, status):
        if not self.running:
            outdata.fill(0)
            return

        self.tick += frames
        if self.tick >= self.ticks_per_step:
            self.tick = 0
            if not hasattr(self, 'r_idx'): self.r_idx = 0
            if not hasattr(self, 'm_idx'): self.m_idx = 0
            
            self.kick_trig = self.rhythm_pattern[self.r_idx % self.steps_rhythm] == 1
            self.hat_trig = (self.r_idx % 2 == 1) 
            
            if self.r_idx % 2 == 0: 
                 self.current_lead_freq = self.melody_pattern[self.m_idx % self.steps_melody]
                 self.m_idx += 1
            self.r_idx += 1
            
            if random.random() < 0.01:
                mutate_idx = random.randint(0, self.steps_melody - 1)
                self.melody_pattern[mutate_idx] = self.melody_gen.next_note()
        else:
            self.kick_trig = False
            self.hat_trig = False

        mono_out = np.zeros(frames, dtype=np.float32)
        
        (self.phase_kick, self.phase_hat, self.phase_bass, self.phase_lead, self.phase_fb,
         self.delay_idx, 
         self.filter_delay[0], self.filter_delay[1], self.filter_delay[2], self.filter_delay[3]) = \
            self._numba_render_block(
                mono_out, frames, self.sample_rate,
                self.phase_kick, self.phase_hat, self.phase_bass, self.phase_lead, self.phase_fb,
                55.0, 
                self.current_lead_freq if hasattr(self, 'current_lead_freq') else 440.0,
                self.kick_trig, self.hat_trig,
                self.delay_buffer, self.delay_idx, self.filter_delay, self.tick
            )
        
        outdata[:, 0] = mono_out * self.volume
        outdata[:, 1] = mono_out * self.volume
        self.data_ready.emit(np.max(np.abs(mono_out)))

    def run(self):
        self.running = True
        self.current_lead_freq = self.scale[0]
        with sd.OutputStream(callback=self.callback, channels=2, samplerate=self.sample_rate, dtype='float32'):
            self.exec_()

    def stop(self):
        self.running = False
        self.quit()

# --- Expert Visual Engines ---

class VisualEngine:
    def render(self, t, audio_level): raise NotImplementedError

class ReactionDiffusionEngine(VisualEngine):
    """
    Simulates Gray-Scott Reaction-Diffusion equations on the GPU.
    """
    def __init__(self, width, height):
        if not CUPY_AVAILABLE: raise ImportError("CuPy required")
        self.w, self.h = width, height
        self.dtype = cp.float32
        
        self.du = 0.2097
        self.dv = 0.105
        self.f = 0.054
        self.k = 0.062
        
        self.U = cp.ones((self.h, self.w), dtype=self.dtype)
        self.V = cp.zeros((self.h, self.w), dtype=self.dtype)
        
        r = 20
        cx, cy = self.w // 2, self.h // 2
        self.V[cy-r:cy+r, cx-r:cx+r] = 1.0
        
        for _ in range(10):
            rx, ry = random.randint(0, self.w-20), random.randint(0, self.h-20)
            self.V[ry:ry+20, rx:rx+20] = 1.0

        self.kernel = cp.array([[0, 1, 0], 
                                [1, -4, 1], 
                                [0, 1, 0]], dtype=self.dtype)

    def render(self, t, audio_level):
        current_f = self.f + audio_level * 0.02
        
        for _ in range(10): 
            Lu = self._convolve2d(self.U)
            Lv = self._convolve2d(self.V)
            
            uvv = self.U * self.V * self.V
            dU = self.du * Lu - uvv + current_f * (1.0 - self.U)
            dV = self.dv * Lv + uvv - (self.f + self.k) * self.V
            
            self.U += dU
            self.V += dV
            
            self.U = cp.clip(self.U, 0.0, 1.0)
            self.V = cp.clip(self.V, 0.0, 1.0)
        
        r = (self.V * 255).astype(cp.uint8)
        g = (self.V * (1 + 0.5*np.sin(t)) * 200).astype(cp.uint8)
        b = ((1 - self.U) * 200).astype(cp.uint8)
        
        img = cp.stack([r, g, b], axis=-1)
        return cp.asnumpy(img)

    def _convolve2d(self, arr):
        return (cp.roll(arr, 1, axis=0) + cp.roll(arr, -1, axis=0) + 
                cp.roll(arr, 1, axis=1) + cp.roll(arr, -1, axis=1) - 4 * arr)

class OrbitTrapFractal(VisualEngine):
    def __init__(self, width, height):
        if not CUPY_AVAILABLE: raise ImportError("CuPy required")
        self.w, self.h = width, height
        self.gpu_out = cp.zeros((height, width, 3), dtype=cp.uint8)
        
        x = cp.linspace(-1.5, 1.5, width).astype(cp.float32)
        y = cp.linspace(-1.5, 1.5, height).astype(cp.float32)
        self.X, self.Y = cp.meshgrid(x, y)

    def render(self, t, audio_level):
        c_re = -0.8 + 0.2 * cp.sin(t * 0.1)
        c_im = 0.156 + 0.1 * cp.cos(t * 0.2)
        
        zoom = 1.0 + audio_level * 0.5
        
        Zx = self.X * zoom
        Zy = self.Y * zoom
        
        dist = cp.full((self.h, self.w), 1e10, dtype=cp.float32)
        
        for i in range(100):
            Zx_new = Zx * Zx - Zy * Zy + c_re
            Zy_new = 2 * Zx * Zy + c_im
            Zx, Zy = Zx_new, Zy_new
            
            d_cross = cp.minimum(cp.abs(Zx), cp.abs(Zy))
            dist = cp.minimum(dist, d_cross)
        
        inv_dist = 1.0 / (dist + 0.001) 
        val = cp.tanh(inv_dist * 0.5)
        
        r = (val * cp.sin(t) * 255).astype(cp.uint8)
        g = (val * 200).astype(cp.uint8)
        b = (val * (1 - cp.sin(t)) * 255).astype(cp.uint8)
        
        self.gpu_out[:,:,0] = r
        self.gpu_out[:,:,1] = g
        self.gpu_out[:,:,2] = b
        
        return cp.asnumpy(self.gpu_out)

class PlasmaEngineNumba(VisualEngine):
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.output = np.zeros((h, w, 3), dtype=np.uint8)
        if NUMBA_AVAILABLE:
            self.render(0,0)

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _render_kernel(img, w, h, t, audio_level):
        for y in prange(h):
            for x in range(w):
                v = np.sin(x / 20.0 + t)
                v += np.sin((y/ 20.0 + t) * 0.5)
                v += np.sin((x + y) / 20.0 + t)
                v += np.sin(np.sqrt(x*x + y*y) / 20.0 + t)
                v = v * (1.0 + audio_level * 3.0)
                img[y, x, 0] = int(np.sin(v * 3.14) * 127 + 128)
                img[y, x, 1] = int(np.sin(v * 3.14 + 2.0) * 127 + 128)
                img[y, x, 2] = int(np.sin(v * 3.14 + 4.0) * 127 + 128)

    def render(self, t, audio_level):
        self._render_kernel(self.output, self.w, self.h, t, audio_level)
        return self.output

# --- Main Window ---

class PsychedelicWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Psychedelic Machine: Expert Algorithms Edition")
        self.setGeometry(100, 100, 800, 600)
        
        self.time_val = 0.0
        self.audio_level = 0.0
        self.active_engine = None
        
        self.setup_ui()
        
        self.audio_thread = AudioEngine()
        self.audio_thread.data_ready.connect(self.update_audio_level)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(16) 

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.video_label = QLabel("Press Start")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-weight: bold;")
        layout.addWidget(self.video_label)
        
        ctrl_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("Start Engine")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.btn_play)
        
        ctrl_layout.addWidget(QLabel("Visual Engine:"))
        self.combo_engine = QComboBox()
        self.combo_engine.addItem("Reaction-Diffusion (GPU)", "rd_gpu")
        self.combo_engine.addItem("Orbit Trap Fractal (GPU)", "fractal_gpu")
        self.combo_engine.addItem("Plasma (CPU)", "plasma_cpu")
        self.combo_engine.currentIndexChanged.connect(self.change_engine)
        ctrl_layout.addWidget(self.combo_engine)

        ctrl_layout.addWidget(QLabel("Volume:"))
        self.slider_vol = QSlider(Qt.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(50)
        self.slider_vol.valueChanged.connect(lambda v: setattr(self.audio_thread, 'volume', v/100.0))
        ctrl_layout.addWidget(self.slider_vol)
        
        layout.addLayout(ctrl_layout)

    def update_audio_level(self, level): self.audio_level = max(level, self.audio_level * 0.8)

    def toggle_play(self, checked):
        if checked:
            self.btn_play.setText("Stop Engine")
            self.change_engine(self.combo_engine.currentIndex())
            self.audio_thread.start()
            self.timer.start()
        else:
            self.btn_play.setText("Start Engine")
            self.timer.stop()
            self.audio_thread.stop()

    def change_engine(self, index):
        data = self.combo_engine.itemData(index)
        w, h = self.width(), self.height()
        try:
            if data == "rd_gpu":
                if CUPY_AVAILABLE: self.active_engine = ReactionDiffusionEngine(w, h)
                else: self.active_engine = None
            elif data == "fractal_gpu":
                if CUPY_AVAILABLE: self.active_engine = OrbitTrapFractal(w, h)
                else: self.active_engine = None
            elif data == "plasma_cpu":
                self.active_engine = PlasmaEngineNumba(w, h)
        except Exception as e:
            print(f"Engine Init Error: {e}")

    def update_frame(self):
        if self.active_engine is None: return
        self.time_val += 0.05
        try:
            frame = self.active_engine.render(self.time_val, self.audio_level)
            if frame is not None:
                h, w, ch = frame.shape
                q_img = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.video_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                ))
        except Exception as e: print(f"Render Error: {e}")
        self.audio_level *= 0.9

    def closeEvent(self, event):
        self.audio_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PsychedelicWindow()
    window.show()
    sys.exit(app.exec())