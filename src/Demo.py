import os, sys, random
import pickle
import tkinter as tk
import numpy as np

from tkinter import font as tkfont
from PIL import Image, ImageTk, ImageGrab, ImageOps

from src.CNN.network import Network
from src.CNN.conv import Conv2D
from src.CNN.pooling import MaxPool2D
from src.CNN.pooling import Flatten
from src.CNN.linear import Dense, InputLayer
from src.CNN.functional import ActivationFunction, LossFunction

# =====================================================================
#  CIFAR-10 Random Image Predictor UI
# =====================================================================
class CifarApp:
  BG_DARK      = "#0f0f1a"
  PANEL_BG     = "#1a1a2e"
  CANVAS_BG    = "#111122"
  ACCENT       = "#f59e0b"   # amber
  ACCENT_HOVER = "#fbbf24"
  TEXT_PRIMARY = "#e2e8f0"
  TEXT_MUTED   = "#94a3b8"
  SUCCESS      = "#10b981"   # emerald
  BAR_BG       = "#2a2a40"
  BAR_FILL     = "#f59e0b"

  PREVIEW_PX = 256  # Display scale for the 32x32 image

  def __init__(self, root: tk.Tk, model: Network, x_test: np.ndarray, y_test: np.ndarray, accuracy: float = 0.0):
    self.root     = root
    self.model    = model
    self.x_test   = x_test
    self.y_test   = y_test
    self.accuracy = accuracy
    
    self.classes = [
      "airplane", "automobile", "bird", "cat", "deer",
      "dog", "frog", "horse", "ship", "truck"
    ]

    root.title("CNN — CIFAR-10 Recogniser")
    root.configure(bg=self.BG_DARK)
    root.resizable(False, False)

    self.font_title = tkfont.Font(family="Segoe UI", size=16, weight="bold")
    self.font_class = tkfont.Font(family="Consolas", size=24, weight="bold")
    self.font_label = tkfont.Font(family="Segoe UI", size=11)
    self.font_btn   = tkfont.Font(family="Segoe UI", size=11, weight="bold")
    self.font_small = tkfont.Font(family="Segoe UI", size=9)
    self.font_bar   = tkfont.Font(family="Consolas", size=9)

    self._build_ui()
    self._next_image() # load first random image

  def _build_ui(self):
    # Title bar
    tf = tk.Frame(self.root, bg=self.BG_DARK, pady=12)
    tf.pack(fill="x")
    tk.Label(tf,
             text="✦  CNN CIFAR-10 Recogniser",
             font=self.font_title,
             fg=self.ACCENT,
             bg=self.BG_DARK).pack()
    tk.Label(tf,
             text=f"Conv + BN + Dropout  ·  Accuracy: {self.accuracy:.2f}%",
             font=self.font_small,
             fg=self.TEXT_MUTED,
             bg=self.BG_DARK).pack()

    # Main row
    main = tk.Frame(self.root, bg=self.BG_DARK, padx=16, pady=4)
    main.pack()

    # Image Preview (Left)
    pf = tk.Frame(main, bg=self.BG_DARK)
    pf.pack(side="left", padx=(0, 16))
    
    pc_border = tk.Frame(pf, bg=self.ACCENT, padx=2, pady=2)
    pc_border.pack()
    self.preview_canvas = tk.Canvas(pc_border,
                                    width=self.PREVIEW_PX,
                                    height=self.PREVIEW_PX,
                                    bg="black",
                                    highlightthickness=0)
    self.preview_canvas.pack()
    
    self.true_label = tk.Label(pf,
             text="True Label: ",
             font=self.font_small,
             fg=self.TEXT_MUTED,
             bg=self.BG_DARK)
    self.true_label.pack(pady=(4, 0))

    # Right panel
    right = tk.Frame(main, bg=self.BG_DARK, padx=16)
    right.pack(side="left", fill="y")

    # Predicted class card
    dc = tk.Frame(right, bg=self.PANEL_BG, padx=24, pady=8)
    dc.pack(pady=(0, 8))
    tk.Label(dc,
             text="Prediction",
             font=self.font_label,
             fg=self.TEXT_MUTED,
             bg=self.PANEL_BG).pack()
    self.class_label = tk.Label(dc,
                                text="—",
                                font=self.font_class,
                                fg=self.SUCCESS,
                                bg=self.PANEL_BG,
                                width=12)
    self.class_label.pack()

    self.conf_label = tk.Label(right,
                               text="",
                               font=self.font_small,
                               fg=self.TEXT_MUTED,
                               bg=self.BG_DARK)
    self.conf_label.pack(pady=(4, 6))

    # Probability bars (one per class)
    bars = tk.Frame(right, bg=self.BG_DARK)
    bars.pack(fill="x", pady=(0, 10))
    self.bar_canvases = []
    self.bar_labels   = []
    BAR_W, BAR_H = 150, 12
    self.BAR_W, self.BAR_H = BAR_W, BAR_H

    for i, cls_name in enumerate(self.classes):
      row = tk.Frame(bars, bg=self.BG_DARK)
      row.pack(fill="x", pady=1)
      tk.Label(row,
               text=cls_name[:12],
               font=self.font_bar,
               fg=self.TEXT_MUTED,
               bg=self.BG_DARK,
               width=11,
               anchor="e").pack(side="left")
      bar = tk.Canvas(row,
                      width=BAR_W,
                      height=BAR_H,
                      bg=self.BAR_BG,
                      highlightthickness=0)
      bar.pack(side="left", padx=(4, 4))
      self.bar_canvases.append(bar)
      pct = tk.Label(row,
                     text="",
                     font=self.font_bar,
                     fg=self.TEXT_MUTED,
                     bg=self.BG_DARK,
                     width=6,
                     anchor="w")
      pct.pack(side="left")
      self.bar_labels.append(pct)

    # Buttons
    bf = tk.Frame(right, bg=self.BG_DARK)
    bf.pack(pady=(4, 0))
    tk.Button(bf,
              text="⚡ Predict",
              font=self.font_btn,
              bg=self.ACCENT,
              fg="white",
              activebackground=self.ACCENT_HOVER,
              activeforeground="white",
              relief="flat",
              padx=18, pady=6,
              cursor="hand2",
              command=self._on_predict).pack(side="left", padx=4)
    tk.Button(bf,
              text="⟳ Next Image",
              font=self.font_btn,
              bg="#334155",
              fg=self.TEXT_PRIMARY,
              activebackground="#475569",
              activeforeground="white",
              relief="flat",
              padx=18, pady=6,
              cursor="hand2",
              command=self._next_image).pack(side="left", padx=4)
    tk.Button(bf,
              text="📋 Paste Image",
              font=self.font_btn,
              bg="#334155",
              fg=self.TEXT_PRIMARY,
              activebackground="#475569",
              activeforeground="white",
              relief="flat",
              padx=18, pady=6,
              cursor="hand2",
              command=self._on_paste).pack(side="left", padx=4)

    tk.Frame(self.root, bg=self.BG_DARK, height=14).pack()

  def _next_image(self):
    # Pick a random image from the test set
    idx = random.randint(0, len(self.x_test) - 1)
    self.current_img_data = self.x_test[idx]
    
    true_class_idx = np.argmax(self.y_test[idx])
    self.true_label.config(text=f"True Label: {self.classes[true_class_idx]}")

    # Convert model format (3, 32, 32) float [0, 1] to PIL Image format (32, 32, 3) uint8 [0, 255]
    img_array = (self.current_img_data.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img_array, mode='RGB')
    
    # Scale up for display (use NEAREST to show the pixels clearly)
    preview_img = pil_img.resize((self.PREVIEW_PX, self.PREVIEW_PX), Image.NEAREST)
    self.tk_preview = ImageTk.PhotoImage(preview_img)
    self.preview_canvas.create_image(self.PREVIEW_PX // 2, self.PREVIEW_PX // 2,
                                     image=self.tk_preview)
                                     
    # Clear predictions
    self.class_label.config(text="—")
    self.conf_label.config(text="")
    for bar in self.bar_canvases:
      bar.delete("all")
    for lbl in self.bar_labels:
      lbl.config(text="")

  def _on_paste(self):
    try:
      img = ImageGrab.grabclipboard()
    except Exception as e:
      print(f"Failed to grab clipboard: {e}")
      return
      
    if img is None:
      print("No image found in clipboard.")
      return
      
    # On Windows, sometimes file paths are copied to clipboard instead of image pixels
    if isinstance(img, list) and len(img) > 0:
      try:
        img = Image.open(img[0])
      except Exception as e:
        print(f"Could not open image from file path: {e}")
        return

    if not isinstance(img, Image.Image):
      print("Clipboard content is not an image.")
      return

    # Convert to RGB just in case it's RGBA
    img = img.convert('RGB')
    
    # Crop and resize to 32x32 to fit the model's expected input
    # ImageOps.fit crops the center of the image to match the aspect ratio
    # and then resizes it.
    try:
        resample_method = Image.Resampling.BICUBIC
    except AttributeError:
        resample_method = Image.BICUBIC
        
    img_32 = ImageOps.fit(img, (32, 32), method=resample_method)
    
    # Update current image data (3, 32, 32) float [0, 1]
    img_array = np.array(img_32).astype(np.float32) / 255.0
    self.current_img_data = img_array.transpose(2, 0, 1)
    
    self.true_label.config(text="True Label: [Pasted Image]")

    # Scale up for display (use NEAREST to show the pixels clearly)
    preview_img = img_32.resize((self.PREVIEW_PX, self.PREVIEW_PX), Image.NEAREST)
    self.tk_preview = ImageTk.PhotoImage(preview_img)
    self.preview_canvas.create_image(self.PREVIEW_PX // 2, self.PREVIEW_PX // 2,
                                     image=self.tk_preview)
                                     
    # Clear predictions
    self.class_label.config(text="—")
    self.conf_label.config(text="")
    for bar in self.bar_canvases:
      bar.delete("all")
    for lbl in self.bar_labels:
      lbl.config(text="")

  def _on_predict(self):
    arr = self.current_img_data.reshape(1, 3, 32, 32)
    probs = self.model.predict(arr)[0]
    best  = int(np.argmax(probs))

    self.class_label.config(text=self.classes[best])
    self.conf_label.config(text=f"Confidence: {float(probs[best]) * 100:.1f}%")
    self._draw_bars(probs, best)

  def _draw_bars(self, probs, best):
    for i, p in enumerate(probs):
      bar = self.bar_canvases[i]
      bar.delete("all")
      fw = max(1, int(p * self.BAR_W))
      bar.create_rectangle(0, 0, fw, self.BAR_H,
                           fill=self.SUCCESS if i == best else self.BAR_FILL,
                           outline="")
      self.bar_labels[i].config(text=f"{p * 100:5.1f}%")


# =====================================================================
#  Main — requires pre-trained weights (run training/cnn_train_cifar10.py first)
# =====================================================================
if __name__ == "__main__":
  WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "cnn_cifar10_weights.npz")

  if not os.path.exists(WEIGHTS_FILE):
    print(f"[ERROR] Weights not found: {WEIGHTS_FILE}")
    print("  -> Run  training/cnn_train_cifar10.py  first to generate weights.")
    sys.exit(1)

  # Load the test set to display random images
  def unpickle(file):
      with open(file, 'rb') as fo:
          dict = pickle.load(fo, encoding='bytes')
      return dict
      
  data_dir = os.path.join(os.path.dirname(__file__), "cifar-10-batches-py")
  test_batch = unpickle(os.path.join(data_dir, "test_batch"))
  x_test_raw = test_batch[b'data']
  y_test_lbl = np.array(test_batch[b'labels'])
  
  x_test = (x_test_raw.reshape(-1, 3, 32, 32) / 255.0).astype(np.float32)
  
  def onehot(labels, n=10):
    m = np.zeros((len(labels), n), dtype=np.float32)
    m[np.arange(len(labels)), labels] = 1
    return m
  y_test = onehot(y_test_lbl)

  act = ActivationFunction
  layers = [
    InputLayer(None, input_shape=(-1, 3, 32, 32)),
    Conv2D(32, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
    Conv2D(32, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
    MaxPool2D(2, 2),
    Conv2D(64, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
    Conv2D(64, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(512, act_func=act.ReLU, use_dropout=True, drop_rate=0.5),
    Dense(10,  act_func=act.softmax, use_dropout=False, drop_rate=0.0)
  ]

  model = Network(layers=layers, loss_func=LossFunction.cc_loss, learning_rate=0.05)
  model.training = False
  accuracy = model.load_weights(WEIGHTS_FILE) or 0.0

  root = tk.Tk()
  CifarApp(root, model, x_test, y_test, accuracy)
  root.mainloop()
