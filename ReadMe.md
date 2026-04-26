# Medical Imaging: k-Space Acquisition & Frequency-Domain Filtering

A Python toolkit for simulating MRI k-space acquisition patterns and applying frequency-domain filters to medical images. Built as part of a graduate-level medical imaging course to explore how undersampling strategies and noise filters affect image reconstruction quality.

---

## Background

In MRI, the scanner collects data in **k-space** — the frequency-domain representation of the image. The pattern in which k-space is sampled directly determines scan time and reconstructed image quality. This project simulates that process computationally:

1. Load a grayscale medical image (brain, cardiac, etc.)
2. Transform it to the frequency domain via the 2D DFT
3. Apply a k-space acquisition mask or frequency filter
4. Reconstruct the image via inverse DFT
5. Observe how different sampling strategies and filters affect the result

---

## Project Structure

```
Medical-Imaging/
├── src/
│   ├── project/
│   │   ├── ImageSynthesisNoise.py       # Frequency-domain filter implementations
│   │   ├── SelectiveImageAcquisition.py # k-space acquisition pattern implementations
│   │   └── Utilities.py                 # DFT, image I/O, masking utilities
│   └── interface/
│       ├── ImageNoiseDriver.py          # CLI for applying noise filters
│       ├── ImageAcquisitionDriver.py    # CLI for applying acquisition patterns
│       └── images/
│           ├── brain.png
│           └── cardiac.jpg
├── test/
│   ├── test_ImageSynthesisNoise.py
│   ├── test_SelectiveImageAcquisition.py
│   └── test_Utilities.py
└── requirements.txt
```

---

## Installation

Requires Python 3.6+.

```bash
pip install -r requirements.txt
```

**Dependencies:** `opencv-python`, `numpy`

---

## Usage

Both drivers are run from `src/interface/`.

```bash
cd src/interface
```

---

### Image Noise Driver — `ImageNoiseDriver.py`

Applies a frequency-domain filter to suppress or isolate noise. The image is transformed to the DFT, the chosen filter mask is applied, and the result is reconstructed and saved.

```bash
python3 ImageNoiseDriver.py -i <image_path> -m <filter_name> [options]
```

**Arguments**

| Flag | Name | Description |
|------|------|-------------|
| `-i` | image | Path to input image |
| `-m` | mask_type | Filter function name (see below) |
| `-c` | cutoff | Cutoff frequency (radius in pixels from center) |
| `-o` | order | Filter order (Butterworth only) |
| `-t` | thickness | Ring thickness (ring filters only) |
| `-matrix` | matrix | Path to `.npy` noisy DFT matrix (alternative to image) |

**Available Filters**

| Filter | Flag Value | Parameters |
|--------|-----------|------------|
| Ideal Lowpass | `idealLowpassFilter` | `-c` |
| Ideal Highpass | `idealHighpassFilter` | `-c` |
| Gaussian Lowpass | `gaussianLowpassFilter` | `-c` |
| Gaussian Highpass | `gaussianHighpassFilter` | `-c` |
| Butterworth Lowpass | `butterworthLowpassFilter` | `-c`, `-o` |
| Butterworth Highpass | `butterworthHighpassFilter` | `-c`, `-o` |
| Ring Lowpass | `ringLowpassFilter` | `-c`, `-t` |
| Ring Highpass | `ringHighpassFilter` | `-c`, `-t` |

**Examples**

```bash
# Ideal lowpass at cutoff radius 100
python3 ImageNoiseDriver.py -i images/brain.png -m idealLowpassFilter -c 100

# Butterworth lowpass, cutoff 100, order 50 (smooth rolloff)
python3 ImageNoiseDriver.py -i images/brain.png -m butterworthLowpassFilter -c 100 -o 50

# Gaussian highpass at cutoff 30 (edge enhancement)
python3 ImageNoiseDriver.py -i images/brain.png -m gaussianHighpassFilter -c 30

# Ring lowpass isolating a frequency band (cutoff 50, thickness 10)
python3 ImageNoiseDriver.py -i images/brain.png -m ringLowpassFilter -c 50 -t 10

# Filter a pre-computed noisy DFT matrix (.npy)
python3 ImageNoiseDriver.py -matrix images/noisyimage.npy
```

---

### Image Acquisition Driver — `ImageAcquisitionDriver.py`

Simulates k-space undersampling by applying an acquisition pattern mask in the frequency domain. Each pattern mimics a real MRI trajectory (Cartesian lines, radial spokes, spiral, etc.), sampling only a subset of k-space before reconstruction.

```bash
python3 ImageAcquisitionDriver.py -i <image_path> -m <pattern_name> [options]
```

**Arguments**

| Flag | Name | Description |
|------|------|-------------|
| `-i` | image | Path to input image |
| `-m` | mask_type | Acquisition pattern name (see below) |
| `-p` | percentage | Fraction of lines to sample (Cartesian, 0–1) |
| `-r` | radius | Circle radius in pixels |
| `-maj` | maj_axis | Ellipse major axis length |
| `-min` | min_axis | Ellipse minor axis length |
| `-a` | angle | Rotation angle in degrees |
| `-w` | width | Band width in pixels |
| `-l` | length | Band length in pixels |
| `-rc` | raycount | Number of radial spokes |
| `-s` | sparsity | Spiral step size (lower = denser) |

**Available Patterns**

| Pattern | Flag Value | Parameters | Description |
|---------|-----------|------------|-------------|
| Cartesian | `cartesianPattern` | `-p` | Evenly-spaced horizontal lines; `-p 0.3` samples 30% of lines |
| Circle | `circlePattern` | `-r` | Circular region of k-space centered at DC |
| Ellipse | `ellipsePattern` | `-maj`, `-min`, `-a` | Rotatable elliptical k-space region |
| Band | `bandPattern` | `-w`, `-l`, `-a` | Rotatable rectangular band through k-space |
| Radial | `radialPattern` | `-rc` | Evenly-distributed radial spokes emanating from center |
| Spiral | `spiralPattern` | `-s` | Concentric diamond-shaped rings at intervals of `sparsity` |

**Examples**

```bash
# Cartesian: sample 30% of k-space lines
python3 ImageAcquisitionDriver.py -i images/cardiac.jpg -m cartesianPattern -p .3

# Circle: sample all frequencies within radius 100
python3 ImageAcquisitionDriver.py -i images/cardiac.jpg -m circlePattern -r 100

# Ellipse: rotated 45 degrees, major axis 200, minor axis 100
python3 ImageAcquisitionDriver.py -i images/cardiac.jpg -m ellipsePattern -maj 200 -min 100 -a 45

# Band: width 10, length 100, rotated 90 degrees
python3 ImageAcquisitionDriver.py -i images/brain.png -m bandPattern -w 10 -l 100 -a 90

# Radial: 180 evenly-spaced spokes
python3 ImageAcquisitionDriver.py -i images/brain.png -m radialPattern -rc 180

# Spiral: step size of 10 pixels
python3 ImageAcquisitionDriver.py -i images/brain.png -m spiralPattern -s 10
```

Each acquisition pattern run saves two files: the reconstructed image and the normalized mask, so you can inspect exactly what region of k-space was sampled.

---

## Core Modules

### `Utilities.py`

Low-level image processing functions used by both drivers:

| Function | Description |
|----------|-------------|
| `loadImage(path)` | Load grayscale image via OpenCV |
| `loadMatrix(path)` | Load a `.npy` matrix |
| `getDFT(image)` | Compute centered 2D FFT |
| `getImage(dft)` | Inverse FFT → uint8 image (with post-processing) |
| `applyMask(dft, mask)` | Element-wise multiply DFT by mask |
| `normalizeImage(image)` | Scale pixel values to 0–255 |
| `saveImage(path, image)` | Save image via OpenCV |

### `ImageSynthesisNoise.py`

All filter functions take `(emptymask, ...)` where `emptymask` is a `[height, width]` shape tuple. They return a float64 NumPy array of the same shape.

- **Ideal** filters use a hard cutoff — sharp transition in frequency space.
- **Gaussian** filters use a smooth exponential rolloff.
- **Butterworth** filters offer a tunable rolloff via the `order` parameter: low order = gradual, high order = approaches ideal.
- **Ring** filters isolate a specific annular band of frequencies (useful for narrowband noise removal).

### `SelectiveImageAcquisition.py`

All pattern functions take `mask_size` (`[height, width]`) and return a binary NumPy mask. Patterns are applied in the frequency domain to simulate partial k-space acquisition, modeling compressed sensing or accelerated MRI scenarios.

---

## Tests

```bash
cd test
python3 -m pytest
```

Tests cover all filter functions, acquisition patterns, and utility functions. Reference outputs in `test/resources/outputs/` are used for regression comparison.

---

## Technologies

- **Python 3.6+**
- **NumPy** — FFT, array operations
- **OpenCV** — image I/O, display, normalization
