# DiffRhythm - AI Music Generation System

<div align="center">
  <p>
    <a href="README.md">Magyar</a> |
    <a href="README_EN.md">English</a>
  </p>
</div>

![DiffRhythm Banner](https://github.com/ASLP-lab/DiffRhythm.github.io/blob/main/static/images/diffrhythem-logo-name.jpg?raw=true)

DiffRhythm is an artificial intelligence-based melody generation system capable of creating complete songs from timed lyrics (LRC format) and reference audio.

## Key Features

- **Complete song generation** from timed lyrics (LRC format)
- **Audio style determination** using reference audio samples
- **LLM-driven lyrics timing** generation
- **Low memory mode** for systems with limited GPU resources
- **Multilingual support** (English, Chinese, and other languages)
- **Flexible output formats** (WAV, MP3, OGG)

## Installation

### Prerequisites

- Python 3.8 or newer
- CUDA 11.7 or newer (for GPU usage)
- FFmpeg (for audio encoding)

### RunPod Installation

DiffRhythm can be run in the cloud using [RunPod](https://runpod.io?ref=2pdhmpu1):

**Recommended Configuration:**
- **GPU:** Nvidia A40 with 48 GByte VRAM
- **Template:** RunPod PyTorch 2.4.0

**Installation Steps in RunPod Environment:**
```bash
git clone https://github.com/mp3pintyo/DiffRhythm.git
cd DiffRhythm
pip install -r requirements.txt
pip install openai spaces
apt-get update && apt-get install -y espeak
```

## Usage

### Starting the Web Interface

```bash
# Models are automatically downloaded on the first run
python app.py
```

### Main Functions

1. **Music Generation** tab:
   - Paste timed lyrics in LRC format
   - Upload a reference audio (minimum 10 seconds)
   - Set generation parameters (number of steps, output format)
   - Click the "Submit" button to generate the song

2. **LLM Generate LRC** tab:
   - **Theme-based Generation**: Enter a theme and style tags
   - **Add Timestamps**: Enter plain lyrics without timestamps, and the system will automatically add appropriate timing

## Troubleshooting

- **Missing Model Files**: Automatically downloaded on the first run
- **Espeak Language Errors**: The program automatically switches to English processing for unsupported languages
- **Short Audio Clips**: The system automatically loops short audio clips to reach the minimum 10-second requirement

## Technical Details

- **Architecture**: Diffusion model for music generation
- **Audio Encoding**: VAE-based audio decoder
- **Language Processing**: Grapheme-to-Phoneme conversion in various languages
- **Style Embedding**: Musical style representation using MuLan embeddings

## Future Development Directions

- Generation of longer songs (currently limited to 95 seconds)
- Combining multiple reference audio samples
- Real-time generation
- Fine-tuning options for generated music

## License

This project is licensed under the [Apache License 2.0](LICENSE), modified from the original [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) project.

## Acknowledgements

- Based on the research paper [DiffRhythm: Modeling and Generating Musical Rhythms with Conditional Latent Diffusion](https://arxiv.org/abs/2503.01183)
- Original implementation source: [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)
- MuQ-MuLan models: [OpenMuQ](https://github.com/OpenMuQ/MuQ)