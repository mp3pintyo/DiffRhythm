# DiffRhythm - AI dallamgeneráló rendszer

<div align="center">
  <p>
    <a href="README.md">Magyar</a> |
    <a href="README_EN.md">English</a>
  </p>
</div>

DiffRhythm (谛韵) egy mesterséges intelligencia alapú dallamgeneráló rendszer, amely időzített dalszövegekből (LRC formátum) és referencia hanganyagból képes teljes dalokat létrehozni.

## Főbb jellemzők

- **Teljes dalok generálása** időzített dalszövegek (LRC formátum) alapján
- **Hangstílus meghatározása** referencia hanganyag segítségével
- **LLM-vezérelt dalszöveg időzítés** generálása
- **Alacsony memóriaigényű üzemmód** korlátozott GPU erőforrásokkal rendelkező rendszerekhez
- **Többnyelvű támogatás** (angol, kínai és egyéb nyelvek)
- **Rugalmas kimeneti formátumok** (WAV, MP3, OGG)

## Telepítés

### Előfeltételek

- Python 3.8 vagy újabb
- CUDA 11.7 vagy újabb (GPU használatához)
- FFmpeg (hang kódoláshoz)

### RunPod telepítés

A DiffRhythm rendszer felhőben is futtatható [RunPod](https://runpod.io?ref=2pdhmpu1) használatával:

**Ajánlott konfiguráció:**
- **GPU:** Nvidia A40 48 GByte VRAM
- **Template:** RunPod PyTorch 2.4.0

**Telepítési lépések RunPod környezetben:**
```bash
git clone https://github.com/mp3pintyo/DiffRhythm.git
cd DiffRhythm
pip install -r requirements.txt
pip install openai spaces
apt-get update && apt-get install -y espeak
```

## Használat

### Webes felület indítása

```bash
# A modellek automatikusan letöltődnek az első futtatáskor
python app.py
```

### Főbb funkciók

1. **Zenegenerálás** fül:
   - Illessze be az időzített dalszöveget LRC formátumban
   - Töltsön fel egy referencia hanganyagot (legalább 10 másodperc)
   - Állítsa be a generálási paramétereket (lépések száma, kimeneti formátum)
   - Kattintson a "Submit" gombra a dal generálásához

2. **LLM-vezérelt LRC generálás** fül:
   - **Téma alapú generálás**: Adjon meg egy témát és stílusjelzőket
   - **Időzítés hozzáadása**: Adjon meg egyszerű dalszöveget időzítés nélkül, és a rendszer automatikusan hozzáadja a megfelelő időzítést

## Hibaelhárítás

- **Hiányzó modellfájlok**: Az első futtatáskor automatikusan letöltődnek
- **Espeak nyelvi hibák**: A program automatikusan angol nyelvű feldolgozásra vált nem támogatott nyelvek esetén
- **Rövid hanganyagok**: A rendszer automatikusan ismétli a rövid hanganyagokat a minimális 10 másodperces hossz eléréséhez

## Technikai részletek

- **Architektúra**: Diffúziós modell zenei generáláshoz
- **Audio kódolás**: VAE alapú audio dekóder
- **Nyelvfeldolgozás**: Grapheme-to-Phoneme konverzió különböző nyelveken
- **Stílus embedding**: Zenei stílusok reprezentálása MuLan embeddinggel

## További fejlesztési irányok

- Hosszabb dalok generálása (jelenleg maximum 95 másodperc)
- Több referencia hanganyag kombinálása
- Valós idejű generálás
- Finomhangolási lehetőségek a generált zenében

## Licensz

Ez a projekt az [Apache License 2.0](LICENSE) alatt áll, módosítva az eredeti [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) projektből.

## Elismerések

- Alapja a [DiffRhythm: Modeling and Generating Musical Rhythms with Conditional Latent Diffusion](https://arxiv.org/abs/2503.01183) kutatási tanulmány
- Az eredeti implementáció forrása: [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)
- MuQ-MuLan modellek: [OpenMuQ](https://github.com/OpenMuQ/MuQ)
