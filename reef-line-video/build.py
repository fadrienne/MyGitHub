#!/usr/bin/env python3
"""Beyond the Reef Line, OCL pitch video. Builds the 2:00 draft cut from assets/.

Usage:
  python3 build.py --preview     one frame per shot into out/preview/ plus a contact sheet
  python3 build.py               full render
  DRAFT=0 python3 build.py       full render with the bracketed review tags hidden

Voiceover: drop your own recordings into vo/ as scene1.wav, scene2a.wav, scene2b.wav,
scene3.wav ... scene6.wav (each starts at its scene window) and the build adds a
*_finalVO.mp4. Without them it makes a no-VO cut and a scratch guide-voice cut (pacing only).
"""
import os, sys, math, subprocess, wave, re, shutil
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda f: os.path.join(HERE, 'assets', f)
OUT = os.path.join(HERE, 'out')
W, H, FPS, DUR = 1920, 1080, 24, 120.0
DRAFT = os.environ.get('DRAFT', '1') == '1'
SR = 48000

NAVY1, NAVY2 = (8, 28, 48), (14, 74, 99)
SAND, TEAL, DIM = (242, 230, 201), (63, 193, 201), (120, 140, 155)

# ---------------------------------------------------------------- helpers
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)

def lerp(a, b, t):
    return a + (b - a) * t

@lru_cache(None)
def F(sz, w='Regular'):
    return ImageFont.truetype(f'/usr/share/fonts/opentype/inter/Inter-{w}.otf', sz)

@lru_cache(None)
def FM(sz):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', sz)

@lru_cache(None)
def navy_base():
    y = np.linspace(0, 1, H)[:, None, None]
    a = np.array(NAVY1)[None, None, :]
    b = np.array(NAVY2)[None, None, :]
    g = (a + (b - a) * y).repeat(W, axis=1).astype(np.uint8)
    return Image.fromarray(g, 'RGB')

def navy():
    return navy_base().copy()

def kb(src, region, z0, z1, p, fx=.5, fy=.5):
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    z = lerp(z0, z1, ease(p))
    w, h = rw / z, rh / z
    wx, wy = x0 + (rw - w) * fx, y0 + (rh - h) * fy
    return src.resize((W, H), Image.LANCZOS, box=(wx, wy, wx + w, wy + h))

_bg = {}
def blurbg(src, key):
    if key not in _bg:
        sw, sh = src.size
        sc = max(W / sw, H / sh)
        s = src.resize((int(sw * sc) + 1, int(sh * sc) + 1))
        l, t = (s.width - W) // 2, (s.height - H) // 2
        s = s.crop((l, t, l + W, t + H)).resize((W // 6, H // 6)).filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BICUBIC)
        _bg[key] = Image.fromarray((np.asarray(s) * 0.55).astype(np.uint8))
    return _bg[key].copy()

def fit(src, key, z0, z1, p, dx=0, dy=0):
    sw, sh = src.size
    sc = min(W / sw, H / sh)
    fw, fh = int(sw * sc), int(sh * sc)
    z = lerp(z0, z1, ease(p))
    ww, hh = sw / z, sh / z
    fg = src.resize((fw, fh), Image.LANCZOS, box=((sw - ww) / 2, (sh - hh) / 2, (sw + ww) / 2, (sh + hh) / 2))
    bg = blurbg(src, key)
    bg.paste(fg, ((W - fw) // 2 + dx, (H - fh) // 2 + dy))
    return bg

def block_h(lines, size, gap=1.25):
    return int(size * gap) * len(lines) + 30

def tover(img, lines, size=64, alpha=1.0, band=True, x=None, cx=None, y=None, yb=None,
          fill=SAND, weight='Bold', gap=1.25):
    if alpha <= 0:
        return
    f = F(size, weight)
    lh = int(size * gap)
    bw = int(max(f.getlength(s) for s in lines)) + 60
    bh = lh * len(lines) + 30
    px = int(cx - bw / 2) if cx is not None else int(x if x is not None else 90)
    py = int(y) if y is not None else int((yb if yb is not None else H - 80) - bh)
    ov = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    if band:
        d.rectangle([0, 0, bw, bh], fill=(5, 15, 25, int(150 * alpha)))
    for i, s in enumerate(lines):
        if cx is not None and not band:
            tw = f.getlength(s)
            d.text(((bw - tw) / 2, 15 + i * lh), s, font=f, fill=fill + (int(255 * alpha),))
        else:
            d.text((30, 15 + i * lh), s, font=f, fill=fill + (int(255 * alpha),))
    reg = img.crop((px, py, px + bw, py + bh)).convert('RGBA')
    img.paste(Image.alpha_composite(reg, ov).convert('RGB'), (px, py))

def appear(t, t_in, dur=.35):
    return ease((t - t_in) / dur)

def tag(img, s):
    if not DRAFT:
        return
    d = ImageDraw.Draw(img)
    f = FM(22)
    w = f.getlength(s)
    x = W - w - 40
    d.rectangle([x - 10, 22, W - 20, 60], fill=(0, 0, 0))
    d.text((x, 29), s, font=f, fill=(255, 200, 90))

def dashed_rect(d, box, col, width=3, dash=26, gap=16):
    x0, y0, x1, y1 = box
    def hline(y, a, b):
        x = a
        while x < b:
            d.line([(x, y), (min(x + dash, b), y)], fill=col, width=width); x += dash + gap
    def vline(x, a, b):
        y = a
        while y < b:
            d.line([(x, y), (x, min(y + dash, b))], fill=col, width=width); y += dash + gap
    hline(y0, x0, x1); hline(y1, x0, x1); vline(x0, y0, y1); vline(x1, y0, y1)

@lru_cache(None)
def gap_base(spec):
    img = Image.new('RGB', (W, H), (28, 34, 42))
    d = ImageDraw.Draw(img)
    dashed_rect(d, (70, 70, W - 70, H - 70), (95, 110, 125))
    heights = [int(sz * 1.5) for _, sz, _ in spec]
    y = (H - sum(heights)) // 2 - 40
    for (s, sz, col), h in zip(spec, heights):
        f = FM(sz)
        d.text(((W - f.getlength(s)) / 2, y), s, font=f, fill=col)
        y += h
    return img

def gap(head, *notes):
    spec = [(head, 46, (200, 215, 230))] + [(n, 30, (140, 158, 175)) for n in notes]
    return gap_base(tuple(spec)).copy()

def stack_card(text, w=400, h=560):
    img = Image.new('RGB', (w, h), (28, 34, 42))
    d = ImageDraw.Draw(img)
    dashed_rect(d, (8, 8, w - 8, h - 8), (95, 110, 125), width=3, dash=16, gap=10)
    f = FM(26)
    lines = []
    for para in text.split('\n'):
        cur = ''
        for wd in para.split():
            if len(cur) + len(wd) + 1 > 22:
                lines.append(cur); cur = wd
            else:
                cur = (cur + ' ' + wd).strip()
        lines.append(cur)
    y = (h - len(lines) * 40) // 2
    for ln in lines:
        d.text(((w - f.getlength(ln)) / 2, y), ln, font=f, fill=(200, 215, 230)); y += 40
    return img

def mock_label(img, alpha=1.0):
    tover(img, ['Concept mockup,', 'AI-generated.', 'Not the final book.'], size=34, x=W - 420, yb=H - 80,
          alpha=alpha, weight='SemiBold', fill=(255, 235, 170))

# ---------------------------------------------------------------- assets
IMG = lambda f: Image.open(A(f)).convert('RGB')
sunrise = IMG('sunrise_anse_royale.jpg')
seine = IMG('beach3_seine.jpg')
pirogue = IMG('beach2_pirogue.jpg')
orange = IMG('school_orange_ladigue.jpg')
library = IMG('library1.jpg')

_m = IMG('mockup_cover.jpg')
mock = _m.crop((684, 414, 1836, 1428))   # cover only, desk and screens cropped out

glacis = IMG('school_mahe_glacis.jpg')   # banners name the school: blur them out
for box in [(330, 156, 482, 210), (360, 252, 460, 304)]:
    reg = glacis.crop(box).filter(ImageFilter.GaussianBlur(22))
    glacis.paste(reg, box[:2])
GLACIS_REGION = (120, 0, 675, 312.1)

def mock_frame(p=0.0, z0=1.0, z1=1.05):
    img = navy()
    sw, sh = mock.size
    fh = H - 60
    fw = int(sw * fh / sh)
    z = lerp(z0, z1, ease(p))
    ww, hh = sw / z, sh / z
    fg = mock.resize((fw, fh), Image.LANCZOS, box=((sw - ww) / 2, (sh - hh) / 2, (sw + ww) / 2, (sh + hh) / 2))
    img.paste(fg, ((W - fw) // 2, 30))
    return img

# QR clip (1280x720, 24 fps): decode once
def load_clip():
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', A('qr_scan_2.4b.mp4'), '-vf', 'fps=24', '-f', 'rawvideo',
                          '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    n = len(raw) // (1280 * 720 * 3)
    return [Image.frombytes('RGB', (1280, 720), raw[i * 1280 * 720 * 3:(i + 1) * 1280 * 720 * 3]) for i in range(n)]
CLIP = None

# ---------------------------------------------------------------- shots
def s11(t, d):
    img = fit(sunrise, 'sun', 1.0, 1.10, t / d)
    a = appear(t, 0.8)
    tover(img, ['115 islands', '1.35 million km²'], size=70, alpha=a, x=90, yb=H - 90)
    tag(img, '[DRAFT: 452x678 file, send original. Label AI-generated if generated]')
    return img

def s12a(t, d):
    img = kb(seine, (0, 520, 930, 520 + 930 * 9 / 16), 1.0, 1.16, t / d, fx=.3, fy=.5)
    tag(img, '[DRAFT: confirm photo owner. Soft at 1080p]')
    return img

def s12b(t, d):
    img = fit(pirogue, 'pir', 1.0, 1.09, t / d)
    tag(img, '[DRAFT: confirm photo owner. Low resolution]')
    return img

def s12c(t, d):
    img = gap('[Book p5, plate 01: tektek in cross-section]', 'Slow push in on the clams.')
    tover(img, ['Tektek', 'Atactodea striata'], size=64, alpha=appear(t, .1), x=90, yb=H - 90)
    return img

def depth_ladder(img, prog, x=1760, y0=150, y1=930):
    dr = ImageDraw.Draw(img)
    dr.line([(x, y0), (x, y1)], fill=(95, 110, 125), width=5)
    f = FM(26)
    dr.text((x + 24, y0 - 14), '0 m', font=f, fill=(170, 185, 200))
    dr.text((x + 24, y1 - 14), '4,600 m', font=f, fill=(170, 185, 200))
    yy = y0 + (y1 - y0) * prog
    dr.ellipse([x - 16, yy - 16, x + 16, yy + 16], fill=TEAL)

def s13(t, d):
    img = gap('[Book p18: vertical depth diagram]', 'Camera drops down the water column.', 'Do not use p32 (partner logo).')
    depth_ladder(img, ease(t / d))
    a = appear(t, 4.0)
    tover(img, ['4,600 m', 'deepest ROV dive'], size=70, alpha=a, x=90, yb=H - 90)
    return img

def s14(t, d):
    img = gap('[Book p18, held at its deepest point]')
    depth_ladder(img, 1.0)
    tover(img, ['4,600 m', 'deepest ROV dive'], size=70, x=90, yb=H - 90)
    f = 1 - ease((t - (d - 0.7)) / 0.7)
    return Image.fromarray((np.asarray(img) * f).astype(np.uint8))

def s21a(t, d):
    img = mock_frame(t / 7.0)
    tover(img, ['Ages 12 to 18', 'Secondary 1 to', 'post-secondary'], size=38, x=10, y=400, band=False, weight='Bold')
    mock_label(img)
    return img

def s21b(t, d):
    img = gap('[Book p5, plate 01: line-art colouring plate]', 'Drifts across the page.')
    tover(img, ['Ages 12 to 18', 'Secondary 1 to post-secondary'], size=52, x=90, yb=H - 90)
    return img

STACK = ['[Book p18:\ndepth diagram]', '[Book p5:\ncolouring plate]', '[Book p36:\nglossary page]',
         '[Book p7:\npuzzle page, fix the word search first]']
STACK_IMGS = [stack_card(s) for s in STACK]

def s22(t, d):
    img = navy()
    p = ease((t - 2.2) / 1.5)
    for i, c in enumerate(STACK_IMGS):
        x0 = 300 + 440 * i
        x1 = 960 + (i - 1.5) * 26
        y0, y1 = 540, 540 - (i - 1.5) * 14
        cx, cy = lerp(x0, x1, p), lerp(y0, y1, p)
        img.paste(c, (int(cx - 200), int(cy - 280)))
    return img

DEPTHS = [('1', '0 to 30 m', 30), ('2', '10 to 65 m', 65), ('3', '0 to 100 m', 100),
          ('4', '30 to 350 m', 350), ('5', '0 to 500 m', 500), ('6', '20 to 4,600 m', 4600)]
MARK_T = [0.3 + 0.55 * i for i in range(6)]

def s23(t, d):
    img = navy()
    dr = ImageDraw.Draw(img)
    y0, y1, x = 120, 960, 960
    dr.line([(x, y0), (x, y1)], fill=(95, 130, 150), width=8)
    for i, (n, lab, dep) in enumerate(DEPTHS):
        a = appear(t, MARK_T[i], .25)
        if a <= 0:
            continue
        y = y0 + (y1 - y0) * math.sqrt(dep / 4600)
        r = 16
        dr.ellipse([x - r, y - r, x + r, y + r], fill=TEAL)
        txt = f'{n}   {lab}'
        bw = int(F(34, 'SemiBold').getlength(txt)) + 60
        px = (x + 10) if i % 2 else (x - 10 - bw)
        tover(img, [txt], size=34, alpha=a, band=False, weight='SemiBold', x=px, y=y - 26)
    tover(img, ['Six chapters', 'shore to 4,600 m'], size=64, alpha=appear(t, .2), x=90, yb=H - 90, band=False)
    tag(img, '[DRAFT: marks sit at the deeper end of each chapter range]')
    return img

def s24a(t, d):
    return gap('[Book p5 and p11: colouring plates, close-up]', 'Then a hand brings a phone toward the QR code.')

def s24b(t, d):
    global CLIP
    if CLIP is None:
        CLIP = load_clip()
    fr = CLIP[min(int(t * FPS), len(CLIP) - 1)].resize((W, H), Image.LANCZOS)
    tag(fr, '[PLACEHOLDER: AI-generated clip. Replace with real footage]')
    return fr

def s25(t, d):
    return gap('[Book p35: Community Drawing Canvas, empty]', 'Crop out the banner at the bottom (it names a partner).')

def s31(t, d):
    img = fit(pirogue, 'pir', 1.04, 1.12, t / d)
    lines = ['Grew up at Beau Vallon', '12 years in GIS', 'Led maritime boundary delimitation', 'Founded a hydrospatial NGO']
    times = [0.4, 2.0, 4.3, 7.0]
    shown = [(l, appear(t, tt)) for l, tt in zip(lines, times) if t >= tt]
    y = 120
    for l, a in shown:
        tover(img, [l], size=56, alpha=a, x=90, y=y)
        y += 95
    tag(img, '[DRAFT: Beach 2 reused per script. Confirm owner]')
    return img

def s41(t, d):
    img = navy()
    if t < 6.4:
        tover(img, ['£5,000'], size=190, alpha=appear(t, .3), x=120, y=120, band=False, fill=SAND)
        items = ['100 printed copies', 'Digital edition', 'QR audio and video', 'Workshops']
        ts = [1.8, 3.0, 4.0, 5.2]
        for i, (s, tt) in enumerate(zip(items, ts)):
            tover(img, [s], size=72, alpha=appear(t, tt), x=130, y=420 + i * 120, band=False, weight='SemiBold', fill=TEAL)
    else:
        dr = ImageDraw.Draw(img)
        isl = [('Mahé', 55.46, -4.65, 6.9), ('Praslin', 55.73, -4.33, 8.0), ('La Digue', 55.84, -4.36, 9.0)]
        for name, lon, lat, tt in isl:
            x = 960 + (lon - 55.65) * 1667
            y = 560 + (-lat - 4.50) * 1667
            a = appear(t, tt, .3)
            dr.ellipse([x - 16, y - 16, x + 16, y + 16], outline=DIM, width=4)
            if a > 0:
                g = int(40 * a)
                for rr in (44, 32):
                    dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=tuple(int(c * 0.35 * a + b * (1 - 0.35 * a)) for c, b in zip(TEAL, NAVY2)))
                dr.ellipse([x - 20, y - 20, x + 20, y + 20], fill=TEAL)
                tover(img, [name], size=54, alpha=a, band=False, x=x + 60, y=y - 40, weight='Bold')
        tag(img, '[DRAFT: island markers only, no coastlines]')
    return img

def s42(t, d):
    img = navy()
    tover(img, ['50'], size=330, alpha=appear(t, .1), cx=W / 2, y=150, band=False)
    tover(img, ['workshop participants'], size=84, alpha=appear(t, .3), cx=W / 2, y=560, band=False, weight='SemiBold', fill=TEAL)
    tover(img, ['pilots that seed the book'], size=52, alpha=appear(t, .9), cx=W / 2, y=720, band=False, weight='Regular')
    return img

def s43(t, d):
    seg = int(t // 1.0)
    p = t - seg
    if seg == 0:
        img = kb(glacis, GLACIS_REGION, 1.0, 1.08, p)
        tover(img, ['Mahé'], size=80, x=90, yb=H - 90)
        tag(img, '[DRAFT: school banners blurred. Confirm owner]')
    elif seg == 1:
        img = gap('[Praslin school photo needed]', 'People-free. The Grand Anse Primary file has pupils', 'on every balcony, so it cannot be cropped clear.')
        tover(img, ['Praslin'], size=80, x=90, yb=H - 90)
    elif seg == 2:
        img = fit(orange, 'org', 1.0, 1.08, p)
        tover(img, ['La Digue'], size=80, x=90, yb=H - 90)
        tag(img, '[DRAFT: confirm this is La Digue. Confirm owner]')
    elif seg == 3:
        img = fit(library, 'lib', 1.12, 1.12, 0, dx=int(lerp(90, -90, p)))
        tag(img, '[DRAFT: library 1 only. Confirm owner and island]')
    else:
        img = mock_frame(p, 1.0, 1.03)
        mock_label(img)
    return img

def s44(t, d):
    img = navy()
    tover(img, ['Digital edition targets'], size=48, cx=W / 2, y=130, band=False, weight='Regular', fill=DIM)
    for i, (num, lab, tt) in enumerate([('500', 'downloads', 3.6), ('200', 'QR scans', 5.2)]):
        cx = 560 + 800 * i
        tover(img, [num], size=300, alpha=appear(t, tt), cx=cx, y=260, band=False)
        tover(img, [lab], size=80, alpha=appear(t, tt + .15), cx=cx, y=640, band=False, weight='SemiBold', fill=TEAL)
    return img

def s51(t, d):
    img = navy()
    dr = ImageDraw.Draw(img)
    xa, xb, y = 460, 1460, 520
    a1 = appear(t, .3)
    pe = ease((t - 1.0) / 1.5)
    dr.line([(xa, y), (lerp(xa, xb, pe), y)], fill=(95, 130, 150), width=8)
    dr.ellipse([xa - 22, y - 22, xa + 22, y + 22], fill=TEAL if a1 > 0 else NAVY1)
    if t > 2.4:
        dr.ellipse([xb - 22, y - 22, xb + 22, y + 22], fill=TEAL)
    tover(img, ['Survey at the', 'first workshop'], size=54, alpha=a1, cx=xa, y=y + 60, band=False, weight='SemiBold')
    tover(img, ['Survey', '6 months later'], size=54, alpha=appear(t, 2.4), cx=xb, y=y + 60, band=False, weight='SemiBold')
    return img

def s52(t, d):
    if t < 3.0:
        img = navy()
        tover(img, ['50%'], size=360, alpha=appear(t, .1), cx=W / 2, y=130, band=False)
        tover(img, ['or more of students'], size=84, alpha=appear(t, .3), cx=W / 2, y=600, band=False, weight='SemiBold', fill=TEAL)
        tover(img, ['more interested in an ocean science career'], size=48, alpha=appear(t, .8), cx=W / 2, y=760, band=False, weight='Regular')
    else:
        img = gap('[Book p24: students in a ship’s lab, illustration]')
        tover(img, ['50% or more of students'], size=64, x=90, yb=H - 90)
    return img

def golden(t, which):
    if which == 2:
        return gap('[Golden photo 2: original needed]', 'Golden-hour light through a tree branch onto sand and waterline.', 'Crop to 16:9. No people in frame.')
    return gap('[Golden photo 1: original needed]', 'Sun low over the water, a catamaran and an island on the horizon.', 'Crop to 16:9. No people in frame.')

def s61(t, d): return golden(t, 2)
def s62(t, d): return golden(t, 1)

def s63(t, d):
    img = golden(t, 1)
    tover(img, ['£5,000'], size=150, alpha=appear(t, .5), x=120, y=640, band=True)
    tover(img, ['OCL grant'], size=70, alpha=appear(t, 1.4), x=120, y=830, band=True, weight='SemiBold', fill=TEAL)
    return img

def s64(t, d):
    img = navy()
    a = appear(t, .2, .5) * (1 - ease((t - (d - 0.9)) / 0.9))
    tover(img, ['Beyond the Reef Line'], size=130, alpha=a, cx=W / 2, y=380, band=False)
    tover(img, ['Francesca Adrienne'], size=60, alpha=a, cx=W / 2, y=580, band=False, weight='Regular', fill=TEAL)
    return img

SHOTS = [
    ('1.1', 0, 6, s11), ('1.2a', 6, 11, s12a), ('1.2b', 11, 13, s12b), ('1.2c', 13, 14, s12c),
    ('1.3', 14, 21, s13), ('1.4', 21, 25, s14),
    ('2.1a', 25, 29, s21a), ('2.1b', 29, 32, s21b), ('2.2', 32, 38, s22), ('2.3', 38, 42, s23),
    ('2.4a', 42, 46, s24a), ('2.4b', 46, 49, s24b), ('2.5', 49, 55, s25),
    ('3.1', 55, 64, s31),
    ('4.1', 64, 74, s41), ('4.2', 74, 77, s42), ('4.3', 77, 82, s43), ('4.4', 82, 89, s44),
    ('5.1', 89, 93, s51), ('5.2', 93, 99, s52),
    ('6.1', 99, 103, s61), ('6.2', 103, 111, s62), ('6.3', 111, 117, s63), ('6.4', 117, 120, s64),
]

def frame_at(t):
    for sid, a, b, fn in SHOTS:
        if a <= t < b:
            img = fn(t - a, b - a)
            break
    else:
        img = navy()
    if t < 0.5:
        f = ease(t / 0.5)
        img = Image.fromarray((np.asarray(img) * f).astype(np.uint8))
    return img

# ---------------------------------------------------------------- audio
def read_audio(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()

def place(buf, sig, t0, gain=1.0):
    i = int(t0 * SR)
    j = min(i + len(sig), len(buf))
    if j > i:
        buf[i:j] += sig[:j - i] * gain

def fades(sig, fi, fo):
    s = sig.copy()
    ni, no = int(fi * SR), int(fo * SR)
    if ni: s[:ni] *= np.linspace(0, 1, ni)
    if no: s[-no:] *= np.linspace(1, 0, no)
    return s

def surf(dur, seed, lp=900, swell=0.11):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= 1 / np.sqrt(np.maximum(f, 20) / 20)
    X[f > lp] = 0
    x = np.fft.irfft(X, n)
    x /= np.abs(x).max()
    t = np.arange(n) / SR
    env = 0.45 + 0.55 * np.sin(2 * np.pi * swell * t + rng.uniform(0, 6)) ** 2
    return (x * env).astype(np.float32)

def tone(freq, dur, decay=None):
    t = np.arange(int(dur * SR)) / SR
    s = np.sin(2 * np.pi * freq * t)
    if decay:
        s *= np.exp(-t * decay)
    return s.astype(np.float32)

def burst(dur, lo, hi, seed, rise=False):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(X, n)
    x /= np.abs(x).max()
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** (1.5 if not rise else 1.0)
    return (x * env).astype(np.float32)

def sfx_bed():
    buf = np.zeros(int(DUR * SR), dtype=np.float32)
    place(buf, fades(surf(14, 1), 1.0, 1.0), 0, 0.22)                    # scene 1 shore
    lt = tone(82.4, 11.0); lt = fades(lt, 1.5, 0.02)
    place(buf, lt, 14.0, 0.14)                                          # low tone under descent, clean cut at 25
    place(buf, burst(0.7, 400, 4000, 3), 35.4, 0.22)                    # cards meet
    for i, fq in enumerate([130.8, 146.8, 164.8, 185.0, 207.7, 233.1]):  # rising tone per mark
        place(buf, fades(tone(fq, 0.55, decay=4), 0.01, 0.05), 38.0 + MARK_T[i], 0.18)
    place(buf, burst(0.25, 2000, 9000, 4), 43.0, 0.12)                  # page turn
    place(buf, fades(tone(1000, 0.14), 0.005, 0.02), 45.7, 0.18)        # phone beep
    place(buf, fades(surf(18, 7), 1.2, 3.0), 99.0, 0.20)                # closing waves
    return buf

def scene_text():
    s = open(os.path.join(HERE, 'script.md')).read()
    vo = s.split('## Voiceover')[1].split('## Storyboard')[0]
    parts = re.split(r'\[(\d+:\d+) to (\d+:\d+), Scene (\d)[^\]]*\]\n', vo)
    return {parts[i + 2]: parts[i + 3].strip() for i in range(1, len(parts), 4)}

WINDOWS = {'scene1': (0.15, 25), 'scene2a': (25.15, 46), 'scene2b': (49.1, 55), 'scene3': (55.1, 64),
           'scene4': (64.1, 89), 'scene5': (89.1, 99), 'scene6': (99.1, 117)}

def guide_voice():
    tx = scene_text()
    parts = {'scene1': tx['1'], 'scene3': tx['3'], 'scene4': tx['4'], 'scene5': tx['5'], 'scene6': tx['6']}
    cut = tx['2'].index('and a Drawing Canvas')
    parts['scene2a'], parts['scene2b'] = tx['2'][:cut].strip(), tx['2'][cut:].strip()
    tmp = os.path.join(OUT, 'tmp'); os.makedirs(tmp, exist_ok=True)
    buf = np.zeros(int(DUR * SR), dtype=np.float32)
    for k, text in parts.items():
        t0, t1 = WINDOWS[k]
        wav = os.path.join(tmp, k + '.wav')
        subprocess.run(['espeak-ng', '-v', 'en-gb', '-s', '145', '-w', wav, text], check=True)
        sig = read_audio(wav)
        win = t1 - t0
        need = len(sig) / SR
        if need > win * 0.98:
            fac = min(need / (win * 0.98), 1.4)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', wav, '-filter:a', f'atempo={fac:.3f}', os.path.join(tmp, k + '_f.wav')], check=True)
            sig = read_audio(os.path.join(tmp, k + '_f.wav'))
        print(f'  guide {k}: {need:.1f}s natural vs {win:.1f}s window ({len(text.split())} words)')
        place(buf, sig / max(np.abs(sig).max(), 1e-6), t0, 0.8)
    return buf

def user_voice():
    vdir = os.path.join(HERE, 'vo')
    found = {k: os.path.join(vdir, k + '.wav') for k in WINDOWS if os.path.exists(os.path.join(vdir, k + '.wav'))}
    if not found:
        return None
    buf = np.zeros(int(DUR * SR), dtype=np.float32)
    for k, p in found.items():
        place(buf, read_audio(p), WINDOWS[k][0], 1.0)
    return buf

def write_wav(path, mono):
    x = np.clip(mono, -1, 1)
    pcm = (x * 32767).astype(np.int16)
    st = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())

def mux(video, audio, out):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-i', audio, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-t', str(DUR), out], check=True)

# ---------------------------------------------------------------- main
def preview():
    pdir = os.path.join(OUT, 'preview'); os.makedirs(pdir, exist_ok=True)
    ims = []
    for sid, a, b, fn in SHOTS:
        t = a + (b - a) * 0.8 if sid not in ('4.1',) else a + 8.5
        if sid == '4.3': t = a + 0.5
        im = frame_at(t)
        im.save(os.path.join(pdir, sid + '.png'))
        ims.append((sid, im.resize((640, 360))))
    sheet = Image.new('RGB', (640 * 4, 360 * 6), (0, 0, 0))
    for i, (sid, im) in enumerate(ims):
        sheet.paste(im, ((i % 4) * 640, (i // 4) * 360))
    sheet.save(os.path.join(pdir, 'contact.jpg'), quality=85)

def render():
    os.makedirs(OUT, exist_ok=True)
    silent = os.path.join(OUT, 'video_silent.mp4')
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                          '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', silent],
                         stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for i in range(n):
        p.stdin.write(frame_at(i / FPS).tobytes())
        if i % 240 == 0:
            print(f'frame {i}/{n}', flush=True)
    p.stdin.close(); p.wait()

    bed = sfx_bed()
    kre = read_audio(A('qr_scan_2.4b.mp4'))
    i0, i1 = int(46.0 * SR), int(49.0 * SR)
    bed[i0:i1] = 0               # nothing else under her Kreol line
    place(bed, kre, 46.0, 1.0)   # full level

    tag_ = 'draft' if DRAFT else 'clean'
    base = f'BeyondTheReefLine_{tag_}'
    write_wav(os.path.join(OUT, 'a_novo.wav'), bed)
    mux(silent, os.path.join(OUT, 'a_novo.wav'), os.path.join(OUT, base + '_noVO.mp4'))

    gv = guide_voice()
    mix = bed + gv
    mix[i0:i1] = bed[i0:i1]   # voiceover stops for the Kreol line
    write_wav(os.path.join(OUT, 'a_guide.wav'), mix)
    mux(silent, os.path.join(OUT, 'a_guide.wav'), os.path.join(OUT, base + '_SCRATCH-guideVO_pacing-only.mp4'))

    uv = user_voice()
    if uv is not None:
        mix = bed + uv
        mix[i0:i1] = bed[i0:i1]
        write_wav(os.path.join(OUT, 'a_final.wav'), mix)
        mux(silent, os.path.join(OUT, 'a_final.wav'), os.path.join(OUT, base + '_finalVO.mp4'))
    shutil.rmtree(os.path.join(OUT, 'tmp'), ignore_errors=True)
    print('done')

if __name__ == '__main__':
    if '--preview' in sys.argv:
        preview()
    else:
        render()
