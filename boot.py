# =============================================================
#  Radionica: prepoznavanje znamenki na M5StickV (MaixPy)
#  Tipka A (velika, sprijeda): pauza / nastavak
#  Tipka B (bočna):            prebaci model (VAS -> REZ -> LIN)
# =============================================================
import sensor, image, lcd, time, gc, os
import KPU as kpu
from Maix import GPIO
from fpioa_manager import fm

# ---------- Postavke (po potrebi prilagodite) ----------
MODEL_UCENIK = "/flash/mnist.kmodel"    # model koji šalju učenici
MODEL_REZERVA = "/flash/rezerva.kmodel" # rezervni model voditelja
MODEL_LINEARNI = "/flash/linearni.bin"  # linearni model (računa ga procesor, ne KPU)
PRAG_LIN = 0.35          # prag za linearni model (njegovi rezultati nisu vjerojatnosti)
ZRCALI_H = 1        # ako je slika zrcaljena, promijenite 1 <-> 0
ZRCALI_V = 1        # ako je slika naopako, promijenite 1 <-> 0
PRAG_SIGURNOSTI = 0.60   # ispod toga prikazujemo "?"
OKVIR = 96          # veličina kvadrata (u pikselima) u kojem tražimo znamenku
DEBLJINA = 1        # 0 = ne podebljavaj, 1 = podebljaj tanke linije

# ---------- Ekran ----------
try:
    lcd.init(type=3)
except Exception:
    lcd.init()
lcd.rotation(2)
lcd.clear(lcd.BLACK)
lcd.draw_string(10, 10, "Pokretanje...", lcd.WHITE, lcd.BLACK)

# ---------- Tipke (M5StickV: A = IO36, B = IO37) ----------
fm.register(36, fm.fpioa.GPIO1, force=True)
fm.register(37, fm.fpioa.GPIO2, force=True)
tipka_a = GPIO(GPIO.GPIO1, GPIO.IN, GPIO.PULL_UP)
tipka_b = GPIO(GPIO.GPIO2, GPIO.IN, GPIO.PULL_UP)

# ---------- Kamera ----------
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QQVGA)   # 160 x 120, stane na ekran 240 x 135
sensor.set_hmirror(ZRCALI_H)
sensor.set_vflip(ZRCALI_V)
sensor.run(1)
sensor.skip_frames(time=500)

SIRINA = 160
VISINA = 120
OX = (SIRINA - OKVIR) // 2
OY = (VISINA - OKVIR) // 2

# Crni kvadrat 56x56 za brisanje panela (radi i na starom firmwareu bez lcd.fill_rectangle)
PRAZNO = sensor.snapshot().copy(roi=(0, 0, 56, 56))
PRAZNO.clear()


def postoji(putanja):
    try:
        os.stat(putanja)
        return True
    except Exception:
        return False


def ucitaj_linearni(putanja):
    """linearni.bin: 'LIN1', mjerilo (2 bajta), zatim 785 x 10 tezina (int8).
    Redak = piksel (y*28 + x), zadnji redak = pomak; stupac = znamenka."""
    with open(putanja, "rb") as f:
        d = f.read()
    if d[0:4] != b"LIN1" or len(d) != 6 + 7850:
        raise ValueError("neispravan linearni.bin")
    mjerilo = d[4] | (d[5] << 8)
    w = [v - 256 if v > 127 else v for v in d[6:]]
    return ("LIN", w, mjerilo)


def linearni_predvidi(model, malo):
    """Zbroj piksel x tezina za svaku znamenku. Slika je crno-bijela,
    pa zbrajamo samo tezine 'upaljenih' piksela."""
    w = model[1]
    s = w[7840:7850]            # pocinjemo od pomaka (bias)
    for y in range(28):
        red = y * 280
        for x in range(28):
            if malo.get_pixel(x, y) > 127:
                p = red + x * 10
                for z in range(10):
                    s[z] += w[p + z]
    return [v / model[2] for v in s]


def je_linearni(t):
    return isinstance(t, tuple)


def ucitaj(putanja):
    try:
        if putanja.endswith(".bin"):
            return ucitaj_linearni(putanja)
        return kpu.load(putanja)
    except Exception as e:
        # Najcesci uzrok: stari firmware (npr. 0.4.0) ne zna citati kmodel v4 -> nadogradite MaixPy
        print("Ne mogu ucitati", putanja, ":", e)
        return None


# ---------- Modeli ----------
modeli = []
if postoji(MODEL_UCENIK):
    modeli.append(("VAS", MODEL_UCENIK))
if postoji(MODEL_REZERVA):
    modeli.append(("REZ", MODEL_REZERVA))
if postoji(MODEL_LINEARNI):
    modeli.append(("LIN", MODEL_LINEARNI))
if not modeli:
    lcd.clear(lcd.RED)
    lcd.draw_string(5, 40, "Nema modela u /flash!", lcd.WHITE, lcd.RED)
    lcd.draw_string(5, 60, "Posaljite mnist.kmodel", lcd.WHITE, lcd.RED)
    raise SystemExit

indeks = 0
task = ucitaj(modeli[indeks][1])
if task is None and len(modeli) > 1:
    indeks = 1
    task = ucitaj(modeli[indeks][1])


def pripremi(img):
    """Iz kadra izreže znamenku i napravi sliku 28x28 kao u skupu MNIST.
    Vraća None ako u okviru nema ničega."""
    isjecak = img.copy(roi=(OX, OY, OKVIR, OKVIR))
    isjecak.invert()                                  # tamno na svijetlom -> svijetlo na tamnom
    prag = isjecak.get_histogram().get_threshold().value()   # automatski prag (Otsu)
    prag = max(prag, 100)                             # prazan papir ne smije dati "mrlje"
    isjecak.binary([(prag, 255)])                     # crno-bijelo
    if DEBLJINA:
        isjecak.dilate(DEBLJINA)
    mrlje = isjecak.find_blobs([(128, 255)], pixels_threshold=15,
                               area_threshold=15, merge=True, margin=4)
    if not mrlje:
        return None, None
    # Spoji sve mrlje u jedan pravokutnik
    x0 = min([b.x() for b in mrlje])
    y0 = min([b.y() for b in mrlje])
    x1 = max([b.x() + b.w() for b in mrlje])
    y1 = max([b.y() + b.h() for b in mrlje])
    w = x1 - x0
    h = y1 - y0
    if w * h > OKVIR * OKVIR * 0.9:      # cijeli okvir je "crn" -> nema papira
        return None, None
    # Kvadrat oko znamenke + rub (MNIST: znamenka ~20 px u slici 28 px)
    strana = int(max(w, h) * 1.4) + 2
    cx = x0 + w // 2
    cy = y0 + h // 2
    sx = max(0, min(OKVIR - strana, cx - strana // 2))
    sy = max(0, min(OKVIR - strana, cy - strana // 2))
    strana = min(strana, OKVIR)
    znamenka = isjecak.copy(roi=(sx, sy, strana, strana))
    malo = znamenka.resize(28, 28)
    del isjecak, znamenka
    return malo, (OX + x0, OY + y0, w, h)


pauza = False
zadnji_a = 1
zadnji_b = 1
rez_tekst = "-"
rez_sig = 0.0
mreza_vidi = None

while True:
    # --- tipke ---
    a = tipka_a.value()
    b = tipka_b.value()
    if a == 0 and zadnji_a == 1:
        pauza = not pauza
    if b == 0 and zadnji_b == 1 and len(modeli) > 1:
        if task is not None and not je_linearni(task):
            try:
                kpu.deinit(task)
            except Exception:
                pass
        task = None
        gc.collect()
        indeks = (indeks + 1) % len(modeli)
        task = ucitaj(modeli[indeks][1])
        lcd.clear(lcd.BLACK)
    zadnji_a = a
    zadnji_b = b

    if pauza:
        time.sleep_ms(50)
        continue

    img = sensor.snapshot()
    malo, okvir_znamenke = pripremi(img)

    if malo is not None and task is not None:
        if je_linearni(task):
            vjer = linearni_predvidi(task, malo)
            prag = PRAG_LIN
        else:
            malo.pix_to_ai()
            fmap = kpu.forward(task, malo)
            vjer = fmap[:]
            prag = PRAG_SIGURNOSTI
        najveca = max(vjer)
        znam = vjer.index(najveca)
        rez_sig = min(max(najveca, 0.0), 1.0)
        rez_tekst = str(znam) if najveca >= prag else "?"
        mreza_vidi = malo.resize(56, 56)
    else:
        rez_tekst = "-"
        rez_sig = 0.0
        mreza_vidi = None

    # --- crtanje na slici kamere (160 x 120) ---
    img.draw_rectangle(OX, OY, OKVIR, OKVIR, color=255, thickness=1)
    if okvir_znamenke is not None:
        img.draw_rectangle(okvir_znamenke, color=128, thickness=1)
    img.draw_string(4, 2, rez_tekst, color=255, scale=4)   # velika znamenka gore lijevo
    lcd.display(img, oft=(0, 7))

    # --- desni panel (x 160..240, oko 9 znakova u retku) ---
    lcd.draw_string(164, 2, "Vidi:    ", lcd.WHITE, lcd.BLACK)
    if mreza_vidi is not None:
        lcd.display(mreza_vidi, oft=(172, 16))
    else:
        lcd.display(PRAZNO, oft=(172, 16))
    boja = lcd.GREEN if rez_tekst not in ("?", "-") else lcd.ORANGE
    lcd.draw_string(164, 78, ("Znam: " + rez_tekst + "  ")[:9], boja, lcd.BLACK)
    lcd.draw_string(164, 96, ("Sig: %d%%  " % int(rez_sig * 100))[:9], lcd.WHITE, lcd.BLACK)
    if task is None:
        lcd.draw_string(164, 118, "GRESKA!  ", lcd.RED, lcd.BLACK)
    else:
        lcd.draw_string(164, 118, "Model:" + modeli[indeks][0], lcd.YELLOW, lcd.BLACK)

    del img
    gc.collect()
