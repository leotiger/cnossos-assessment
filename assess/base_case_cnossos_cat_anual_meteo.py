# -*- coding: utf-8 -*-
# BASE-CASE PRO — CNOSSOS (ANUAL) amb 3 JOCS METEO (conservador, central, sec)
# - DEM 5 m, hillshade en grisos (nord amunt; valls clares, carenes fosques)
# - LwA espectral Nordex N175 (oficial)
# - CNOSSOS like: difracció multi-aresta (Deygout), sòl Gs/Gm/Gr, absorció ISO9613-1
# - Barreja meteo CNOSSOS: neutral vs favourable amb p_fav per Ld/Le/Ln
# - Ponderació estacional (DJF/MAM/JJA/SON) per Ld/Le/Ln → Lden (14/2/8)
# - SUC acolorits pel Lden total (turbines+preexistent), cercles Ø172 m, creu posició, llegenda distàncies
VERSION = 1.0

import argparse, sys, math
import numpy as np, rasterio, matplotlib.pyplot as plt, fiona, csv
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from shapely.geometry import Point, box, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import multiprocessing as mp
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os, tempfile, shutil
from affine import Affine

# ----------------------- YAML CONFIG SUPPORT -----------------------
try:
    import yaml  # PyYAML
except Exception as e:
    yaml = None

from typing import Mapping

def _deep_update(dst, src):
    """Recursively update dict/list/scalars in-place."""
    if isinstance(dst, dict) and isinstance(src, Mapping):
        for k, v in src.items():
            if k in dst and isinstance(dst[k], (dict, list)):
                dst[k] = _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst
    if isinstance(dst, list) and isinstance(src, list):
        return src  # replace list by default
    return src

def apply_yaml_overrides(cfg: dict):
    """Apply top-level keys from YAML onto same-named globals.
    - If a key matches an existing GLOBAL dict, deep-merge.
    - Otherwise, replace variable entirely.
    """
    g = globals()
    for k, v in cfg.items():
        if k in g:
            try:
                if isinstance(g[k], dict) and isinstance(v, Mapping):
                    g[k] = _deep_update(g[k], v)
                else:
                    g[k] = v
            except Exception:
                g[k] = v
        else:
            g[k] = v  # allow introducing new config variables

    global BBOX_POLY
    BBOX_POLY = box(XMIN, YMIN, XMAX, YMAX)
    load_suc_geoms()
    global active_limits_key
    active_limits_key = globals().get("active_limits_key", list(limits_sets.keys())[0])
    global LIMITS
    LIMITS = limits_sets[active_limits_key]
            
            
def load_config_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML no instal·lat: pip install pyyaml")
    with open(path, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError("El YAML de configuració ha de tenir un mapping al nivell superior")
    return cfg

def ensure_output_dir(output_dir: str | None) -> str | None:
    """
    Si s'especifica output_dir, assegura que existeix (mkdir -p).
    Retorna la ruta normalitzada o None si no s'ha indicat.
    """
    if output_dir:
        out = os.path.abspath(output_dir)
        os.makedirs(out, exist_ok=True)        
        return out
    return None


# ----------------------- FITXERS D’ENTRADA -----------------------
# console command to prepare oficial DEM tiff to reduce processing data to avoid overflow
# gdalwarp -t_srs EPSG:25831 -te 342000 4604000 355000 4596000 \
#  -r bilinear -of GTiff -co COMPRESS=LZW \
#  PNOA_MDT05_ETRS89_HU30_0390_LID.tif \
#  PNOA_MDT05_ETRS89_HU30_0418_LID.tif \
#  PNOA_MDT05_ETRS89_HU31_0390_LID.tif \
#  PNOA_MDT05_ETRS89_HU31_0418_LID.tif \
#  DEM_clip_5m-rpglobal.tif

# gdalwarp -t_srs EPSG:25831 -te 340000 4604000 355000 4594000 \
#  -r bilinear -of GTiff -co COMPRESS=LZW \
#  PNOA_MDT05_ETRS89_HU30_0389_LID.tif \
#  PNOA_MDT05_ETRS89_HU30_0390_LID.tif \
#  PNOA_MDT05_ETRS89_HU30_0417_LID.tif \
#  PNOA_MDT05_ETRS89_HU30_0418_LID.tif \
#  PNOA_MDT05_ETRS89_HU31_0389_LID.tif \
#  PNOA_MDT05_ETRS89_HU31_0390_LID.tif \
#  PNOA_MDT05_ETRS89_HU31_0417_LID.tif \
#  PNOA_MDT05_ETRS89_HU31_0418_LID.tif \
#  DEM_clip_5m-rpglobal.tif

# tiff oficial retallat a projectes RP GLOBAL 342000 4604000 35500 4594000
#DEM_PATH = "DEM_clip_5m-sant-roc-marti-comp.tif" # DEM 5m tiff retallat a Sant Roc + Martí
DEM_PATH = "DEM_clip_5m-rpglobal.tif" # DEM 5m tiff retallat a Sant Roc + Martí + Gardeny
GML_PATH = "MUC_classificacio.gml"      # Classificació de sòl (ha de contenir SUC)

# ----------------------- ÀMBIT (UTM) -----------------------
XMIN, YMIN, XMAX, YMAX = 344500, 4595500, 353500, 4600500
# Ambit Sant Roc + Martí
#xmin, ymin, xmax, ymax = 346000, 4596000, 354000, 4604000
# Ambit Sant Roc + Martí + Gardeny
#XMIN, YMIN, XMAX, YMAX = 340000, 4604000, 355000, 4594000

# BBOX_POLY = box(XMIN, YMIN, XMAX, YMAX)

YEAR = 2025
PROJECT = ""
ESIA = ""
OUTPUT_DIR = "SCENES"


# ----------------------- TURBINES -----------------------
TURBINES = []

RADIUS_NORM_CAT = 500.0
RADIUS_NORM_ESP = 1000.0
H_RECEIVER = 4.0  # m (façana)
FACANA_DB = 3.0   # +3 dB a façana
N_PROF_PER_KM = 60
# ----------------------- Limits d'immissió -----------------------
# Límits (configurable): dos escenaris habituals
limits_sets = {
    "Sens_55": [55.0, 50.0, 45.0],
    "Sens_60": [60.0, 60.0, 50.0],  # ← nou perfil
    "Sens_65":  [65.0, 60.0, 55.0],
}


# ----------------------- RECEPTORS (nuclis) + PREEXISTENT Ld/Le/Ln -----------------------
# --- RECEPTORS multi-punt per nucli (coords UTM, altura de recepció per punt)
# Registres reals han de tenir:
# ID, nom receptor i ús, tipus del punt (model/mesura), coordenades UTM, Cota sobre nivell del mar, 
# Alçada del microfòn, Façana i orientació de la façana, categoria urbanístaca i limits aplicables, tipus de sòl, 
# distància a reflectors
# A totes les mesures cal afegir 0.5dB i desprès agafar l'integre més a prop segons normativa
RECEIVERS = {}
    
# A tots els valors mesurats cal afegir 0.5dB i agafar l'integre més pròxim després segons normativa
# Projecte Sant Roc
# 6. Forès 352854 4595139 Nit 47,3 47,3 47,3 Dia 47,4 47,4 47,4

# Projecte Martí
# 4. Vallfogana del Riucorb 352660 4602743 Nit 32,7 32,7 32,7 Dia 46,1 46,1 46,1
# 5. Rocallaura  345461 4596767 Nit 47.3 47,3 47,3 Dia 45,4 45,4 45,4

# Projecte Gardeny Mesuraments Fake Promotor
# 1. Vallbona de les Monges 340592 4599036 nit: 35,6 35,6 35,6 dia 36,9 36,7 36,9
# 2. Rocafort de Vallbona 341313 4601854 nit 40,4 40,4 40,4 dia 58.0 32.0 32.0
# 3. Nalec 342846 4602553 nit 24,9 24,9 24,9 dia 35,8 35,8 35,8
# 4. Ciutadilla 344945 4602536 nit 26,7 26,7 26,7 dia 49,4 35,8 49,4
# 5. C14 345840 4599548 nit 35 35 35 dia 50,7 43,9 43,9
# 6. Rocallaura  345461 4596767 nit 52,5 52,5 52,5 dia 55,4 55,4 55,4

# ----------------------- EMISSIÓ ESPECTRAL (Nordex N175 oficial) -----------------------
BANDS = np.array([63,125,250,500,1000,2000,4000,8000], float)  # Hz
# LwA_OVB = np.array([91.8,98.6,102.0,102.5,103.4,101.3,92.0,75.5], float)  # dB (A), per banda

# ----------------------- CNOSSOS: p_fav (prob. favourable) per període -----------------------
P_FAV = {
    # "Ld": 0.30, "Le": 0.45, "Ln": 0.60
}  # AJUSTA segons climatologia local

# ----------------------- Ponderació estacional (suma=1 per clau) -----------------------
W_SEASON = {
#    "Ld": {"DJF":0.25,"MAM":0.25,"JJA":0.30,"SON":0.20},
#    "Le": {"DJF":0.25,"MAM":0.25,"JJA":0.25,"SON":0.25},
#    "Ln": {"DJF":0.30,"MAM":0.25,"JJA":0.20,"SON":0.25},
}

# ----------------------- Paràmetres CNOSSOS: sòl -----------------------
# Gs = coeficient de sòl prop de la font
# Gm = coeficient de sòl al mig del trajecte
# Gr = coeficient de sòl prop del receptor
GROUND_GS = 0.0   # sòl dur (prop de la turbina)
GROUND_GM = 0.6   # sòl mixt
GROUND_GR = 0.2   # sòl tou 0.8; sòl mixt: 0.5 (prop del receptor); sòl dur, habitatges a tocar d’asfalt, roca, carrers, etc.: 0.0 a 0.4

# ----------------------- PERFILS METEO (Sta. Coloma XEMA per T; bellcam per HR) -----------------------
PRESS_KPA = 97.0  # ~500 a 800 m
TEMP_XEMA = {
#    "DJF": 6.0, "MAM": 12.0, "JJA": 22.0, "SON": 14.0
}
HR_LOCUS = {
#    "DJF": 78.0, "MAM": 68.0, "JJA": 58.0, "SON": 72.0
}
TEMP_INVTERM = 4.0 # Tenim molta inversió tèrmica a Passanant, Forès i els inverns són cada vegada menys freds...
TEMP_CANVI = 0.0 # A l'estiu les temperatures apujen cada vegada més...

# Ignorar excedències (soroll) del DEM, no aplicar més d'un metr3
TOL_M = 0.5


def get_ambient_for_point(nuc_name: str, pt: dict):
    amb = pt.get("ambient")
    if amb and all(k in amb for k in ("Ld","Le","Ln")):
        return amb
    # fallback al nivell del nucli si existeix
    pre = RECEIVERS.get(nuc_name, {}).get("pre")
    if pre and all(k in pre for k in ("Ld","Le","Ln")):
        return pre
    # últim recurs: zero
    return {"Ld":0.0, "Le":0.0, "Ln":0.0}

def _cap(x, lo, hi): 
    return max(lo, min(hi, x))

def build_profiles(set_name):
    profs = []
    for season in ["DJF","MAM","JJA","SON"]:
        T = TEMP_XEMA[season]
        if set_name == "robust":
            RH = _cap(HR_LOCUS[season] + 10.0, 25.0, 95.0)
            T  = T + TEMP_INVTERM  # opcional per minimitzar absorció
        elif set_name == "sec":
            RH = _cap(HR_LOCUS[season] - 15.0, 25.0, 95.0)
            T  = T - TEMP_CANVI  # opcional per maximitzar absorció
        else:  # central
            RH = _cap(HR_LOCUS[season], 25.0, 95.0)
        profs.append({"name": season, "T": T, "RH": RH, "P": PRESS_KPA})
    return profs

# ----------------------- PARÀMETRES PLOT i MALLA -----------------------
LOS_MAX_DIST = 3000.0 # Dibuixar LOS tant sols si distancia inferior a 1200m
GRID_STEP_M = 25.0  # 10.0 costa molt computar, per damunt de 25.0 no és suficent... (més fi, més cost)
ISOP_BOUNDS = [0,40,45,50,55,60,65,70,80]
ISOP_COLORS = ["#d0f0c0","#a8e6a3","#f0f07e","#ffd966","#f6b26b","#e06666","#a61c1c","#741b47"]
CMAP = ListedColormap(ISOP_COLORS); NORM = BoundaryNorm(ISOP_BOUNDS, CMAP.N, clip=True)

# ----------------------- DEM + HILLSHADE (nord amunt; valls clares) -----------------------
with rasterio.open(DEM_PATH) as dem:
    Z = dem.read(1).astype("float32")
    TRANS = dem.transform
    DEM_BOUNDS = dem.bounds
    XRES, YRES = dem.res

gy, gx = np.gradient(Z, -abs(YRES), abs(XRES))  # eix Y invertit de ràster → corregim
gy = -gy
slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(gy, -gx)
az = np.deg2rad(315); alt = np.deg2rad(45)
hillshade = np.sin(alt)*np.cos(slope) + np.cos(alt)*np.sin(slope)*np.cos(az - aspect)
hillshade = (hillshade - np.nanmin(hillshade)) / (np.nanmax(hillshade) - np.nanmin(hillshade) + 1e-12)
HILLSHADE = np.flipud(hillshade)  # origen “lower” → flip per nord amunt

def sample_dem(x, y):
    col = (x - TRANS.c) / TRANS.a
    row = (y - TRANS.f) / TRANS.e
    r0 = np.floor(row).astype(int); c0 = np.floor(col).astype(int)
    r1 = r0 + 1; c1 = c0 + 1
    r0c = np.clip(r0, 0, Z.shape[0]-1); r1c = np.clip(r1, 0, Z.shape[0]-1)
    c0c = np.clip(c0, 0, Z.shape[1]-1); c1c = np.clip(c1, 0, Z.shape[1]-1)
    fr = row - r0; fc = col - c0
    z00 = Z[r0c, c0c]; z10 = Z[r1c, c0c]; z01 = Z[r0c, c1c]; z11 = Z[r1c, c1c]
    z0 = z00*(1-fr) + z10*fr; z1 = z01*(1-fr) + z11*fr
    return z0*(1-fc) + z1*fc

# ----------------------- FÍSICA: difracció, sòl, absorció, meteo -----------------------
def _knife_edge_nu(h, s1, s2, f_hz):
    c_sound = 340.0; lam = c_sound / f_hz
    return h * np.sqrt(2.0/lam * (s1 + s2) / (s1*s2 + 1e-12))
def _knife_edge_loss(nu):
    return np.where(nu <= -0.78, 0.0, 6.9 + 20.0*np.log10(np.sqrt((nu - 0.1)**2 + 1.0) + nu - 0.1))
def _elev_profile(xt, yt, xr, yr, sampler, n=64):
    xs = np.linspace(xt, xr, n); ys = np.linspace(yt, yr, n)
    zs = sampler(xs, ys); dx = np.diff(xs); dy = np.diff(ys)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))]); return xs, ys, zs, s
def _los_line(hs, hr, n): return hs + (hr - hs) * np.linspace(0.0, 1.0, n)
def _find_crests(z_minus_los):
    z = z_minus_los
    if np.all(z <= 0): return []
    iP = int(np.argmax(z)); idxs=[iP]
    left=z[:iP]; right=z[iP+1:]
    if left.size:
        i2=int(np.argmax(left)); 
        if left[i2]>0: idxs.append(i2)
    if right.size:
        j2=int(np.argmax(right)); 
        if right[j2]>0: idxs.append(iP+1+j2)
    return sorted(set(idxs))

def deygout_multi_edge_diffraction(xt, yt, xr, yr, h_source, h_receiver, bands_hz, dem_sampler, n_prof=64):
    xs, ys, zs, s = _elev_profile(xt, yt, xr, yr, dem_sampler, n=n_prof)
    z_s = float(dem_sampler(np.array([xt]), np.array([yt])))
    z_r = float(dem_sampler(np.array([xr]), np.array([yr])))
    hs = z_s + h_source; hr = z_r + h_receiver
    z_los = _los_line(hs, hr, n_prof); 
    
    h_exc = zs - z_los
    
    # --- Tolerància per evitar falsos obstacles per soroll del DEM
    h_exc_tol = h_exc - TOL_M
    
    if np.all(h_exc_tol <= 0.0): return np.zeros_like(bands_hz, float)
    
    crest_ids = _find_crests(h_exc); iP = crest_ids[np.argmax(h_exc[crest_ids])]
    s_tot = s[-1]; s1P = s[iP]; s2P = s_tot - s1P; hP = h_exc[iP]
    loss_primary = np.array([_knife_edge_loss(_knife_edge_nu(hP, s1P, s2P, f)) for f in bands_hz])
    loss_secondary = np.zeros_like(loss_primary)
    if iP > 1:
        iL = int(np.argmax(h_exc[:iP]))
        if h_exc[iL] > 0:
            s1 = s[iL]; s2 = s1P - s1; h = h_exc[iL]
            loss_secondary += np.array([_knife_edge_loss(_knife_edge_nu(h, s1, s2, f)) for f in bands_hz])
    if iP < (len(h_exc)-2):
        iR = iP + 1 + int(np.argmax(h_exc[iP+1:]))
        if h_exc[iR] > 0:
            s1 = max(s1P - (s[iR] - s[iP]), 1e-3); s2 = s[-1] - s[iR]; h = h_exc[iR]
            loss_secondary += np.array([_knife_edge_loss(_knife_edge_nu(h, s1, s2, f)) for f in bands_hz])
    return np.minimum(loss_primary + loss_secondary, 30.0)

def partition_path_lengths(r3d, r2d, src_near=30.0, rec_near=30.0):
    rs = min(r2d/2.0, src_near); rr = min(r2d/2.0, rec_near); rm = max(r2d - rs - rr, 0.0)
    scale = r3d / max(r2d, 1e-6); return rs*scale, rm*scale, rr*scale

def ground_term_db(f_hz, r_m, G=0.6):
    r_eff = np.maximum(r_m, 1.0)
    base_soft = 6.0 * (1.0 - np.exp(-r_eff/400.0)) * (f_hz/1000.0)**0.15
    base_soft = np.clip(base_soft, 0.0, 8.0)
    base_hard = 0.6 * (1.0 - np.exp(-r_eff/600.0))
    return (1.0 - G) * base_hard + G * base_soft

def cnossos_ground_attenuation_per_band(r3d, r2d, bands_hz, Gs=0.0, Gm=0.6, Gr=0.8):
    rs, rm, rr = partition_path_lengths(r3d, r2d)
    out = []
    for f in bands_hz:
        out.append(ground_term_db(f, rs, Gs) + ground_term_db(f, rm, Gm) + ground_term_db(f, rr, Gr))
    return np.array(out)

def iso9613_alpha_db_per_m(f_hz, T=10.0, RH=70.0, p_kpa=93.5):
    T_k = T + 273.15; p_atm = p_kpa / 101.325; Ts = 293.15
    frO = p_atm * (24.0 + 4.04e4 * RH * (0.02 + RH) / (0.391 + RH))
    frN = p_atm * (T_k/Ts)**(-0.5) * (9.0 + 280.0*RH*np.exp(-4.17*((T_k/Ts)**(-1/3) - 1)))
    f = np.asarray(f_hz, dtype=float)
    alpha = 8.686 * f*f * (1.84e-11*(1/p_atm*np.sqrt(T_k/Ts)) +
            (T_k/Ts)**(-2.5) * (0.01275*np.exp(-2239.1/T_k)/(frO + f*f/frO) +
            0.1068*np.exp(-3352.0/T_k)/(frN + f*f/frN)))
    return alpha

def meteo_correction(r2d_vec, bands_hz, mode="favourable"):
    r = np.asarray(r2d_vec)
    if mode == "neutral":
        return np.zeros((r.size, len(bands_hz)))
    # perfil favourable moderat (base)
    band_scale = np.array([1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5])
    delta_r = 3.0 * (1.0 - np.exp(-r/800.0))
    out = np.zeros((r.size, len(bands_hz)))
    for ib in range(len(bands_hz)):
        out[:, ib] = delta_r * band_scale[ib]
    return out

# ----------------------- UTILITATS CNOSSOS -----------------------
def combine_energy(L1, L2):
    return 10*np.log10(10**(L1/10.0) + 10**(L2/10.0))

# normativa catalana horaris
def Lden_from_cat(Ld, Le, Ln):
    # 14/2/8 (Catalunya): +5 dB vespre, +10 dB nit
    num = 14*10**(Ld/10.0) + 2*10**((Le+5.0)/10.0) + 8*10**((Ln+10.0)/10.0)
    return 10*np.log10(num/24.0)

# normativata europea:
#def Lden_from_cat(Ld, Le, Ln):
#    num = 12.0*10**(Ld/10.0) + 4.0*10**((Le+5.0)/10.0) + 8.0*10**((Ln+10.0)/10.0)
#    return 10.0*np.log10(num/24.0)

def mix_nf(Lneu, Lfav, p):
    return 10*np.log10(p*10**(Lfav/10.0) + (1.0-p)*10**(Lneu/10.0) + 1e-30)

# ----------------------- MALLA del plànol -----------------------
#inv = ~TRANS
#c0,r0 = inv*(XMIN, YMAX); c1,r1 = inv*(XMAX, YMIN)
#rmin,rmax = int(np.floor(min(r0,r1))), int(np.ceil(max(r0,r1)))
#cmin,cmax = int(np.floor(min(c0,c1))), int(np.ceil(max(c0,c1)))
#step_pix = max(1, int(GRID_STEP_M/abs(XRES)))
#rows = np.arange(max(0,rmin), min(Z.shape[0], rmax), step_pix)
#cols = np.arange(max(0,cmin), min(Z.shape[1], cmax), step_pix)
#COL, ROW = np.meshgrid(cols+0.5, rows+0.5)
#a,b,c_,d,e,f_ = TRANS.a, TRANS.b, TRANS.c, TRANS.d, TRANS.e, TRANS.f
#XX = a*COL + b*ROW + c_
#YY = d*COL + e*ROW + f_
#Z_GRID = sample_dem(XX, YY)
XX = YY = Z_GRID = None     # np.memmap (read-only)
rows = cols = None          # np.ndarray petits (copiats)
step_pix = None             # int
GRID_META = None            # dict amb info memmap (compartida)

@dataclass
class PlanGrid:
    rows: np.ndarray
    cols: np.ndarray
    XX: np.ndarray
    YY: np.ndarray
    Z_GRID: np.ndarray
    step_pix: int

def build_plan_grid_obj(XMIN, YMIN, XMAX, YMAX, GRID_STEP_M, TRANS: Affine, Z: np.ndarray, sample_dem):
    inv = ~TRANS
    c0, r0 = inv * (XMIN, YMAX); c1, r1 = inv * (XMAX, YMIN)
    rmin, rmax = int(np.floor(min(r0, r1))), int(np.ceil(max(r0, r1)))
    cmin, cmax = int(np.floor(min(c0, c1))), int(np.ceil(max(c0, c1)))

    a, b, c_, d, e, f_ = TRANS.a, TRANS.b, TRANS.c, TRANS.d, TRANS.e, TRANS.f
    XRES = a
    step_pix = max(1, int(GRID_STEP_M / abs(XRES)))

    rs = np.arange(max(0, rmin), min(Z.shape[0], rmax), step_pix, dtype=int)
    cs = np.arange(max(0, cmin), min(Z.shape[1], cmax), step_pix, dtype=int)

    # mateixes fórmules que al teu codi (+0.5 centre de píxel)
    COL, ROW = np.meshgrid(cs.astype(float) + 0.5, rs.astype(float) + 0.5)
    XX = a*COL + b*ROW + c_
    YY = d*COL + e*ROW + f_
    ZG = sample_dem(XX, YY)

    return PlanGrid(rows=rs, cols=cs, XX=XX, YY=YY, Z_GRID=ZG, step_pix=step_pix)

def export_plan_to_memmap(plan: PlanGrid, dirpath=None):
    """ Escriu XX/YY/Z_GRID a memmap i retorna meta (paths+shapes+dtypes+rows/cols/step_pix). """
    if dirpath is None:
        dirpath = tempfile.mkdtemp(prefix="malla_")
    paths = {}
    def _write(name, arr):
        path = os.path.join(dirpath, f"{name}.mmap")
        mm = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
        mm[:] = arr[:]
        del mm
        paths[name] = (path, arr.shape, str(arr.dtype))
    _write("XX", plan.XX)
    _write("YY", plan.YY)
    _write("Z_GRID", plan.Z_GRID)
    # meta petit (en memòria normal)
    meta = {
        "dirpath": dirpath,
        "paths": paths,
        "rows": plan.rows,
        "cols": plan.cols,
        "step_pix": int(plan.step_pix),
    }
    return meta

def _init_grid_from_memmap(meta):
    """Initializer per a ProcessPoolExecutor: obre memmaps i inicialitza globals dins del worker."""
    global XX, YY, Z_GRID, rows, cols, step_pix, GRID_META
    GRID_META = meta
    def _read(path, shape, dtype_str):
        return np.memmap(path, dtype=np.dtype(dtype_str), mode='r', shape=tuple(shape))
    XX = _read(*meta["paths"]["XX"])
    YY = _read(*meta["paths"]["YY"])
    Z_GRID = _read(*meta["paths"]["Z_GRID"])
    rows = meta["rows"]
    cols = meta["cols"]
    step_pix = meta["step_pix"]

def make_grid_executor(max_workers):
    """Crea un ProcessPoolExecutor amb l'initializer que munta la malla en cada worker."""
    #from concurrent.futures import ProcessPoolExecutor
    assert GRID_META is not None, "GRID_META no inicialitzat: crida setup_grid_memmap() abans."
    return ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_grid_from_memmap,
        initargs=(GRID_META,)
    )

def setup_grid_memmap_after_yaml(XMIN, YMIN, XMAX, YMAX, GRID_STEP_M, TRANS, Z, sample_dem):
    """
    Construeix la malla, exporta memmaps, inicialitza globals al PROC principal.
    Retorna GRID_META (guarda-ho en un global per usar a make_grid_executor()).
    """
    global GRID_META
    plan = build_plan_grid_obj(XMIN, YMIN, XMAX, YMAX, GRID_STEP_M, TRANS, Z, sample_dem)
    GRID_META = export_plan_to_memmap(plan)
    # IMPORTANT: inicialitza també globals al procés principal reutilitzant el mateix initializer
    _init_grid_from_memmap(GRID_META)
    return GRID_META

def cleanup_grid_memmap():
    """Esborra el directori temporal del memmap (si existeix). Cridar al final del programa."""
    if GRID_META and "dirpath" in GRID_META and os.path.isdir(GRID_META["dirpath"]):
        try:
            shutil.rmtree(GRID_META["dirpath"], ignore_errors=True)
        except Exception:
            pass
# ============================================================================== 

# ----------------------- LLEGIR SUC -----------------------
SUC_GEOMS = []
def load_suc_geoms():
    with fiona.open(GML_PATH) as src:
        for feat in src:
            if not feat["geometry"]: continue
            props = feat.get("properties") or {}
            if props.get("CODI_CLAS_MUC") == "SUC":
                geom = shape(feat["geometry"])
                inter = geom.intersection(BBOX_POLY)
                if not inter.is_empty: SUC_GEOMS.append(inter)

# ----------------------- CORE: càlcul per un conjunt meteo -----------------------
def alpha_per_profile(bands_hz, prof):
    return iso9613_alpha_db_per_m(bands_hz, T=prof["T"], RH=prof["RH"], p_kpa=prof["P"])

def lp_map_per_band_with_alpha(alpha_f_loc, mode_meteo):
    Lp_sum = np.zeros(XX.shape + (len(BANDS),), float)
    for t in TURBINES:
        tname, xt, yt, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power, lwa_ovb = (
            t["id"], t["x"], t["y"], t.get("lat"), t.get("lon"),
            t.get("elev_platform_m", 0.0), t.get("hub_height_m", 112.0),
            t.get("rotor_diam_m", 172.0), t.get("solidity", 0.5),
            t.get("cut_in_ms", 3.0), t.get("cut_out_ms", 25.0),
            t.get("model", ""), t.get("rated_power_MW", 0.0),
            t.get("LwA_octave_dB", [91.8, 98.6, 102.0, 102.5, 103.4, 101.3, 92.0, 75.5])
        )
        
        t_LwA_OVB = np.array(lwa_ovb, float)  # dB (A), per banda
        
        #Lp_sum = np.zeros(XX.shape + (len(t_BANDS),), float)
        
        
        z_s = float(sample_dem(np.array([xt]), np.array([yt]))); hs = z_s + hub_h
        hr = Z_GRID + H_RECEIVER
        r2d = np.hypot(XX - xt, YY - yt); r3d = np.hypot(r2d, (hs - hr)); r3d = np.maximum(r3d, 1.0)
        Adiv = 20*np.log10(r3d) + 11.0

        # Difracció multi-aresta
        Adiff = np.zeros(XX.shape + (len(BANDS),), float)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                
                Adiff[i,j,:] = deygout_multi_edge_diffraction(
                    xt, yt, XX[i, j], YY[i, j], hub_h, H_RECEIVER, BANDS, sample_dem, n_prof=64
                )
        
        #alpha_f_loc = alpha_per_profile(t_BANDS, prof)
        # Absorció i sòl
        Aatm = np.zeros_like(Adiff)
        for ib in range(len(BANDS)): Aatm[..., ib] = alpha_f_loc[ib] * r3d

        Agr = np.zeros_like(Adiff)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                Agr[i,j,:] = cnossos_ground_attenuation_per_band(r3d[i,j], r2d[i,j], BANDS, Gs=GROUND_GS, Gm=GROUND_GM, Gr=GROUND_GR)

        # Nivell per banda sense meteo + meteo
        Lp_nb = np.zeros_like(Adiff)
        for ib, Lw in enumerate(t_LwA_OVB):
            Lp_nb[..., ib] = Lw - Adiv - Aatm[..., ib] - Adiff[..., ib] - Agr[..., ib]

        dmet = meteo_correction(r2d.reshape(-1), BANDS, mode=mode_meteo)\
               .reshape(XX.shape + (len(BANDS),))
        Lp_b = Lp_nb + dmet

        Lp_sum += 10**(Lp_b/10.0)

    return 10*np.log10(np.maximum(Lp_sum, 1e-30))

def sum_A(band_cube):
    return 10*np.log10(np.maximum(np.sum(10**(band_cube/10.0), axis=-1), 1e-30))

def combine_seasons_energy(field_by_prof, weights):
    acc = None
    for name, field in field_by_prof.items():
        w = weights.get(name, 0.0)
        if w <= 0: continue
        ener = (10**(field/10.0)) * w
        acc = ener if acc is None else acc + ener
    return 10*np.log10(np.maximum(acc, 1e-30))

def nuc_rep_xy(nuc_name):
    pts = RECEIVERS[nuc_name]["points"]
    if not pts:
        raise ValueError(f"Nucli {nuc_name} sense receptors definits.")
    return float(pts[0]["x"]), float(pts[0]["y"])

"""
def LpA_point_annual(x, y, h_rec_m, facade_db, profiles):
    def point_mode(alpha_f, mode):
        acc = 0.0
        for (tname, xt, yt) in TURBINES:
            z_s = float(sample_dem(np.array([xt]), np.array([yt])))
            z_r = float(sample_dem(np.array([x]), np.array([y])))
            hs = z_s + H_SOURCE
            hr = z_r + h_rec_m                # <-- alçada del receptor del punt
            r2d = np.hypot(x - xt, y - yt)
            r3d = np.hypot(r2d, (hs - hr)); r3d = max(r3d, 1.0)
            Adiv = 20*np.log10(r3d) + 11.0
            Adiff = deygout_multi_edge_diffraction(
                xt, yt, x, y, H_SOURCE, h_rec_m, BANDS, sample_dem, n_prof=128
            )                                  # <-- també a la difracció
            Aatm = alpha_f * r3d
            Agr  = cnossos_ground_attenuation_per_band(r3d, r2d, BANDS, Gs=GROUND_GS, Gm=GROUND_GM, Gr=GROUND_GR)
            Lp_nb = LwA_OVB - Adiv - Aatm - Adiff - Agr
            dmet = meteo_correction(np.array([r2d]), BANDS, mode=mode)[0]
            acc += np.sum(10**((Lp_nb + dmet)/10.0))
        return 10*np.log10(acc) + facade_db
    # … barreja neutral/favourable + ponderació estacional …
"""

def LpA_point_annual(x, y, h_rec_m, facade_db, profiles):
    # calcula Ld/Le/Ln anuals (turbines) al receptor (x,y) amb h_rec_m específic
    def point_mode(alpha_f, mode):
        acc = 0.0
        for t in TURBINES:
            tname, xt, yt, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power, lwa_ovb = (
                t["id"], t["x"], t["y"], t.get("lat"), t.get("lon"),
                t.get("elev_platform_m", 0.0), t.get("hub_height_m", 112.0),
                t.get("rotor_diam_m", 172.0), t.get("solidity", 0.5),
                t.get("cut_in_ms", 3.0), t.get("cut_out_ms", 25.0),
                t.get("model", ""), t.get("rated_power_MW", 0.0),
                t.get("LwA_octave_dB", [91.8, 98.6, 102.0, 102.5, 103.4, 101.3, 92.0, 75.5])
            )        
            t_LwA_OVB = np.array(lwa_ovb, float)  # dB (A), per banda
            z_s = float(sample_dem(np.array([xt]), np.array([yt])))
            z_r = float(sample_dem(np.array([x]), np.array([y])))
            hs = z_s + hub_h
            hr = z_r + h_rec_m  # <--- altura del receptor específica
            r2d = np.hypot(x - xt, y - yt)
            r3d = np.hypot(r2d, (hs - hr)); r3d = max(r3d, 1.0)
            Adiv = 20*np.log10(r3d) + 11.0
            Adiff = deygout_multi_edge_diffraction(xt, yt, x, y, hub_h, h_rec_m, BANDS, sample_dem, n_prof=128)
            Aatm = alpha_f * r3d
            Agr  = cnossos_ground_attenuation_per_band(r3d, r2d, BANDS, Gs=GROUND_GS, Gm=GROUND_GM, Gr=GROUND_GR)
            Lp_nb = t_LwA_OVB - Adiv - Aatm - Adiff - Agr
            dmet = meteo_correction(np.array([r2d]), BANDS, mode=mode)[0]
            Lp_b = Lp_nb + dmet
            acc += np.sum(10**(Lp_b/10.0))
        return 10*np.log10(acc) + facade_db  # <--- +3 dB si façana

    Ld_prof, Le_prof, Ln_prof = {}, {}, {}
    for prof in profiles:
        alpha_f = iso9613_alpha_db_per_m(BANDS, T=prof["T"], RH=prof["RH"], p_kpa=prof["P"])
        L_neu = point_mode(alpha_f, "neutral")
        L_fav = point_mode(alpha_f, "favourable")
        Ld_prof[prof["name"]] = mix_nf(L_neu, L_fav, P_FAV["Ld"])
        Le_prof[prof["name"]] = mix_nf(L_neu, L_fav, P_FAV["Le"])
        Ln_prof[prof["name"]] = mix_nf(L_neu, L_fav, P_FAV["Ln"])

    Ld_ann = 10*np.log10(sum(W_SEASON["Ld"][k]*10**(Ld_prof[k]/10.0) for k in Ld_prof))
    Le_ann = 10*np.log10(sum(W_SEASON["Le"][k]*10**(Le_prof[k]/10.0) for k in Le_prof))
    Ln_ann = 10*np.log10(sum(W_SEASON["Ln"][k]*10**(Ln_prof[k]/10.0) for k in Ln_prof))
    return Ld_ann, Le_ann, Ln_ann


#import matplotlib.pyplot as plt
#from matplotlib.patches import Patch
#import csv

# Guarda dades de compropació los en csv
def save_los_results_to_csv(res, filename="LOS_resultats.csv"):
    """
    Desa la llista de dicts retornada per check_los_multi(...) a un CSV.
    """
    if not res:
        print("No hi ha resultats LOS per desar.")
        return
    fields = ["turbina","receptor_id","x","y","dist_m","h_max_exced_m","tolerance_m","LOS_clara"]
    result_path = os.path.join(OUTPUT_DIR, filename)        

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        w.writeheader()
        for row in res:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"Desat CSV: {filename}")

# mostra los sense o amb obstacles entre emissor i receptor
def plot_los_on_map(res, ax, color_ok="#2ca02c", color_bad="#d62728", linestyle="-", lw=1.0, alpha=0.6):
    """
    Sobreposa segments (turbina→receptor) en verd si LOS_clara=True, vermell si False.
    Requereix globals TURBINES ([(name,x,y),...]).
    """
    if not res:
        return

    # index de turbines per nom
    #tdict = {nm: (x, y) for (nm, x, y, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power) in TURBINES}
    tdict = {t["id"]: (t["x"], t["y"]) for t in TURBINES}
    ok_count = 0
    bad_count = 0
    for r in res:
        tname = r["turbina"]
        if tname not in tdict:
            continue
        tx, ty = tdict[tname]
        rx, ry = float(r["x"]), float(r["y"])
        r2d = float(np.hypot(rx - tx, ry - ty))
        good = bool(r.get("LOS_clara", False))
        # dist = float(r.get("dist_m"), LOS_MAX_DIST)
        if r2d < LOS_MAX_DIST:
            """ no mostrem, massa soroll al mapa...
            ax.plot([tx, rx], [ty, ry],
                    color=(color_ok if good else color_bad),
                    linewidth=lw, alpha=alpha, linestyle=linestyle, solid_capstyle="round")

            # opcional: punts
            ax.scatter([rx], [ry], s=18, color=(color_ok if good else color_bad), zorder=3)
            """
        ok_count += int(good)
        bad_count += int(not good)

    # llegenda simple
    handles = [
        Patch(facecolor=color_ok, edgecolor="none", label=f"LOS clara ({ok_count})", alpha=alpha),
        Patch(facecolor=color_bad, edgecolor="none", label=f"LOS defracció ({bad_count})", alpha=alpha),
    ]
    ax.legend(handles=handles, title=f"Línia de visió (LOS)", loc="lower right", frameon=True, framealpha=0.9, fontsize=9)

# snippet per comprovar obstacles entre buc i receptor per un aerogenerador i un receptor en concret
# Exemples: 
# res = check_los_multi("Belltall") dona los per tots els aerogeneradors i tots els receptors per Belltall
# res = check_los_multi("Belltall", rec_id="BEL-1") dona los per receptor amb id BEL-1 i totes les turbines
# res = check_los_multi("Glorieta", tnames=["YA3","Y07"], tol_m=1.0, n_prof_per_km=60) dona sol per tots els receptors de Glorieta pels aerogeneradors especificats, marge de tolerància sobre DEM de 1m i densidad més alta de punts de comprobació

# ----------------------- NUMBA LOS FAST-PATH -----------------------
# Enable with USE_NUMBA: true in YAML.
try:
    from numba import njit
except Exception:
    njit = None

# DEM grid and inverse affine transform coefficients
DEM_ARRAY = None
INV_A0 = INV_A1 = INV_A2 = INV_A3 = INV_A4 = INV_A5 = None

def _load_dem_to_memory():
    """Load DEM into memory and store inverse affine parameters for fast XY->RC mapping."""
    global DEM_ARRAY, INV_A0, INV_A1, INV_A2, INV_A3, INV_A4, INV_A5
    if DEM_ARRAY is not None:
        return
    with rasterio.open(DEM_PATH) as ds:
        DEM_ARRAY = ds.read(1)  # assume band 1
        inv = ~ds.transform  # inverse Affine: (col,row) = inv * (x,y)
        INV_A0, INV_A1, INV_A2, INV_A3, INV_A4, INV_A5 = inv.a, inv.b, inv.c, inv.d, inv.e, inv.f

def _xy_to_rowcol_fast(x, y):
    """Use inverse affine to map projected coords to (row, col) indices (nearest)."""
    col = INV_A0 * x + INV_A1 * y + INV_A2
    row = INV_A3 * x + INV_A4 * y + INV_A5
    return int(round(row)), int(round(col))

# Numba kernels (if available)
if njit is not None:
    @njit
    def _in_bounds(arr, r, c):
        return (r >= 0) and (c >= 0) and (r < arr.shape[0]) and (c < arr.shape[1])

    @njit
    def _los_max_exceed_numba(DEM, row0, col0, z0, row1, col1, z1, max_steps=4096):
        """Returns maximum (terrain - line) along the segment. >0 means blocked."""
        dr = row1 - row0
        dc = col1 - col0
        steps = int(max(abs(dr), abs(dc)))
        if steps < 1:
            steps = 1
        if steps > max_steps:
            steps = max_steps
        stepr = dr / steps
        stepc = dc / steps
        max_exceed = -1e9
        for k in range(1, steps):
            rr = int(round(row0 + stepr * k))
            cc = int(round(col0 + stepc * k))
            if not _in_bounds(DEM, rr, cc):
                continue
            z_terr = DEM[rr, cc]
            # Linear interpolation of the line height
            z_line = z0 + (z1 - z0) * (k / steps)
            diff = z_terr - z_line
            if diff > max_exceed:
                max_exceed = diff
        if max_exceed == -1e9:
            max_exceed = -1e9  # no samples, treat as clear
        return max_exceed

def check_los_multi_numba(nuc_name: str, rec_id: str | None = None,
                          h_rec_override: float | None = None,
                          tol_m: float = 0.5, n_prof_per_km: int = 40):
    """
    Numba fast-path LOS check:
      - Loads DEM to memory once.
      - For each (WT, receptor) computes max exceedance (terrain - line).
      - Returns list of dicts: {'turbina','receptor','dist_m','h_max_exced_m','LOS_clara'}.
    """
    if njit is None:
        raise RuntimeError("Numba no disponible. Instal·la 'numba' o desactiva USE_NUMBA.")
    _load_dem_to_memory()
    # Build turbine list (id, x, y, hub height, platform elev)
    tlist = []
    for t in TURBINES:
        if isinstance(t, dict):
            nm = t.get("id") or t.get("name") or "WT"
            tx, ty = t["x"], t["y"]
            hub = float(t.get("hub_height_m", 112.0))
            plat = float(t.get("elev_platform_m", 0.0))
        else:
            # Fallback: tuple legacy (nm, x, y, ..., hub, ...)
            nm, tx, ty = t[0], t[1], t[2]
            hub = float(t[6]) if len(t) > 6 else 112.0
            plat = float(t[5]) if len(t) > 5 else 0.0
        tlist.append((nm, tx, ty, hub, plat))

    if nuc_name not in RECEIVERS:
        raise KeyError(f"Nucli '{nuc_name}' no trobat a RECEIVERS")
    points = RECEIVERS[nuc_name].get("points", [])
    # select subset
    pts = [p for p in points if (rec_id is None or p.get("id") == rec_id)]

    out = []
    for p in pts:
        rx, ry = float(p["x"]), float(p["y"])
        h_rec = float(h_rec_override if h_rec_override is not None else p.get("h_rec", p.get("h_mic", 4.0)))
        # Elevations at endpoints from DEM
        r0, c0 = _xy_to_rowcol_fast(rx, ry)
        # receiver terrain elevation:
        zterr_rec = float(DEM_ARRAY[r0, c0]) if 0 <= r0 < DEM_ARRAY.shape[0] and 0 <= c0 < DEM_ARRAY.shape[1] else 0.0
        z_rec = zterr_rec + h_rec
        for (nm, tx, ty, hub, plat) in tlist:
            r1, c1 = _xy_to_rowcol_fast(tx, ty)
            zterr_src = float(DEM_ARRAY[r1, c1]) if 0 <= r1 < DEM_ARRAY.shape[0] and 0 <= c1 < DEM_ARRAY.shape[1] else 0.0
            z_src = zterr_src + plat + hub
            max_ex = _los_max_exceed_numba(DEM_ARRAY, r1, c1, z_src, r0, c0, z_rec)
            dist_m = ((tx - rx)**2 + (ty - ry)**2) ** 0.5
            los_clear = (max_ex <= tol_m)
            out.append({
                "turbina": nm,
                "receptor_id": p.get("id"),
                "x": rx, "y": ry,
                "dist_m": float(dist_m),
                "h_max_exced_m": float(max_ex if max_ex > -1e8 else -9999.0),
                "LOS_clara": bool(los_clear),
                "tolerance_m": tol_m,
            })
            
    return out



def check_los_multi_orig(
    nuc_name: str,
    rec_id: str | None = None,   # si None → tots els receptors del nucli
    #tnames: list[str] | None = None,  # si None → totes les turbines
    h_rec_override: float | None = None,  # si vols provar una alçada diferent temporalment
    tol_m: float = 0.5,           # tolerància d'excedència del DEM (m)
    n_prof_per_km: int = 40       # densitat del perfil (p. ex. 40 punts/km ≈ 1 punt/25 m)
):
    """
    Retorna una llista de dicts amb la comprovació LOS per a cada (turbina, receptor) seleccionat.
    Cada dict inclou: turbina, receptor, dist_m, h_max_exced_m, LOS_clara (True/False).
    """
    # --- receptors del nucli
    if nuc_name not in RECEIVERS:
        raise ValueError(f"Nucli {nuc_name!r} no definit a RECEIVERS")

    pts_all = RECEIVERS[nuc_name].get("points", [])
    if not pts_all:
        raise ValueError(f"Nucli {nuc_name!r} sense receptors")

    if rec_id is not None:
        # filtra per ID concret
        pts = [p for p in pts_all if str(p.get("id")) == str(rec_id)]
        if not pts:
            raise ValueError(f"Receptor {rec_id!r} no trobat al nucli {nuc_name!r}")
    else:
        pts = pts_all  # tots

    out = []
    for t in TURBINES: 
        tname, tx, ty, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power = (
            t["id"], t["x"], t["y"], t.get("lat"), t.get("lon"),
            t.get("elev_platform_m", 0.0), t.get("hub_height_m", 112.0),
            t.get("rotor_diam_m", 172.0), t.get("solidity", 0.5),
            t.get("cut_in_ms", 3.0), t.get("cut_out_ms", 25.0),
            t.get("model", ""), t.get("rated_power_MW", 0.0)
        )            
        
    #for (tname, tx, ty) in tlist:
        # cota a la turbina
        z_s = float(sample_dem(np.array([tx]), np.array([ty])))
        hs  = z_s + hub_h

        for p in pts:
            rx, ry = float(p["x"]), float(p["y"])
            # alçada del receptor (pot ser sobreescrita)
            h_rec = float(p.get("h_rec", 4.0)) if h_rec_override is None else float(h_rec_override)

            # perfil
            z_r = float(sample_dem(np.array([rx]), np.array([ry])))
            hr  = z_r + h_rec
            r2d = float(np.hypot(rx - tx, ry - ty))
            n_prof = max(64, int((r2d/1000.0) * n_prof_per_km))  # escalar amb distància

            xs = np.linspace(tx, rx, n_prof)
            ys = np.linspace(ty, ry, n_prof)
            zs = sample_dem(xs, ys)

            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
            if s[-1] <= 0:
                out.append({
                    "turbina": tname, "receptor_id": p.get("id"),
                    "x": rx, "y": ry, "dist_m": 0.0,
                    "h_max_exced_m": 0.0, "tolerance_m": tol_m, "LOS_clara": True
                })
                continue

            z_los = hs + (hr - hs)*(s/s[-1])
            exced = zs - z_los
            h_max = float(np.max(exced))

            clear = (h_max <= tol_m)
            out.append({
                "turbina": tname,
                "receptor_id": p.get("id"),
                "x": rx, "y": ry,
                "dist_m": r2d,
                "h_max_exced_m": h_max,
                "tolerance_m": tol_m,
                "LOS_clara": bool(clear)
            })
    return out


# Dispatcher: choose Numba fast path if enabled; else original
def check_los_multi(nuc_name: str,
                    rec_id: str | None = None,
                    h_rec_override: float | None = None,
                    tol_m: float = 0.5,
                    n_prof_per_km: int = 40):
    if bool(globals().get("USE_NUMBA", False)):
        return check_los_multi_numba(nuc_name, rec_id, h_rec_override, tol_m, n_prof_per_km)
    return check_los_multi_orig(nuc_name, rec_id, h_rec_override, tol_m, n_prof_per_km)
    

# ----------------------- EXECUCIÓ per cada JOC METEO -----------------------

# ---------- Top-level worker for parallel profile computation ----------
def compute_maps_for_profile__worker(args, compute_args):
    """args = (prof, bands)"""
    cfg = load_config_yaml(args.config)
    apply_yaml_overrides(cfg)
    
    prof = compute_args
    alpha_f = alpha_per_profile(BANDS, prof)
    neu = lp_map_per_band_with_alpha(alpha_f, "neutral")
    fav = lp_map_per_band_with_alpha(alpha_f, "favourable")
    #print(fav)
    #print(prof["name"])
    return prof["name"], neu, fav


# ---------- Multicore per-punt (receptors) ----------
def _comb_energy_db(a, b):
    # suma energètica en dB
    return 10*np.log10(10**(a/10.0) + 10**(b/10.0))

def compute_point_row__worker(args, compute_args):
    """
    args: (nuc_name, pt_dict, profiles)
    Retorna (nuc_name, row, Lden_total)
    """
    #args = parser.parse_args()
    cfg = load_config_yaml(args.config)
    apply_yaml_overrides(cfg)
    
    nuc_name, pt, profiles = compute_args
    x, y = float(pt["x"]), float(pt["y"])
    h_rec = float(pt.get("h_rec", pt.get("h_mic", 4.0)))
    f_db  = float(pt.get("facade_db", 3.0))

    # 1) turbines anuals al punt (Ld/Le/Ln)
    Ld_t, Le_t, Ln_t = LpA_point_annual(x, y, h_rec, f_db, profiles)

    # 2) ambient del punt (per-receptor)
    pre = get_ambient_for_point(nuc_name, pt)

    # 3) combinació energètica
    Ld_T = _comb_energy_db(Ld_t, pre["Ld"])
    Le_T = _comb_energy_db(Le_t, pre["Le"])
    Ln_T = _comb_energy_db(Ln_t, pre["Ln"])

    # 4) Lden (turb/ambient/total)
    Lden_t = Lden_from_cat(Ld_t, Le_t, Ln_t)
    Lden_p = Lden_from_cat(pre["Ld"], pre["Le"], pre["Ln"])
    Lden_T = Lden_from_cat(Ld_T, Le_T, Ln_T)

    row = [nuc_name, pt["id"], f"{h_rec:.1f}",
           round(Ld_t,1), round(pre["Ld"],1), round(Ld_T,1),
           round(Le_t,1), round(pre["Le"],1), round(Le_T,1),
           round(Ln_t,1), round(pre["Ln"],1), round(Ln_T,1),
           round(Lden_t,1), round(Lden_p,1), round(Lden_T,1),
           x, y]
    return nuc_name, row, Lden_T

def compute_all_points_multicore(args, profiles):
    """
    Construeix items (nuc, pt, profiles) i els processa en paral·lel amb parallel_map().
    Retorna (rows, metric_suc) on metric_suc[nuc] = pitjor Lden_total.
    """
    use_mp = bool(globals().get("USE_MULTIPROC", True))
    # Llista d'items per a tots els punts de tots els nuclis
    items = []
    for nuc_name, recinfo in RECEIVERS.items():
        for pt in recinfo.get("points", []):
            items.append((nuc_name, pt, profiles))

    results = parallel_map(compute_point_row__worker, args, items, use_mp=use_mp)

    rows = []
    metric_suc = {}
    for nuc_name, row, lden_T in results:
        rows.append(row)
        prev = metric_suc.get(nuc_name, None)
        if prev is None or lden_T > prev:
            metric_suc[nuc_name] = lden_T
    return rows, metric_suc

# ---------- Check for available processing cores ----------
def get_optimal_workers():
    """
    Retorna un nombre de workers òptim segons la màquina:
    - Apple Silicon (M1/M2): limita a 4 (nuclis performance).
    - Altres: num. nuclis físics, com a mínim 2.
    """
    system = platform.system()
    machine = platform.machine().lower()
    cpu_count = mp.cpu_count()

    if system == "Darwin" and ("arm" in machine or "apple" in machine):
        return min(4, cpu_count)
    else:
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
            if physical:
                return max(2, physical)
        except ImportError:
            pass
        return max(2, cpu_count)

def resolve_num_workers():
    """Respecta el YAML: NUM_CORES 'auto' o int; si no, usa get_optimal_workers()."""
    v = globals().get("NUM_CORES", "auto")
    if isinstance(v, str) and v.lower() == "auto":
        return get_optimal_workers()
    try:
        n = int(v)
    except Exception:
        return get_optimal_workers()
    n = max(2, n)
    n = min(n, mp.cpu_count())
    return n

def parallel_map(func, args, items, use_mp=True):
    """Executa func(item) en paral·lel si use_mp i hi ha més d'1 worker."""
    workers = resolve_num_workers()
    if (not use_mp) or workers <= 1 or len(items) <= 1:
        return [func(it) for it in items]
    
    cfg = load_config_yaml(args.config)
    apply_yaml_overrides(cfg)
    print(args)
    with make_grid_executor(max_workers=workers) as ex:
        futs = [ex.submit(func, args, it) for it in items]
        return [f.result() for f in futs]    
        
    #with ProcessPoolExecutor(max_workers=workers) as ex:
    #    futs = [ex.submit(func, args, it) for it in items]
        # conserva l'ordre d'entrada
    #    return [f.result() for f in futs]

def run_set(args, suffix):
    # 0) DISTÀNCIES mínimes (cercle rotor → SUC més proper del nucli)
    suc_tree = STRtree(SUC_GEOMS)

    # For each nucleus, find the SUC polygon that is closest to ANY of its receivers
    NUC_CLOSEST_SUC = {}    
    MAX_SUC_DIST_M = 500.0  # or 1500 / 2000, as you prefer
    
    for nuc, data in RECEIVERS.items():
        pts = data.get("points", [])
        if not pts:
            continue

        # Precompute a simple "nucleus centroid" for fallback
        xs = [float(p["x"]) for p in pts]
        ys = [float(p["y"]) for p in pts]
        nuc_centroid = Point(sum(xs) / len(xs), sum(ys) / len(ys))

        best_poly = None
        best_dp   = math.inf

        for pinfo in pts:
            rx = float(pinfo["x"])
            ry = float(pinfo["y"])
            pt = Point(rx, ry)

            best_local_poly = None
            best_local_dp   = math.inf

            for poly in SUC_GEOMS:
                if not isinstance(poly, BaseGeometry):
                    continue
                dpt = pt.distance(poly)
                if dpt < best_local_dp:
                    best_local_dp = dpt
                    best_local_poly = poly

            if best_local_poly is None:
                continue

            if best_local_dp < best_dp:
                best_dp   = best_local_dp
                best_poly = best_local_poly

        if best_poly is not None and best_dp <= MAX_SUC_DIST_M:
            # Valid SUC found reasonably near the receivers of this nucleus
            NUC_CLOSEST_SUC[nuc] = best_poly
        else:
            # Fallback: create a tiny "synthetic SUC" around the nucleus centroid
            # (this basically reduces to rotor ↔ nucleus distance if no SUC is near)
            NUC_CLOSEST_SUC[nuc] = nuc_centroid.buffer(50.0)  # 10.0 m radius

            
            
    dist_rows = []
    for t in TURBINES:
        tname, xt, yt, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power = (
            t["id"], t["x"], t["y"], t.get("lat"), t.get("lon"),
            t.get("elev_platform_m", 0.0), t.get("hub_height_m", 112.0),
            t.get("rotor_diam_m", 172.0), t.get("solidity", 0.5),
            t.get("cut_in_ms", 3.0), t.get("cut_out_ms", 25.0),
            t.get("model", ""), t.get("rated_power_MW", 0.0)
        )

        circ = Point(xt, yt).buffer(rotor_d / 2.0, 128)

        best_nuc = None
        best_dd  = math.inf

        for nuc, suc_poly in NUC_CLOSEST_SUC.items():
            dd = circ.distance(suc_poly)
            if dd < best_dd:
                best_dd  = dd
                best_nuc = nuc

        if best_nuc is not None:
            dist_rows.append((tname, best_nuc, int(round(best_dd))))
                        
        # --- ordena per distància ---
        dist_rows.sort(key=lambda x: x[2])
        
        """
        best_d, best_n = None, None
        for nuc in RECEIVERS.keys():
            #print(nuc)
            rx, ry = nuc_rep_xy(nuc)
            pt = Point(rx, ry)
            best_poly, best_dp = None, 1e12
            for poly in SUC_GEOMS:
                dpt = pt.distance(poly)
                if dpt < best_dp:
                    best_poly, best_dp = poly, dpt
            if best_poly is not None:
                dd = circ.distance(best_poly)
                if best_d is None or dd < best_d:
                    best_d, best_n = dd, nuc
                    print(nuc)
        if best_d is not None:
            #print(tname)
            #print(best_n)
            dist_rows.append((tname, best_n, int(round(best_d))))
        """
    
    
    set_name = args.scenario
    profiles = build_profiles(set_name)
    
    # 1) MAPES per perfil i mode meteo → anual (per Ld/Le/Ln) → Lden (turbines)
    use_mp = bool(globals().get("USE_MULTIPROC", True))

    results = parallel_map(compute_maps_for_profile__worker, args, profiles, use_mp=use_mp)    
    Lp_neu_by_prof, Lp_fav_by_prof = {}, {}
    for name, neu, fav in results:
        Lp_neu_by_prof[name] = neu
        Lp_fav_by_prof[name] = fav    
        
    Ld_prof = {}
    Le_prof = {}
    Ln_prof = {}
    for k in Lp_neu_by_prof.keys():
        Ld_neu = sum_A(Lp_neu_by_prof[k]); Ld_fav = sum_A(Lp_fav_by_prof[k])
        # emissió idèntica pels 3 períodes → base comuna
        Ld_prof[k] = mix_nf(Ld_neu, Ld_fav, P_FAV["Ld"])
        Le_prof[k] = mix_nf(Ld_neu, Ld_fav, P_FAV["Le"])
        Ln_prof[k] = mix_nf(Ld_neu, Ld_fav, P_FAV["Ln"])

    Ld_ann = combine_seasons_energy(Ld_prof, W_SEASON["Ld"])
    Le_ann = combine_seasons_energy(Le_prof, W_SEASON["Le"])
    Ln_ann = combine_seasons_energy(Ln_prof, W_SEASON["Ln"])
    Lden_turb_ann = Lden_from_cat(Ld_ann, Le_ann, Ln_ann)

    # 2) RECEPTORS (turbines anuals + preexistent) i SUC per acolorir

    rows, metric_suc = compute_all_points_multicore(args, profiles)
    #for nuc_name, recinfo in RECEIVERS.items():    
    #    if rows:
    #        worst_ldent = max(r[14] for r in rows)  # col 14 = Lden_tot
    #        metric_suc[nuc_name] = worst_ldent
    

    # 4) EXPORT CSV
    csv_name = f"TAULA_BASE_PRO_60_50_CAT_ANNUAL_{suffix}.csv"
    result_path = os.path.join(OUTPUT_DIR, csv_name)        

    with open(result_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv, delimiter=";")
        w.writerow(["Nucli","ReceptorID","h_rec(m)",
                    "Ld_turb","Ld_pre","Ld_tot","Le_turb","Le_pre","Le_tot",
                    "Ln_turb","Ln_pre","Ln_tot","Lden_turb","Lden_pre","Lden_tot","X","Y"])
        for r in rows: w.writerow(r)
    

    # 5) PLOT
    def color_from_palette(val):
        for i in range(len(ISOP_BOUNDS)-1):
            if ISOP_BOUNDS[i] <= val < ISOP_BOUNDS[i+1]: return ISOP_COLORS[i]
        return ISOP_COLORS[-1]

    # Figure with GRID
    fig = plt.figure(figsize=(13,15), layout="constrained")
    fig.suptitle(f"CNOSSOS ANUAL — ΣLpA mapa (8 bandes) — Límits (dia/vespre/nit): {active_limits_key} — SUC acolorits per Lden total", fontsize=13)

    gs = GridSpec(2, 2, figure=fig, width_ratios=[4, 12], height_ratios=[3.5, 2.2])
    # ax_legmap = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[0,:])
    ax_legex = fig.add_subplot(gs[1, :-1])
    ax_tab = fig.add_subplot(gs[1:, -1])

    ax.imshow(HILLSHADE, cmap="gray",
              extent=(DEM_BOUNDS.left, DEM_BOUNDS.right, DEM_BOUNDS.bottom, DEM_BOUNDS.top),
              origin="lower")

    im = ax.pcolormesh(XX, YY, Lden_turb_ann, cmap=CMAP, norm=NORM, shading="auto", alpha=0.6)
    cs = ax.contour(XX, YY, Lden_turb_ann, levels=[45,50,55], colors=["#006400","#FF8C00","#B22222"], linewidths=1.3)
    ax.clabel(cs, fmt=lambda v: f"{int(v)} dB", fontsize=8)

    # SUC acolorits pel Lden total (turb + preexistent)
    #for nuc, suc_poly in NUC_CLOSEST_SUC.items():
    """
    for nuc in RECEIVERS.keys():
        rx, ry = nuc_rep_xy(nuc)
        pt = Point(rx, ry)
        best_poly, best_d = None, 1e12
        for poly in SUC_GEOMS:
            d = pt.distance(poly)
            if d < best_d:
                best_poly, best_d = poly, d
        if best_poly:
            geoms = best_poly.geoms if isinstance(best_poly, MultiPolygon) else [best_poly]
            col = color_from_palette(metric_suc[nuc])  # metric_suc l’has calculat abans
            patches = [MplPolygon(np.asarray(g.exterior.coords)) for g in geoms]
            pc = PatchCollection(patches, facecolor=col, edgecolor="black", alpha=0.85, linewidths=0.8)
            ax.add_collection(pc)
            ax.text(rx+80, ry+50, f"{nuc}", fontsize=9, color="#1f77b4", weight="bold")
    """
    for nuc, best_poly in NUC_CLOSEST_SUC.items():
        rx, ry = nuc_rep_xy(nuc)
        geoms = best_poly.geoms if isinstance(best_poly, MultiPolygon) else [best_poly]

        val = metric_suc.get(nuc, 0.0)
        col = color_from_palette(val)

        patches = [MplPolygon(np.asarray(g.exterior.coords)) for g in geoms]
        pc = PatchCollection(
            patches,
            facecolor=col,
            edgecolor="black",
            alpha=0.85,
            linewidths=0.8,
        )
        ax.add_collection(pc)

        ax.text(rx + 80, ry + 50, f"{nuc}", fontsize=9,
                color="#1f77b4", weight="bold")    

    # Turbines: cercle Ø172 m + creu + nom
    for t in TURBINES:
        tname, x, y, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power = (
            t["id"], t["x"], t["y"], t.get("lat"), t.get("lon"),
            t.get("elev_platform_m", 0.0), t.get("hub_height_m", 112.0),
            t.get("rotor_diam_m", 172.0), t.get("solidity", 0.5),
            t.get("cut_in_ms", 3.0), t.get("cut_out_ms", 25.0),
            t.get("model", ""), t.get("rated_power_MW", 0.0)
        )                    
        radius = rotor_d / 2
        circ = plt.Circle((x, y), radius, fill=False, color="black"); ax.add_patch(circ)
        circ2 = plt.Circle((x, y), RADIUS_NORM_CAT, fill=False, color="violet", linestyle="--"); ax.add_patch(circ2)
        circ3 = plt.Circle((x, y), RADIUS_NORM_ESP, fill=False, color="blue", linestyle="--"); ax.add_patch(circ3)
        ax.scatter([x], [y], marker="+", s=85, color="#c81e1e", linewidths=1.8)
        ax.text(x+95, y-15, tname, fontsize=9, color="#c81e1e", weight="bold")

    # llegenda per norm distances
    handles = [
        Patch(facecolor="violet", edgecolor="none", label="500m"),
        Patch(facecolor="blue", edgecolor="none", label="1000m"),
    ]
    norm_dist = ax.legend(handles=handles, title="Distància mínima CAT/ESP", loc="lower left", frameon=True, framealpha=0.9, fontsize=9)
    ax.add_artist(norm_dist)
    
    # punts de receptors (tots)
    for r in rows:
        nuc_name, rec_id = r[0], r[1]
        x, y, Lden_tot = float(r[15]), float(r[16]), float(r[14])
        # ax.scatter([x], [y], marker="o", s=40, color="#1f77b4")
        # ax.text(x+60, y+36, f"{nuc_name} - {rec_id}: Lden {Lden_tot:.1f}dB", fontsize=8.5, color="#1f77b4", weight="bold")
        # ax.text(x+60, y+36, f"{nuc_name}", fontsize=9, color="#1f77b4", weight="bold")
            
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.038)
    cbar.set_label(f"Lden (dB) — anual CNOSSOS — joc meteo: {set_name.upper()}")


    # mostra los per tots els receptors
    res_all = []
    for nuc in RECEIVERS.keys():
        res_all.extend(check_los_multi_numba(nuc, rec_id=None, tol_m=TOL_M, n_prof_per_km=N_PROF_PER_KM))
    save_los_results_to_csv(res_all, filename="LOS_TOTS_ELS_RECEPTORS.csv")
    plot_los_on_map(res_all, ax)
    
    
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.set_aspect("equal")
    ax.set_xlabel("UTM X (m)"); ax.set_ylabel("UTM Y (m)")
    # ax.set_title(f"CNOSSOS ANUAL — ΣLpA mapa (8 bandes) — Límits (dia/vespre/nit): {active_limits_key} — SUC acolorits per Lden total", 
    # fontsize=13)

    # --- Taula al peu (monospace) adaptada a rows = [Nucli, ReceptorID, h_rec, ... , Lden_tot, X, Y]
    def _fmt1(x):
        try:
            return f"{float(x):.1f}"
        except Exception:
            return str(x)

    def _marge(x, norm):
        if x >= norm:
            return "red"
        if norm - x < 3.0:
            return "orange"        
        return "black"
        
    header = (
        "Nucli/Receptor(h) | Ld_turb Ld_pre Ld_tot | "
        "Le_turb Le_pre Le_tot | Ln_turb Ln_pre Ln_tot | "
        "Lden_turb Lden_pre Lden_tot"
    )

    ax_tab.axis("off")
    ax_tab.text(
        0.0, 1,
        header + "\n",
        ha="left", va="top", family="monospace", fontsize=8
        #bbox=dict(facecolor="white", alpha=0.95, edgecolor="gray")
    )
    #ax_tab.annotate(
    #    "\n", xycoords=results, xy=(0, 0.7), verticalalignment="bottom")  # custom 
    
    rowy = 0.98
    lines = []
    for r in rows:
        # r index map:
        # 0 Nucli | 1 ReceptorID | 2 h_rec(m) | 3..5 Ld_* | 6..8 Le_* | 9..11 Ln_* | 12..14 Lden_* | 15 X | 16 Y
        line = (
            f"{r[0]}/{r[1]}({_fmt1(r[2])}m) | "
            f"{_fmt1(r[3])} {_fmt1(r[4])} {_fmt1(r[5])} | "
            f"{_fmt1(r[6])} {_fmt1(r[7])} {_fmt1(r[8])} | "
            f"{_fmt1(r[9])} {_fmt1(r[10])} {_fmt1(r[11])} | "
            f"{_fmt1(r[12])} {_fmt1(r[13])} {_fmt1(r[14])}"
        )        
        lines.append(line)
        results = ax_tab.text(0.0, rowy,
            f"\n{r[0]}/{r[1]} ({_fmt1(r[2])}m) | {_fmt1(r[3])} {_fmt1(r[4])}",
                             ha="left", va="top", family="monospace", fontsize=8)  # custom 
        rowy -= 0.035
        
        colorstr = _marge(r[5], LIMITS[0])
        results = ax_tab.annotate(
            " Dia: " + f"{r[5]}", xycoords=results, xy=(1, 0.0), verticalalignment="bottom", color=f"{colorstr}", weight="bold")  # 

        results = ax_tab.annotate(
            f" | {r[6]} {r[7]}", xycoords=results, xy=(1, 0.0), verticalalignment="bottom")  #         
        colorstr = _marge(r[8], LIMITS[1])
        results = ax_tab.annotate(
            " Vespre: " + f"{r[8]}", xycoords=results, xy=(1, 0.0), verticalalignment="bottom", color=f"{colorstr}", weight="bold")  # 
        
        results = ax_tab.annotate(
            f" | {r[9]} {r[10]}", xycoords=results, xy=(1, 0.0), verticalalignment="bottom")  #         
        colorstr = _marge(r[11], LIMITS[2])
        results = ax_tab.annotate(
            " Nit: " + f"{r[11]}", xycoords=results, xy=(1, 0), verticalalignment="bottom", color=f"{colorstr}", weight="bold") 
        
        results = ax_tab.annotate(
            f" | {r[12]} {r[13]}", xycoords=results, xy=(1, 0), verticalalignment="bottom")  #         
        results = ax_tab.annotate(
            f" {r[14]}", xycoords=results, xy=(1, 0), verticalalignment="bottom", color="blue")  #         

        # Comprovar obstacles entre emissors i receptors segons terreny (DEM), surt a la consola 
        # print(check_los_multi(r[0], r[1]))
    
    rowy -= 0.05
    ax_tab.text(0.0, rowy,
        "Vermell: no permesos Catalunya. Taronja: sense marge best practice >= 3dB.\n"
        f"*Anual CNOSSOS: barreja neutral/favourable amb p_fav Ld={P_FAV['Ld']:.2f}, Le={P_FAV['Le']:.2f}, Ln={P_FAV['Ln']:.2f}.\n"
        f"Perfils meteo {set_name}: T/HR/Pressió estacional (XEMA Sta. Coloma + bellcam HR).\n"
        f"Absorció ISO9613-1; Deygout multi-aresta; Gs/Gm/Gr={GROUND_GS}/{GROUND_GM}/{GROUND_GR}; receptors a façana +3 dB.\n"
        "En la fórmula de CNOSSOS/Directiva 2002/49/CE per Lden s’hi afegeixen penalitzacions subjectives.\n"
        "Lden: 14h dia (Ld), 2h vespre (Le +5 dB), 8h nit (Ln +10 dB) — normativa EU, horaris cat."
        f"CNOSSOS-EU Assessment, version: {VERSION}",
        ha="left", va="top", fontsize=8, family="monospace")    
    
    """
    ax_legex.axis("off")
    ax_legex.text(0.0, 0.08,
        "Valors vermells: no permesos segons normativa catalana.\n"
        "Valors taronja: sense marge de seguretat >=3dB de best practice.\n"
        "Valors blau: Lden segons CNOSSOS amb penalitzacions.\n\n"
        f"*Anual CNOSSOS: barreja neutral/favourable amb p_fav Ld={P_FAV['Ld']:.2f}, Le={P_FAV['Le']:.2f}, Ln={P_FAV['Ln']:.2f}.\n"
        f"Perfils meteo {set_name}: T/HR/Pressió estacional (XEMA Sta. Coloma + bellcam HR).\n"
        f"Absorció ISO9613-1; Deygout multi-aresta; Gs/Gm/Gr={GROUND_GS}/{GROUND_GM}/{GROUND_GR}; receptors a façana +3 dB.\n\n"
        "En la fórmula de CNOSSOS/Directiva 2002/49/CE per Lden s’hi afegeixen penalitzacions subjectives.\n"
        "Lden: 14h dia (Ld), 2h vespre (Le +5 dB), 8h nit (Ln +10 dB) — normativa EU, horaris cat.",
        ha="left", va="bottom", fontsize=9)    
    """
    
    ax_legex.axis("off")
    # Llegenda distàncies (dins mapa, cantonada esquerra adalt)
    dist_patches = [Patch() for i in dist_rows]

    # leg = ["Distàncies mínimes rotor"] + [f"{a} – {b}: {d} m" for a,b,d in sorted(dist_rows, key=lambda t: t[0])]
    dist_labels = [f"{a} – {b}: {d} m" for a,b,d in sorted(dist_rows, key=lambda t: t[0])]

    dist_leg = ax_legex.legend(dist_patches, dist_labels, title="Distàncies mínimes rotor", loc="upper left",
                        frameon=True, framealpha=0.9, borderpad=0.6, labelspacing=0.4, prop={"size":8}, title_fontsize=9)
    
    
    ax_legex.add_artist(dist_leg)
    
    """
    suc_patches = [Patch(facecolor=ISOP_COLORS[i], edgecolor="black", linewidth=0.5) for i in range(len(ISOP_COLORS))]
    labels = [f"{ISOP_BOUNDS[i]}–{ISOP_BOUNDS[i+1]} dB" if ISOP_BOUNDS[i+1] < 80 else f"{ISOP_BOUNDS[i]}+ dB"
              for i in range(len(ISOP_BOUNDS)-1)]
    suc_leg = ax_legex.legend(suc_patches, labels, title="SUC — Lden total (turb + preexist.)", loc="upper left",
                        frameon=True, framealpha=0.9, borderpad=0.6, labelspacing=0.4, prop={"size":8}, title_fontsize=8)
    ax_legex.add_artist(suc_leg)
    """
    
    png_name = f"MAPA_BASE_PRO_60_50_CAT_ANNUAL_{suffix}.png"
    result_path = os.path.join(OUTPUT_DIR, png_name)        
    plt.savefig(result_path, dpi=220)
    plt.show()
    #plt.close(fig)

    print(f"Exportats: {png_name}, {csv_name}")

# ----------------------- LLANÇA ELS TRES JOCS -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BASE-CASE PRO — CNOSSOS (anual) amb YAML de configuració")
    parser.add_argument("--config", "-c", help="Ruta a config.yaml", default=None)
    parser.add_argument("--scenario", "-s", help="Escenari meteo (robust|central|sec)", default="robust")
    parser.add_argument("--suffix", "-i", help="Sufix per als fitxers exportats (p.ex. ROBUST)", default=None)
    parser.add_argument("--limits", "-l", help="Acustic limits", default="Sens_60")
    args = parser.parse_args()

    if args.config:
        try:
            cfg = load_config_yaml(args.config)
            apply_yaml_overrides(cfg)
            ensure_output_dir(OUTPUT_DIR)
            # we need to make this available inside of every worker as well
            global BBOX_POLY
            BBOX_POLY = box(XMIN, YMIN, XMAX, YMAX)
            load_suc_geoms()
            global active_limits_key
            active_limits_key = args.limits  # ← usa aquest
            global LIMITS
            LIMITS = limits_sets[active_limits_key]            
            print(f"[YAML] Config aplicat des de: {args.config}")
            
            setup_grid_memmap_after_yaml(
                XMIN=XMIN, YMIN=YMIN, XMAX=XMAX, YMAX=YMAX,
                GRID_STEP_M=GRID_STEP_M, TRANS=TRANS, Z=Z, sample_dem=sample_dem
            )
            print("[GRID] Malla creada i exportada a memmap (globals inicialitzats).")
            
        except Exception as e:
            print(f"[YAML] ERROR llegint config: {e}", file=sys.stderr)
            print(f"[GRID] ERROR creant malla/memmap: {e}", file=sys.stderr)
            sys.exit(2)

    
    # Deriva un sufix si no s'ha passat explícitament
    suf = args.suffix or args.scenario.upper()    
    run_set(args, suf)
    cleanup_grid_memmap()

