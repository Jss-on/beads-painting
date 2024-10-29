import numpy as np
import math
from PIL import Image
from cdither import dither, buffer_to_svg
# from dummy import dither
# Constants for luminance calculation
Pr, Pg, Pb = 0.2126, 0.7152, 0.0722

class RgbQuant:
    def __init__(self, opts=None):
        opts = opts or {}
        self.method = opts.get('method', 2)
        self.colors = opts.get('colors', 256)
        self.init_colors = opts.get('initColors', 4096)
        self.init_dist = opts.get('initDist', 0.01)
        self.dist_incr = opts.get('distIncr', 0.005)
        self.hue_groups = opts.get('hueGroups', 10)
        self.sat_groups = opts.get('satGroups', 10)
        self.lum_groups = opts.get('lumGroups', 10)
        self.min_hue_cols = opts.get('minHueCols', 0)
        self.hue_stats = HueStats(self.hue_groups, self.min_hue_cols) if self.min_hue_cols > 0 else None
        self.box_size = opts.get('boxSize', [64, 64])
        self.box_pxls = opts.get('boxPxls', 2)
        self.pal_locked = False
        self.dith_kern = opts.get('dithKern', None)
        self.dith_serp = opts.get('dithSerp', False)
        self.dith_delta = opts.get('dithDelta', 0)
        self.histogram = {}
        self.idx_rgb = opts.get('palette', []).copy()
        self.idx_i32 = []
        self.i32_idx = {}
        self.i32_rgb = {}
        self.use_cache = opts.get('useCache', True)
        self.cache_freq = opts.get('cacheFreq', 10)
        self.re_index = opts.get('reIndex', len(self.idx_rgb) == 0)
        self.color_dist = dist_manhattan if opts.get('colorDist') == "manhattan" else dist_euclidean

        if self.idx_rgb:
            for i, rgb in enumerate(self.idx_rgb):
                i32 = (255 << 24) | (rgb[2] << 16) | (rgb[1] << 8) | rgb[0]
                self.idx_i32.append(i32)
                self.i32_idx[i32] = i
                self.i32_rgb[i32] = rgb

    def sample(self, img, width=None):
        if self.pal_locked:
            raise ValueError("Cannot sample additional images, palette already assembled.")
        data = get_image_data(img, width)
        if self.method == 1:
            self.color_stats_1d(data['buf32'])
        elif self.method == 2:
            self.color_stats_2d(data['buf32'], data['width'])

    def reduce(self, img, ret_type=1, dith_kern=None, dith_serp=None, dither_type="slow"):
        if not self.pal_locked:
            self.build_pal()

        dith_kern = dith_kern or self.dith_kern
        dith_serp = dith_serp if dith_serp is not None else self.dith_serp

        if dith_kern:
            if dither_type == "slow":
            # print("Jession")
                out32 = self.slow_dither(img, dith_kern, dith_serp)
            else:
                out32 = self.fast_dither(img, dith_kern, dith_serp)
        else:
            data = get_image_data(img)
            buf32 = data['buf32']
            out32 = np.zeros_like(buf32, dtype=np.uint32)
            for i in range(len(buf32)):
                out32[i] = self.nearest_color(buf32[i])

        if ret_type == 1:
            return out32
        elif ret_type == 2:
            return [self.i32_idx[i32] for i32 in out32]

    def slow_dither(self, img, kernel, serpentine):
        kernels = {
            'FloydSteinberg': [
                [7 / 16, 1, 0], [3 / 16, -1, 1], [5 / 16, 0, 1], [1 / 16, 1, 1]
            ],
            'Atkinson': [
                [1 / 8, 1, 0], [1 / 8, 2, 0], [1 / 8, -1, 1], [1 / 8, 0, 1], [1 / 8, 1, 1], [1 / 8, 0, 2]
            ],
			"Sierra24A": [
				[2 / 4, 1, 0],
				[1 / 4, -1, 1],
				[1 / 4, 0, 1]
			],
			"Fan": [
				[7 / 16, 1, 0],
				[1 / 16, -2, 1],
				[3 / 16, -1, 1],
				[5 / 16, 0, 1]
			],
			"ShiauFan": [
				[4 / 8, 1, 0],
				[1 / 8, -2, 1],
				[1 / 8, -1, 1],
				[2 / 8, 0, 1]
			],
			"ShiauFan2": [
				[8 / 16, 1, 0],
				[1 / 16, -3, 1],
				[1 / 16, -2, 1],
				[2 / 16, -1, 1],
				[4 / 16, 0, 1]
			],
			"JarvisJudiceNinke": [
				[7 / 48, 1, 0],
				[5 / 48, 2, 0],
				[3 / 48, -2, 1],
				[5 / 48, -1, 1],
				[7 / 48, 0, 1],
				[5 / 48, 1, 1],
				[3 / 48, 2, 1],
				[1 / 48, -2, 2],
				[3 / 48, -1, 2],
				[5 / 48, 0, 2],
				[3 / 48, 1, 2],
				[1 / 48, 2, 2]
			],
			"Stucki": [
				[8 / 42, 1, 0],
				[4 / 42, 2, 0],
				[2 / 42, -2, 1],
				[4 / 42, -1, 1],
				[8 / 42, 0, 1],
				[4 / 42, 1, 1],
				[2 / 42, 2, 1],
				[1 / 42, -2, 2],
				[2 / 42, -1, 2],
				[4 / 42, 0, 2],
				[2 / 42, 1, 2],
				[1 / 42, 2, 2]
			],
			"Burkes": [
				[8 / 32, 1, 0],
				[4 / 32, 2, 0],
				[2 / 32, -2, 1],
				[4 / 32, -1, 1],
				[8 / 32, 0, 1],
				[4 / 32, 1, 1],
				[2 / 32, 2, 1]
			],
			"Sierra3": [
				[5 / 32, 1, 0],
				[3 / 32, 2, 0],
				[2 / 32, -2, 1],
				[4 / 32, -1, 1],
				[5 / 32, 0, 1],
				[4 / 32, 1, 1],
				[2 / 32, 2, 1],
				[2 / 32, -1, 2],
				[3 / 32, 0, 2],
				[2 / 32, 1, 2]
			],
			"Sierra2": [
				[4 / 16, 1, 0],
				[3 / 16, 2, 0],
				[1 / 16, -2, 1],
				[2 / 16, -1, 1],
				[3 / 16, 0, 1],
				[2 / 16, 1, 1],
				[1 / 16, 2, 1]
			],
            # Other kernels can be added here
        }

        if kernel not in kernels:
            raise ValueError(f"Unknown dithering kernel: {kernel}")

        ds = kernels[kernel]
        data = get_image_data(img)
        buf32 = data['buf32'].astype(np.uint32)
        width = data['width']
        height = data['height']
        dir = -1 if serpentine else 1

        for y in range(height):
            if serpentine:
                dir *= -1
            for x in range(width) if dir == 1 else range(width - 1, -1, -1):
                idx = y * width + x
                i32 = buf32[idx]
                r1 = (i32 & 0xff)
                g1 = (i32 & 0xff00) >> 8
                b1 = (i32 & 0xff0000) >> 16
                i32x = self.nearest_color(i32)
                # print(i32x)
                r2 = (i32x & 0xff)
                g2 = (i32x & 0xff00) >> 8
                b2 = (i32x & 0xff0000) >> 16
                buf32[idx] = (255 << 24) | (b2 << 16) | (g2 << 8) | r2

                if self.dith_delta:
                    dist = self.color_dist([r1, g1, b1], [r2, g2, b2])
                    if dist < self.dith_delta:
                        continue

                er, eg, eb = r1 - r2, g1 - g2, b1 - b2
                for i in range(len(ds))[::dir]:
                    x1 = ds[i][1] * dir #
                    y1 = ds[i][2]

                    if 0 <= x + x1 < width and 0 <= y + y1 < height:
                        d = ds[i][0]
                        idx2 = idx + int(y1 * width + x1)

                        r3 = (buf32[idx2] & 0xff)
                        g3 = (buf32[idx2] & 0xff00) >> 8
                        b3 = (buf32[idx2] & 0xff0000) >> 16

                        r4 = max(0, min(255, r3 + er * d))
                        g4 = max(0, min(255, g3 + eg * d))
                        b4 = max(0, min(255, b3 + eb * d))
                        # res = (255 << 24) | (int(b4) << 16) | (int(g4) << 8) | int(r4)
                        # print(res)
                        buf32[idx2] = (255 << 24) | (int(b4) << 16) | (int(g4) << 8) | int(r4)

        return buf32
    
    def fast_dither(self, img, kernel, serpentine, color_dist="manhattan"):
        # Prepare the parameters
        data = get_image_data(img)
        buf32 = data['buf32'].astype(np.uint32)
        width = data['width']
        height = data['height']
        

        # Call the Cython dither function
        out32 = dither(buf32, width, height, kernel, serpentine, color_dist, self.idx_rgb, self.idx_i32, self.dith_delta, self.use_cache, self.i32_idx)
        return out32
        

    def build_pal(self, no_sort=False):
        if self.pal_locked or (self.idx_rgb and len(self.idx_rgb) <= self.colors):
            return

        hist_g = self.histogram
        sorted_hist = sorted(hist_g, key=hist_g.get, reverse=True)

        if not sorted_hist:
            raise ValueError("Nothing has been sampled, palette cannot be built.")

        if self.method == 1:
            cols = self.init_colors
            last = sorted_hist[cols - 1]
            freq = hist_g[last]
            idxi32 = sorted_hist[:cols]
            while len(sorted_hist) > cols and hist_g[sorted_hist[cols]] == freq:
                idxi32.append(sorted_hist[cols])
                cols += 1
            if self.hue_stats:
                self.hue_stats.inject(idxi32)
        elif self.method == 2:
            idxi32 = sorted_hist

        idxi32 = [int(v) for v in idxi32]
        self.reduce_pal(idxi32)

        if not no_sort and self.re_index:
            self.sort_pal()

        if self.use_cache:
            self.cache_histogram(idxi32)

        self.pal_locked = True

    def palette(self, tuples=True, no_sort=False):
        self.build_pal(no_sort)
        if tuples:
            return self.idx_rgb
        else:
            return np.array(self.idx_i32, dtype=np.uint8)

    def prune_pal(self, keep):
        for i in range(len(self.idx_rgb)):
            if not keep[i]:
                i32 = self.idx_i32[i]
                self.idx_rgb[i] = None
                self.idx_i32[i] = None
                del self.i32_idx[i32]

        if self.re_index:
            idx_rgb = []
            idx_i32 = []
            i32_idx = {}
            for i in range(len(self.idx_rgb)):
                if self.idx_rgb[i]:
                    i32 = self.idx_i32[i]
                    idx_rgb.append(self.idx_rgb[i])
                    idx_i32.append(i32)
                    i32_idx[i32] = len(idx_i32) - 1
            self.idx_rgb = idx_rgb
            self.idx_i32 = idx_i32
            self.i32_idx = i32_idx

    def reduce_pal(self, idxi32):
        if len(self.idx_rgb) > self.colors:
            keep = {}
            uniques = 0
            for i32 in idxi32:
                if uniques == self.colors:
                    self.prune_pal(keep)
                    break
                idx = self.nearest_index(i32)
                if uniques < self.colors and not keep.get(idx):
                    keep[idx] = True
                    uniques += 1
            self.prune_pal(keep)
        else:
            idx_rgb = [[(i32 & 0xff), (i32 >> 8) & 0xff, (i32 >> 16) & 0xff] for i32 in idxi32]
            thold = self.init_dist
            while len(idx_rgb) > self.colors:
                mem_dist = []
                for i, pxi in enumerate(idx_rgb):
                    if not pxi:
                        continue
                    for j, pxj in enumerate(idx_rgb[i + 1:], i + 1):
                        if not pxj:
                            continue
                        dist = self.color_dist(pxi, pxj)
                        if dist < thold:
                            mem_dist.append((j, pxj, dist))
                            idx_rgb[j] = None
                thold += self.init_dist if len(idx_rgb) > self.colors * 3 else self.dist_incr
            if len(idx_rgb) < self.colors:
                mem_dist.sort(key=lambda x: -x[2])
                while len(idx_rgb) < self.colors:
                    idx_rgb[mem_dist[0][0]] = mem_dist.pop(0)[1]

            for i in range(len(idx_rgb)):
                if idx_rgb[i]:
                    self.idx_rgb.append(idx_rgb[i])
                    self.idx_i32.append(idxi32[i])
                    self.i32_idx[idxi32[i]] = len(self.idx_i32) - 1
                    self.i32_rgb[idxi32[i]] = idx_rgb[i]

    def nearest_color(self, i32):
        idx = self.nearest_index(i32)
        return self.idx_i32[idx] if idx is not None else 0

    def nearest_index(self, i32):
        if (i32 & 0xff000000) >> 24 == 0:
            return None

        if self.use_cache and str(i32) in self.i32_idx:
            return self.i32_idx[str(i32)]

        min_dist = 1000
        idx = None
        rgb = [
            i32 & 0xff,
            (i32 & 0xff00) >> 8,
            (i32 & 0xff0000) >> 16,
        ]
        for i, color in enumerate(self.idx_rgb):
            if not color:
                continue
            dist = self.color_dist(rgb, color)
            if dist < min_dist:
                min_dist = dist
                idx = i
        return idx

    def cache_histogram(self, idxi32):
        for i32 in idxi32:
            if self.histogram[i32] >= self.cache_freq:
                self.i32_idx[i32] = self.nearest_index(i32)

    def color_stats_1d(self, buf32):
        for col in buf32:
            if (col & 0xff000000) >> 24 == 0:
                continue
            if self.hue_stats:
                self.hue_stats.check(col)
            if col in self.histogram:
                self.histogram[col] += 1
            else:
                self.histogram[col] = 1

    def color_stats_2d(self, buf32, width):
        box_w, box_h = self.box_size
        area = box_w * box_h
        boxes = make_boxes(width, len(buf32) // width, box_w, box_h)
        for box in boxes:
            effc = max(round((box['w'] * box['h']) / area) * self.box_pxls, 2)
            hist_l = {}
            for i in range(box['y'], box['y'] + box['h']):
                for j in range(box['x'], box['x'] + box['w']):
                    col = buf32[i * width + j]
                    if (col & 0xff000000) >> 24 == 0:
                        continue
                    if self.hue_stats:
                        self.hue_stats.check(col)
                    if col in self.histogram:
                        self.histogram[col] += 1
                    elif col in hist_l:
                        if hist_l[col] + 1 >= effc:
                            self.histogram[col] = hist_l[col] + 1
                        hist_l[col] += 1
                    else:
                        hist_l[col] = 1
        if self.hue_stats:
            self.hue_stats.inject(self.histogram)

    def sort_pal(self):
        self.idx_i32.sort(key=lambda i32: (
            self.i32_idx[i32],
            rgb2hsl(*self.idx_rgb[self.i32_idx[i32]])
        ))

        self.idx_rgb = [self.i32_rgb[i32] for i32 in self.idx_i32]
        self.i32_idx = {i32: i for i, i32 in enumerate(self.idx_i32)}

class HueStats:
    def __init__(self, num_groups, min_cols):
        self.num_groups = num_groups
        self.min_cols = min_cols
        self.stats = {i: {'num': 0, 'cols': []} for i in range(-1, num_groups)}
        self.groups_full = 0

    def check(self, i32):
        if self.groups_full == self.num_groups + 1:
            return

        r = i32 & 0xff
        g = (i32 >> 8) & 0xff
        b = (i32 >> 16) & 0xff
        hg = -1 if r == g == b else hue_group(rgb2hsl(r, g, b)['h'], self.num_groups)
        gr = self.stats[hg]

        gr['num'] += 1
        if gr['num'] == self.min_cols:
            self.groups_full += 1
        if gr['num'] <= self.min_cols:
            gr['cols'].append(i32)

    def inject(self, hist_g):
        for i in range(-1, self.num_groups):
            if self.stats[i]['num'] <= self.min_cols:
                for col in self.stats[i]['cols']:
                    if isinstance(hist_g, list):
                        if col not in hist_g:
                            hist_g.append(col)
                    elif isinstance(hist_g, dict):
                        if col not in hist_g:
                            hist_g[col] = 1
                        else:
                            hist_g[col] += 1

def rgb2lum(r, g, b):
    """Calculate the luminance of an RGB color using Rec. 709 coefficients."""
    return math.sqrt(Pr * r * r + Pg * g * g + Pb * b * b)

def dist_euclidean(rgb0, rgb1):
    """Calculate the perceptual Euclidean color distance between two RGB colors."""
    rd = rgb1[0] - rgb0[0]
    gd = rgb1[1] - rgb0[1]
    bd = rgb1[2] - rgb0[2]
    return math.sqrt(Pr * rd * rd + Pg * gd * gd + Pb * bd * bd) / math.sqrt(Pr * 255 * 255 + Pg * 255 * 255 + Pb * 255 * 255)

def dist_manhattan(rgb0, rgb1):
    """Calculate the perceptual Manhattan color distance between two RGB colors."""
    rd = abs(rgb1[0] - rgb0[0])
    gd = abs(rgb1[1] - rgb0[1])
    bd = abs(rgb1[2] - rgb0[2])
    return (Pr * rd + Pg * gd + Pb * bd) / (Pr * 255 + Pg * 255 + Pb * 255)

def rgb2hsl(r, g, b):
    """Convert an RGB color to HSL color space."""
    r /= 255.0
    g /= 255.0
    b /= 255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    l = (max_val + min_val) / 2
    
    if max_val == min_val:
        h = s = 0
    else:
        d = max_val - min_val
        s = d / (2.0 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
        if max_val == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6
    
    return {'h': h, 's': s, 'l': rgb2lum(r * 255, g * 255, b * 255)}

def hue_group(hue, segs):
    seg = 1 / segs
    haf = seg / 2
    if hue >= 1 - haf or hue <= haf:
        return 0
    for i in range(1, segs):
        mid = i * seg
        if hue >= mid - haf and hue <= mid + haf:
            return i
    return 0

def make_boxes(wid, hgt, w0, h0):
    wnum = wid // w0
    wrem = wid % w0
    hnum = hgt // h0
    hrem = hgt % h0
    xend = wid - wrem
    yend = hgt - hrem
    boxes = [{'x': x, 'y': y, 'w': w0 if x != xend else wrem, 'h': h0 if y != yend else hrem} for y in range(0, hgt, h0) for x in range(0, wid, w0)]
    return boxes

def get_image_data(img, width=None):
    if isinstance(img, Image.Image):
        width, height = img.size
        buf8 = np.array(img.convert('RGBA')).astype(np.uint8)
        buf32 = buf8.view(dtype=np.uint32).reshape((height, width))
    elif isinstance(img, np.ndarray):
        height, width, channels = img.shape
        if channels == 4:  # Assuming img is RGBA
            buf8 = img.astype(np.uint8)
            buf32 = buf8.view(dtype=np.uint32).reshape((height, width))
        else:
            raise ValueError("Expected an RGBA image with 4 channels")
    else:
        raise TypeError("Unsupported image type")

    return {
        'buf8': buf8,
        'buf32': buf32.flatten(),
        'width': width,
        'height': height,
    }
