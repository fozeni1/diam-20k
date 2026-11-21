"""Построение сетки учебного полигона из PNG Terrarium 12/789/1587."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from diam20k.geography import MapOrigin


def build(path):
    origin = MapOrigin(37.45, -110.61, 1300)
    with Image.open(path) as image:
        if image.size != (256, 256) or image.mode not in ("RGB", "RGBA"):
            raise ValueError("Ожидается цветной тайл Terrarium размером 256 × 256")
        rgb = np.asarray(image, dtype=float)
    dem = rgb[:, :, 0]*256 + rgb[:, :, 1] + rgb[:, :, 2]/256 - 32768

    def elevation(n, e):
        lat, lon, _ = origin.to_geodetic((n, e, 0))
        x = ((lon+180)/360*4096 - 789)*256 - 0.5
        y = ((1-math.asinh(math.tan(math.radians(lat)))/math.pi)*2048 - 1587)*256 - 0.5
        i, j = math.floor(y), math.floor(x)
        if not (0 <= i < 255 and 0 <= j < 255):
            raise ValueError("Узел вышел за тайл")
        a, b = y-i, x-j
        return ((1-a)*((1-b)*dem[i,j] + b*dem[i,j+1]) +
                a*((1-b)*dem[i+1,j] + b*dem[i+1,j+1]))

    zero = elevation(0, 0)
    return {"north0": -300, "east0": -700, "step": 50,
            "heights": [[round(float(elevation(n,e)-zero), 2) for e in range(-700,701,50)]
                        for n in range(-300,4301,50)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.tile), separators=(",", ":"))+"\n", encoding="utf-8")
