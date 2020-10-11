import os
from cairosvg import svg2png

INPUT_DIR = 'data'
OUTPUT_DIR = 'output'


class Comparator:
    def __init__(self, s):
        inputs = s.strip().split(":")
        input1 = int(inputs[0])
        input2 = int(inputs[1])
        if input1 < input2:
            self.i1 = input1
            self.i2 = input2
        else:
            self.i1 = input2
            self.i2 = input1

    def __str__(self):
        return f"{self.i1}:{self.i2}"

    def __hash__(self):
        return f"{self.i1}:{self.i2}".__hash__()

    def overlaps(self, other):
        return (self.i1 < other.i1 < self.i2) or (self.i1 < other.i2 < self.i2) or \
               (other.i1 < self.i1 < other.i2) or (other.i1 < self.i2 < other.i2)

    def has_same_input(self, other):
        return self.i1 == other.i1 or \
               self.i1 == other.i2 or \
               self.i2 == other.i1 or \
               self.i2 == other.i2


def build_svg(comparators: list):
    scale = 1
    x_scale = scale * 35
    y_scale = scale * 20
    comparators_svg = ""
    w = x_scale
    group = {}
    for c in comparators:
        # If the comparator inputs are the same position as any other comparator in the group, then start a new group
        for other in group:
            if c.has_same_input(other):
                for _, pos in group.items():
                    if pos > w:
                        w = pos
                w += x_scale
                group = {}
                break
        # Adjust the comparator x position to avoid overlapping any existing comparators in the group
        cx = w
        for other, other_pos in group.items():
            if other_pos >= cx and c.overlaps(other):
                cx = other_pos + x_scale / 3
        # Generate two circles and a line representing the comparator
        y0 = y_scale + c.i1 * y_scale
        y1 = y_scale + c.i2 * y_scale
        comparators_svg += f"<circle cx='{cx}' cy='{y0}' r='3' style='stroke:black;stroke-width:1;fill=yellow' />" + \
            f"<line x1='{cx}' y1='{y0}' x2='{cx}' y2='{y1}' style='stroke:black;stroke-width:1' />" + \
            f"<circle cx='{cx}' cy='{y1}' r='3' style='stroke:black;stroke-width:1;fill=yellow' />"
        group[c] = cx
    lines_svg = ""
    w += x_scale
    n = max([c.i2 for c in comparators]) + 1
    for i in range(0, n):
        y = y_scale + i * y_scale
        lines_svg += f"<line x1='0' y1='{y}' x2='{w}' y2='{y}' style='stroke:black;stroke-width:1' />"

    h = (n + 1) * y_scale
    return "<?xml version='1.0' encoding='utf-8'?>" + "<!DOCTYPE svg>" + \
        f"<svg width='{w}' height='{h}' xmlns='http://www.w3.org/2000/svg'>{comparators_svg}{lines_svg} </svg>"


def read_comparison_network(filename):
    with open(filename) as f:
        return [Comparator(c) for line in f for c in line.split(',')]


def main():
    for file in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, file)
        network = read_comparison_network(file_path)
        svg_output = build_svg(network)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0] + '.png')
        svg2png(bytestring=svg_output, write_to=output_path, dpi=300, output_width=800, output_height=600)


if __name__ == '__main__':
    main()
