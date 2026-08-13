from copy import deepcopy
from pprint import pprint  # noqa: F401

from manim import *  # pyright: ignore[reportWildcardImportFromLibrary]S

# region – config
config.frame_height = 20
config.frame_width = config.frame_height * 16 / 9

KEY_PADDING = 0.1

KEY_WIDTH_AVERAGE = 1.372
WHITE_KEY_WIDTH = KEY_WIDTH_AVERAGE * 12 / 7
C_TO_E_KEY_WIDTH = WHITE_KEY_WIDTH * 3 / 5
F_TO_B_KEY_WIDTH = WHITE_KEY_WIDTH * 4 / 7
CONV_WHITE_KEY_LENGTH = 14.6
CONV_BLACK_KEY_LENGTH = 9
JANKO_KEY_LENGTH = 2.85

BLACK_INDICES = {1, 3, 6, 8, 10}
FLAT_KEYS = {1, 3, 5, 6, 8, 10}
MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11, 12])
# endregion


# region – helpers
def flatten(top_list: list[list]) -> list:
    return [x for sublist in top_list for x in sublist]


def make_weight(original: Text, weight: str = "BOLD") -> Text:
    new = Text(
        original.original_text,
        weight=weight,
        font_size=original.font_size,
        font=original.font,
        slant=original.slant,
    )
    new.match_style(original)
    new.move_to(original.get_center())
    new.align_to(original, ORIGIN)
    return new


# endregion


class RoundedPolyline(VMobject):
    def __init__(self, points, radius=0.3, **kwargs):
        super().__init__(**kwargs)

        if len(points) < 2:
            return

        pts = [np.array(p, dtype=float) for p in points]
        r = radius

        self.start_new_path(pts[0])
        current_point = pts[0]

        for i in range(1, len(pts) - 1):
            p_prev, p, p_next = pts[i - 1], pts[i], pts[i + 1]

            v1 = p - p_prev
            v2 = p_next - p

            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if l1 < 1e-6 or l2 < 1e-6:
                continue

            u1, u2 = v1 / l1, v2 / l2

            # Angle between segments
            cos_ang = np.clip(np.dot(u1, u2), -1, 1)
            ang = np.arccos(cos_ang)

            # Skip nearly straight
            if ang < 1e-3 or ang > np.pi - 1e-3:
                self.add_line_to(p)
                current_point = p
                continue

            # Cut distance
            cut = r / np.tan(ang / 2)
            cut = min(cut, 0.9 * l1, 0.9 * l2)

            t_in = p - cut * u1
            t_out = p + cut * u2

            # Line up to arc start
            if np.linalg.norm(t_in - current_point) > 1e-6:
                self.add_line_to(t_in)

            # --- Arc construction ---
            dir1 = -u1
            dir2 = u2

            bisector = dir1 + dir2
            norm = np.linalg.norm(bisector)

            if norm < 1e-6:
                # fallback (almost straight)
                self.add_line_to(t_out)
                current_point = t_out
                continue

            bisector /= norm

            dist = r / np.sin(ang / 2)
            center = p + bisector * dist

            # Angles
            start_angle = np.arctan2(*(t_in - center)[1::-1])
            end_angle = np.arctan2(*(t_out - center)[1::-1])

            # Shortest signed angle
            delta = (end_angle - start_angle + np.pi) % (2 * np.pi) - np.pi

            # Orientation
            cross = np.cross(v1, v2)[2]
            if cross > 0:
                delta = abs(delta)
            else:
                delta = -abs(delta)

            arc = Arc(
                radius=r,
                start_angle=start_angle,
                angle=delta,
                arc_center=center,
            )

            # Append arc as part of same path
            self.append_points(arc.points)

            current_point = t_out

        # Final segment
        if np.linalg.norm(pts[-1] - current_point) > 1e-6:
            self.add_line_to(pts[-1])


class Key:
    SHARP_LABELS = (
        "A",
        "A♯",
        "B",
        "C",
        "C♯",
        "D",
        "D♯",
        "E",
        "F",
        "F♯",
        "G",
        "G♯",
    )
    FLAT_LABELS = (
        "A",
        "B♭",
        "B",
        "C",
        "D♭",
        "D",
        "E♭",
        "E",
        "F",
        "G♭",  # "f#/g♭",
        "G",
        "A♭",
    )

    def __init__(
        self,
        full_piano_idx: int,
        base_width: float,
        vmob: VMobject,
        left_offset: float,
        centers: list[tuple[float, float, float]],
    ) -> None:
        self.full_piano_idx: int = full_piano_idx
        self.octave_idx: int = (self.full_piano_idx - 3) % 12
        self.is_white: bool = self.octave_idx not in BLACK_INDICES

        self.base_width: float = base_width
        self.vmob: VMobject = vmob
        self.apply_default_style()
        self.left_offset: float = left_offset  # only used in the conventional piano
        self.centers: list[tuple[float, float, float]] = centers

        octave_nr: int = int(np.floor((full_piano_idx - 3) / 12))
        self.sharp_label = self.SHARP_LABELS[full_piano_idx % 12]
        self.key_label_w_octave = f"{self.sharp_label}_{octave_nr}"
        self.flat_label = self.FLAT_LABELS[full_piano_idx % 12]

    def __str__(self) -> str:
        return self.key_label_w_octave

    def __repr__(self) -> str:
        return f"Key({self.key_label_w_octave})"

    def apply_default_style(self):
        STROKE_WIDTH = 10
        if self.is_white:  # is_white
            self.vmob.set_color(WHITE).set_fill(WHITE, opacity=1).set_stroke(
                BLACK, opacity=0, width=STROKE_WIDTH
            ).set_z_index(-1)
        else:  # is_black
            self.vmob.set_color(BLACK).set_fill(BLACK, opacity=1).set_stroke(
                BLACK, opacity=0, width=STROKE_WIDTH
            )
            # ToDo: Remove
            # for vm in self.vmob:  ## Super hacky, remove this in the future!!
            #    vm.scale(1.04)
        return self


class Piano:
    def __init__(
        self,
        key_gen_func,
        vgroup_start: int = 0,
        vgroup_end: int = 88,
        add_bounding_box: bool = True,
    ) -> None:
        self.keys_backup: list[Key] = []  # These should remain unchanged!!

        for full_piano_idx in range(88):
            key: Key = key_gen_func(full_piano_idx)
            self.keys_backup.append(key)

        self.keys: list[Key] = deepcopy(self.keys_backup[vgroup_start:vgroup_end])
        self.vgroup: VGroup = VGroup(
            self.get_positioned_vmobs(vgroup_start, vgroup_end)
        )
        if add_bounding_box:
            self.vgroup = self.add_bounding_box(self.vgroup)
        self.has_bounding_box: bool = add_bounding_box

    def get_positioned_vmobs(
        self,
        abs_start_idx: int = 0,
        abs_end_idx: int = 88,
        abs_positioning: bool = False,
    ) -> list[VMobject]:
        res: list[VMobject] = []

        pos_ref_start_idx = 0 if abs_positioning else abs_start_idx

        for i, key in enumerate(self.keys_backup):
            if i >= abs_start_idx and i < abs_end_idx:
                vmob = key.vmob.copy()
                pos = sum(k.base_width for k in self.keys_backup[pos_ref_start_idx:i])
                res.append(vmob.shift(RIGHT * pos))

        return res

    def get_center_positions(
        self, relative_indices: set[int]
    ) -> list[list[tuple[float, float, float]]]:
        centers_list: list[list[tuple[float, float, float]]] = []

        for rel_idx, key in enumerate(self.keys):
            if rel_idx in relative_indices:
                # Exact original slot left + local bottom_center
                cum_pos = sum(
                    k.base_width
                    for k in self.keys
                    if k.full_piano_idx < key.full_piano_idx
                )
                centers = [(cum_pos + c[0], c[1], c[2]) for c in key.centers]
                centers_list.append(centers)

        return centers_list

    def reset_bounding_box(self, surround_vgroup: VGroup) -> VGroup:
        surround_vgroup[0].set_color(GREEN).set_opacity(0).set_fill(
            opacity=0
        ).set_stroke(width=30).set_z_index(98)
        return surround_vgroup

    def reset_own_bounding_box(self, in_target: bool = False):
        if self.has_bounding_box:
            if in_target:
                self.reset_bounding_box(
                    self.vgroup.target  # pyright: ignore[reportArgumentType]
                )
            else:
                self.reset_bounding_box(self.vgroup)
        else:
            warnings.warn(
                "WARNING: `reset_bounding_box_stroke` has been called even though there was no bounding-box!!"
            )
        return self

    def add_bounding_box(self, vmob: VGroup) -> VGroup:
        centers = flatten(self.get_center_positions(set(range(len(self.keys)))))

        tops = np.array([key.vmob.get_top() for key in self.keys])
        bottoms = np.array([key.vmob.get_bottom() for key in self.keys])

        xmin = self.keys[0].vmob.get_left()[0]
        xmax = centers[-1][0] + WHITE_KEY_WIDTH / 2
        ymin = bottoms[:, 1].min()
        ymax = tops[:, 1].max()

        bounding_box = Polygon(
            (xmin, ymin, 0), (xmax, ymin, 0), (xmax, ymax, 0), (xmin, ymax, 0)
        )
        res = VGroup(bounding_box, vmob)
        return self.reset_bounding_box(res)

    def get_pure_vgroup(self, in_target: bool = False) -> VGroup:
        source = self.vgroup.target if in_target else self.vgroup
        res = (
            source[1]  # pyright: ignore[reportOptionalSubscript]
            if self.has_bounding_box
            else source
        )
        return res  # pyright: ignore[reportReturnType]

    def create_fresh_vgroup(
        self, start_idx: int, end_idx: int, w_bounding_box: bool = True
    ):
        self.keys = deepcopy(self.keys_backup[start_idx:end_idx])
        self.vgroup = VGroup(self.get_positioned_vmobs(start_idx, end_idx))

        if w_bounding_box:
            self.vgroup = self.add_bounding_box(self.vgroup)
        self.has_bounding_box = w_bounding_box
        return self

    def get_scale_shape(
        self, relative_indices, color: ManimColor, w_bounding_box: bool = True
    ) -> VGroup:
        all_centers: list[list[tuple[float, float, float]]] = self.get_center_positions(
            relative_indices
        )

        def get_lowest_center(centers):
            return min(centers, key=lambda c: c[1])

        first_center = get_lowest_center(all_centers[0])
        selected_centers: list[tuple[float, float, float]] = [first_center]

        for key_centers in all_centers[1:]:
            if len(key_centers) == 1:
                selected_center = key_centers[0]
            else:
                eligible_centers = [
                    c for c in key_centers if c[1] >= first_center[1] - 0.001
                ]
                selected_center = get_lowest_center(eligible_centers)
            selected_centers.append(selected_center)

        point_radius = 0.4
        stroke_width = 30
        points = VGroup(  # noqa: F841
            Dot(point=pos, radius=point_radius, color=color) for pos in selected_centers
        )
        lines = (
            VMobject()
            .set_points_as_corners(selected_centers)
            .set_stroke(color=color, width=stroke_width)
        )

        res = VGroup(lines)
        if w_bounding_box:
            res = self.add_bounding_box(res)
        return res

    def apply_to_vmobs(self, func, in_target: bool = False):
        # [print((key, vmob)) for key, vmob in zip(self.keys, self.get_pure_vgroup())]
        for rel_idx, (key, vmob) in enumerate(
            zip(self.keys, self.get_pure_vgroup(in_target))
        ):
            func(rel_idx, key, vmob)
        return self

    def get_pure_vgroup_w_labels_original(self, flat: bool, font_size: int) -> VGroup:
        vmobs = []
        for key, vmob in zip(self.keys, self.get_pure_vgroup()):
            texts = [
                Text(
                    key.flat_label if flat else key.sharp_label,
                    font_size=font_size,
                    weight=BOLD,
                    color=BLACK if key.is_white else WHITE,
                )
                .set_opacity(1 if i == 0 else 0.4)
                .move_to(vmob[i])
                # .shift(center)
                for i, _ in enumerate(key.centers)
            ]

            vmobs.append(VGroup(vmob, *texts))

        return VGroup(*vmobs)

    def get_pure_vgroup_w_labels(self, flat: bool, font_size: int) -> VGroup:
        counter = 0
        vmobs = []
        for key, vmob in zip(self.keys, self.get_pure_vgroup()):
            texts = [
                Text(
                    key.flat_label if flat else key.sharp_label,
                    font_size=font_size,
                    weight=BOLD,
                    color=BLACK if key.is_white else WHITE,
                ).move_to(vmob[0]),
            ]

            START_IDX = 39
            END_IDX = 56
            if (
                key.is_white
                and key.full_piano_idx >= START_IDX
                and key.full_piano_idx < END_IDX
            ):
                counter += 1
                if counter == 10:
                    counter = 0

                texts.append(
                    Text(
                        str(counter),
                        font_size=font_size,
                        weight=BOLD,
                        color=BLACK if key.is_white else WHITE,
                    ).move_to(vmob[-1])
                )

            vmobs.append(VGroup(vmob, *texts))

        return VGroup(*vmobs)


# region – generators
def create_interval_chain(
    intervals, half_step_widht: float, stroke_width: int = 20, font_size: int = 50
) -> VGroup:
    start_indices: list[int] = intervals[:-1]
    end_indices: list[int] = intervals[1:]
    vgroup = VGroup()

    BAR_HEIGHT = 0.75
    PADDING = 0.15

    for idx, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
        start_x = start_idx * half_step_widht + PADDING
        end_x = end_idx * half_step_widht - PADDING
        if idx == 0:
            start_x -= PADDING
        elif idx == len(intervals) - 2:
            end_x += PADDING

        line = RoundedPolyline(
            [
                (start_x, 0, 0),
                (start_x, BAR_HEIGHT, 0),
                (end_x, BAR_HEIGHT, 0),
                (end_x, 0, 0),
            ],
            radius=0.2,
            color=WHITE,
            stroke_width=stroke_width,
        )

        t = Text(str(end_idx - start_idx), font_size=font_size).next_to(line, UP, 0.4)
        vgroup.add(line, t)

    return vgroup


def create_linear_key(full_piano_idx: int, key_width: float = KEY_WIDTH_AVERAGE) -> Key:
    # of_cde_keys = octave_idx <= 4
    # key_width = C_TO_E_KEY_WIDTH if of_cde_keys else F_TO_B_KEY_WIDTH

    shift = (key_width / 2, -key_width / 2, 0)
    vmob = Square(key_width - KEY_PADDING * 2).shift(shift).round_corners(radius=0.3)
    # vmob = Circle(radius=(key_width / 2) - KEY_PADDING).shift(shift)

    return Key(full_piano_idx, key_width, vmob, 0, [shift])


def create_conv_key(full_piano_idx: int) -> Key:
    octave_idx = (full_piano_idx - 3) % 12
    is_white = octave_idx not in BLACK_INDICES

    of_cde_keys = octave_idx <= 4
    key_width = C_TO_E_KEY_WIDTH if of_cde_keys else F_TO_B_KEY_WIDTH

    key_idx_in_group = octave_idx if of_cde_keys else octave_idx - 5
    white_idx_in_group = key_idx_in_group / 2

    left_white_pos = white_idx_in_group * WHITE_KEY_WIDTH - key_idx_in_group * key_width
    right_white_pos = left_white_pos + WHITE_KEY_WIDTH

    if full_piano_idx == 0:
        vertices = [
            # top left
            (left_white_pos, 0, 0),
            # bottom left
            (left_white_pos, -CONV_WHITE_KEY_LENGTH, 0),
            # bottom right
            (right_white_pos - KEY_PADDING, -CONV_WHITE_KEY_LENGTH, 0),
            # right side up to notch
            (
                right_white_pos - KEY_PADDING,
                -CONV_BLACK_KEY_LENGTH - KEY_PADDING,
                0,
            ),
            # notch inward (right)
            (
                key_width - KEY_PADDING,
                -CONV_BLACK_KEY_LENGTH - KEY_PADDING,
                0,
            ),
            # top right
            (key_width - KEY_PADDING, 0, 0),
        ]
        vmob = Polygon(*np.roll(vertices, 2, axis=0))
    elif full_piano_idx == 87:
        vmob = Rectangle(
            height=CONV_WHITE_KEY_LENGTH - KEY_PADDING,
            width=WHITE_KEY_WIDTH - 2 * KEY_PADDING,
            color=WHITE,
        ).shift(
            (
                WHITE_KEY_WIDTH / 2,
                -CONV_WHITE_KEY_LENGTH / 2,
                0,
            )
        )
    elif is_white:
        straight_left = key_idx_in_group == 0
        straight_right = octave_idx in [4, 11]

        vertices = [
            # top left
            (KEY_PADDING, 0, 0),
            # notch inward (left)
            (KEY_PADDING, -CONV_BLACK_KEY_LENGTH - KEY_PADDING, 0),
            # left side up to notch
            (
                KEY_PADDING + left_white_pos,
                -CONV_BLACK_KEY_LENGTH - KEY_PADDING,
                0,
            ),
            # bottom left
            (KEY_PADDING + left_white_pos, -CONV_WHITE_KEY_LENGTH, 0),
            # bottom right
            (right_white_pos - KEY_PADDING, -CONV_WHITE_KEY_LENGTH, 0),
            # right side up to notch
            (
                right_white_pos - KEY_PADDING,
                -CONV_BLACK_KEY_LENGTH - KEY_PADDING,
                0,
            ),
            # notch inward (right)
            (
                key_width - KEY_PADDING,
                -CONV_BLACK_KEY_LENGTH - KEY_PADDING,
                0,
            ),
            # top right
            (key_width - KEY_PADDING, 0, 0),
        ]

        vertices_to_remove = []
        if straight_left:
            vertices_to_remove = [1, 2]
        if straight_right:
            vertices_to_remove = [5, 6]

        vertices = np.array(
            [v for i, v in enumerate(vertices) if i not in vertices_to_remove]
        )
        vertices = np.roll(vertices, 2 if not straight_right else 1, axis=0)

        vmob = Polygon(*vertices)
    else:  # is_black
        vmob = Rectangle(
            height=CONV_BLACK_KEY_LENGTH - KEY_PADDING,
            width=key_width - 2 * KEY_PADDING,
            color=BLACK,
        ).shift(
            (
                key_width / 2,
                -CONV_BLACK_KEY_LENGTH / 2,
                0,
            )
        )

    if is_white:
        left_offset = left_white_pos
        white_middle = right_white_pos - WHITE_KEY_WIDTH / 2
        center = (white_middle, -CONV_WHITE_KEY_LENGTH + WHITE_KEY_WIDTH / 2, 0)
    else:
        left_offset = 0
        center = (key_width / 2, -CONV_BLACK_KEY_LENGTH + key_width / 2, 0)

    return Key(
        full_piano_idx,
        key_width,
        vmob.round_corners(0.2),
        left_offset,
        [center],
    )


def create_iso_key(full_piano_idx: int, key_width: float = KEY_WIDTH_AVERAGE) -> Key:
    is_top = full_piano_idx % 2 == 0

    left_white_pos = -key_width * 0.5
    right_white_pos = key_width * 1.5
    left_offset = 0 if is_top else left_white_pos

    if full_piano_idx == 87:
        vertices = [
            # top left
            (KEY_PADDING, 0, 0),
            # notch inward (left)
            (KEY_PADDING, -CONV_BLACK_KEY_LENGTH - KEY_PADDING, 0),
            # left side up to notch
            (
                KEY_PADDING + left_white_pos,
                -CONV_BLACK_KEY_LENGTH - KEY_PADDING,
                0,
            ),
            # bottom left
            (KEY_PADDING + left_white_pos, -CONV_WHITE_KEY_LENGTH, 0),
            # bottom right
            (right_white_pos - KEY_PADDING, -CONV_WHITE_KEY_LENGTH, 0),
            # top right
            (right_white_pos - KEY_PADDING, 0, 0),
        ]
        vmob = Polygon(*np.roll(vertices, 1, axis=0))
        white_middle = right_white_pos - WHITE_KEY_WIDTH / 2
        center = (white_middle, -CONV_WHITE_KEY_LENGTH + WHITE_KEY_WIDTH / 2, 0)
    elif is_top:
        vmob = Rectangle(
            height=CONV_BLACK_KEY_LENGTH - KEY_PADDING,
            width=key_width - 2 * KEY_PADDING,
            color=BLACK,
        ).shift((key_width / 2, -CONV_BLACK_KEY_LENGTH / 2, 0))
        center = (key_width / 2, -CONV_BLACK_KEY_LENGTH + key_width / 2, 0)
    else:  # Bottom Keys
        vertices = [
            # top left
            (KEY_PADDING, 0, 0),
            # notch inward (left)
            (KEY_PADDING, -CONV_BLACK_KEY_LENGTH - KEY_PADDING, 0),
            # left side up to notch
            (
                KEY_PADDING + left_white_pos,
                -CONV_BLACK_KEY_LENGTH - KEY_PADDING,
                0,
            ),
            # bottom left
            (KEY_PADDING + left_white_pos, -CONV_WHITE_KEY_LENGTH, 0),
            # bottom right
            (right_white_pos - KEY_PADDING, -CONV_WHITE_KEY_LENGTH, 0),
            # right side up to notch
            (
                right_white_pos - KEY_PADDING,
                -CONV_BLACK_KEY_LENGTH - KEY_PADDING,
                0,
            ),
            # notch inward (right)
            (
                key_width - KEY_PADDING,
                -CONV_BLACK_KEY_LENGTH - KEY_PADDING,
                0,
            ),
            # top right
            (key_width - KEY_PADDING, 0, 0),
        ]
        # pprint(vertices)
        vertices = np.roll(vertices, 2, axis=0)
        # pprint(vertices, end="\n\n")
        vmob = Polygon(*vertices)

        white_middle = right_white_pos - WHITE_KEY_WIDTH / 2
        center = (white_middle, -CONV_WHITE_KEY_LENGTH + WHITE_KEY_WIDTH / 2, 0)

    return Key(
        full_piano_idx,
        key_width,
        vmob.round_corners(0.2),
        left_offset,
        [center],
    )


def create_janko_key(
    full_piano_idx: int, nr_rows: int = 5, key_width: float = KEY_WIDTH_AVERAGE
) -> Key:
    start_row_idx_from_bottom = (full_piano_idx + 1) % 2

    width = 2 * key_width - 2 * KEY_PADDING
    x_shift = key_width / 2
    vmobs = []
    centers: list[tuple[float, float, float]] = []
    left_offset = x_shift - width / 2
    for row_idx in range(start_row_idx_from_bottom, nr_rows, 2):
        y_shift = (-0.5 - nr_rows + row_idx) * JANKO_KEY_LENGTH
        shift = (x_shift, y_shift, 0)

        vmobs.append(
            Rectangle(
                height=JANKO_KEY_LENGTH - 2 * KEY_PADDING,
                width=width,
            )
            .shift(shift)
            .round_corners(0.5)
        )
        centers.append(shift)

    return Key(
        full_piano_idx,
        key_width,
        VGroup(vmobs),
        left_offset,
        centers,
    )
