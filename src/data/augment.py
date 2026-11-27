"""Apply random training augmentations to heatmap or point-cloud batches.

Transforms use Config_4 units and run before input standardisation.
The training loop keeps them out of validation and test batches.
"""

import torch
import torch.nn.functional as F

RANGE_BIN_M = 0.0237  # Config_4 range-bin spacing in metres; 13 bins is about 0.3 m.


def _pick_one_per_group(b: int, enabled: list[str], conflicts: tuple[tuple[str, ...], ...],
                        device: torch.device) -> dict[str, torch.Tensor]:
    """Choose which enabled transforms may run on each window.

    When multiple transforms from a conflict group are enabled, allow exactly
    one per window, chosen uniformly. Transforms outside these groups stay eligible.

    Args:
        b: Number of windows in the batch.
        enabled: Names of enabled transforms.
        conflicts: Groups of transforms that must not run together on one window.
        device: Device on which to create the masks.

    Returns:
        A boolean mask of shape (B,) for each enabled transform. True permits
        the transform; its own random sampling may still leave a window unchanged.
    """
    allowed = {name: torch.ones(b, dtype=torch.bool, device=device) for name in enabled}
    for group in conflicts:
        members = [name for name in group if name in allowed]
        if len(members) < 2:
            continue
        chosen = torch.randint(len(members), (b,), device=device)
        for i, name in enumerate(members):
            allowed[name] = chosen == i
    return allowed


class HeatmapAugment:
    """Reverse time, shift range, vary the noise floor and add heatmap noise.

    Range shifts and floor jitter are alternatives on each window. Floor jitter
    operates in raw units, so it must not follow range normalisation. Time reversal
    requires order-independent labels. Frames are never masked because one lying
    frame can determine an any_lying window label.
    """

    CONFLICTS: tuple[tuple[str, ...], ...] = (("range_shift_bins", "floor_jitter"),)

    def __init__(
        self,
        range_shift_bins: float = 0.0,
        floor_jitter: float = 0.0,
        time_reverse: bool = False,
        floor_knots: int = 8,
        noise: float = 0.0,
    ) -> None:
        """Set transform sizes for Config_4 heatmaps; zero disables a numeric transform.

        Args:
            range_shift_bins: Maximum absolute shift, sampled uniformly per window.
                Fractional shifts are interpolated; one bin spans 0.0237 metres.
            floor_jitter: Standard deviation of smooth baseline noise in raw heatmap units.
            time_reverse: Reverse each window with probability 0.5. The configured
                pipeline permits this only with the any_lying label rule.
            floor_knots: Number of random baseline values interpolated along range.
            noise: Gaussian noise standard deviation relative to the training input scale.

        Raises:
            ValueError: If a size is negative or floor_knots is below two.
        """
        if min(range_shift_bins, floor_jitter, noise) < 0:
            raise ValueError("augmentation sizes must not be negative")
        if floor_knots < 2:
            raise ValueError("floor_knots must be at least 2")
        self.range_shift_bins = range_shift_bins
        self.floor_jitter = floor_jitter
        self.time_reverse = time_reverse
        self.floor_knots = floor_knots
        self.noise = noise
        self._input_std = torch.ones(1)

    def set_input_std(self, std: torch.Tensor) -> None:
        """Store the training input scale so added noise has the requested magnitude."""
        self._input_std = std.detach().clone()

    @property
    def active(self) -> bool:
        """Whether at least one transform is enabled."""
        return bool(self.range_shift_bins or self.floor_jitter or self.time_reverse or self.noise)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enabled transforms to a heatmap batch without editing the input in place.

        Args:
            x: Log1p heatmaps of shape (B, T, 1, R, D), before standardisation.

        Returns:
            Augmented heatmaps with the same shape.
        """
        enabled = [n for n in ("time_reverse", "range_shift_bins", "floor_jitter") if getattr(self, n)]
        run = _pick_one_per_group(len(x), enabled, self.CONFLICTS, x.device)
        if self.time_reverse:
            x = self._reverse(x)
        if self.range_shift_bins:
            x = self._shift(x, run["range_shift_bins"])
        if self.floor_jitter:
            x = self._jitter(x, run["floor_jitter"])
        if self.noise:
            x = x + (self.noise * self._input_std.to(x)) * torch.randn_like(x)
        return x

    def _reverse(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the frame order of each window with probability 0.5."""
        flip = torch.rand(len(x), device=x.device) < 0.5
        return torch.where(flip.view(-1, 1, 1, 1, 1), x.flip(1), x)

    def _shift(self, x: torch.Tensor, run: torch.Tensor) -> torch.Tensor:
        """Shift permitted windows along range using linear interpolation.

        Fill exposed bins with the window's median for each Doppler bin; do not
        wrap values around. A False entry in run gives that window a zero shift.
        """
        B, T, C, R, D = x.shape
        delta = (torch.rand(B, device=x.device) * 2 - 1) * self.range_shift_bins * run
        src = (torch.arange(R, device=x.device)[None, :] - delta[:, None]).clamp(-1, R)  # output bin <- input position
        low = src.floor()
        weight = (src - low).view(B, 1, 1, R, 1)
        low = low.long()
        fill = x.permute(0, 4, 1, 2, 3).reshape(B, D, -1).median(-1).values.view(B, 1, 1, 1, D)
        padded = torch.cat([fill.expand(B, T, C, 1, D), x, fill.expand(B, T, C, 1, D)], dim=3)  # input j is at j + 1
        first = (low + 1).clamp(0, R + 1).view(B, 1, 1, R, 1).expand(B, T, C, R, D)
        second = (low + 2).clamp(0, R + 1).view(B, 1, 1, R, 1).expand(B, T, C, R, D)
        return (1 - weight) * padded.gather(3, first) + weight * padded.gather(3, second)

    def _jitter(self, x: torch.Tensor, run: torch.Tensor) -> torch.Tensor:
        """Add a smooth random range baseline in raw units, then apply log1p again.

        Windows with run=False receive a zero baseline.
        """
        B, _, _, R, _ = x.shape
        knots = torch.randn(B, 1, self.floor_knots, device=x.device) * self.floor_jitter * run.view(B, 1, 1)
        baseline = F.interpolate(knots, size=R, mode="linear", align_corners=True).view(B, 1, 1, R, 1)
        return torch.log1p((torch.expm1(x) + baseline).clamp(min=0))



RANGE_MAX_M = 4.85  # Config_4 range limit in metres; discard points moved beyond it.
GHOST_CONE_DEG = 60.0  # Maximum ghost-point angle from the sensor's forward axis.


def _rotations(b: int, tilt: float | torch.Tensor, yaw: float | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return random rotation matrices of shape (B, 3, 3) for sensor-frame points.

    Args:
        b: Number of windows.
        tilt: Maximum absolute angles about x and z, in degrees.
        yaw: Maximum absolute angle about the sensor's forward axis y, in degrees.
        device: Device on which to create the matrices.

    Returns:
        Matrices applying x tilt, then z tilt, then y yaw to column vectors.
        Each limit may be a scalar or a tensor with one value per window.
    """

    def angle(limit_deg: float | torch.Tensor) -> torch.Tensor:
        """Sample one angle per window in radians within the supplied degree limit."""
        return (torch.rand(b, device=device) * 2 - 1) * limit_deg * torch.pi / 180

    def stack(rows: list[list[torch.Tensor]]) -> torch.Tensor:
        """Assemble batches of scalar entries into rotation matrices."""
        return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)

    one, zero = torch.ones(b, device=device), torch.zeros(b, device=device)
    yaw_a, tilt_z, tilt_x = angle(yaw), angle(tilt), angle(tilt)
    c, s = yaw_a.cos(), yaw_a.sin()
    r_y = stack([[c, zero, s], [zero, one, zero], [-s, zero, c]])
    c, s = tilt_z.cos(), tilt_z.sin()
    r_z = stack([[c, -s, zero], [s, c, zero], [zero, zero, one]])
    c, s = tilt_x.cos(), tilt_x.sin()
    r_x = stack([[one, zero, zero], [zero, c, -s], [zero, s, c]])
    return r_y @ r_z @ r_x


class PointAugment:
    """Transform sensor-frame points, add false alarms, then drop and jitter points.

    The sensor points down: y is its forward axis and x-z is the floor plane;
    z is not height above the floor. Geometry updates range-dependent SNR and
    removes out-of-range points. Mirror and yaw are alternatives per window.
    A single SNR offset is sampled per window before ghosts are added.
    """

    CONFLICTS: tuple[tuple[str, ...], ...] = (("yaw_deg", "mirror"),)

    def __init__(
        self,
        shift_m: float = 0.0,
        height_m: float = 0.0,
        yaw_deg: float = 0.0,
        tilt_deg: float = 0.0,
        mirror: bool = False,
        ghost_points: float = 0.0,
        snr_jitter_db: float = 0.0,
        range_max_m: float = RANGE_MAX_M,
        point_drop: float = 0.0,
        jitter: float = 0.0,
    ) -> None:
        """Set Config_4 point transforms; zero or False disables a transform.

        Args:
            shift_m: Maximum absolute translation along each floor axis, x and z.
            height_m: Maximum absolute translation along the sensor's forward axis y.
            yaw_deg: Maximum absolute rotation about y, in degrees. The configured
                pipeline requires collapsed lying labels because orientation changes.
            tilt_deg: Maximum absolute tilt about each of x and z, in degrees.
            mirror: Flip x with probability 0.5; requires collapsed lying labels.
            ghost_points: Mean Poisson count of false alarms added to empty slots.
            snr_jitter_db: Standard deviation of one Gaussian SNR offset per window.
            range_max_m: Sensor range limit; points beyond it are removed.
            point_drop: Probability of independently removing each remaining point.
            jitter: Gaussian noise standard deviation for x, y, z, Doppler and SNR,
                relative to each feature's training scale.

        Raises:
            ValueError: If a transform size is negative or point_drop is outside [0, 1].
        """
        if min(shift_m, height_m, yaw_deg, tilt_deg, ghost_points, snr_jitter_db, jitter) < 0:
            raise ValueError("augmentation sizes must not be negative")
        if not 0 <= point_drop <= 1:
            raise ValueError("point_drop must be in [0, 1]")
        self.shift_m = shift_m
        self.height_m = height_m
        self.yaw_deg = yaw_deg
        self.tilt_deg = tilt_deg
        self.mirror = mirror
        self.ghost_points = ghost_points
        self.snr_jitter_db = snr_jitter_db
        self.range_max_m = range_max_m
        self.point_drop = point_drop
        self.jitter = jitter
        self._input_std = torch.ones(5)

    def set_input_std(self, std: torch.Tensor) -> None:
        """Store training feature scales for x, y, z, Doppler and SNR jitter."""
        self._input_std = std.detach().clone()

    @property
    def active(self) -> bool:
        """Whether at least one transform is enabled."""
        return bool(self.shift_m or self.height_m or self.yaw_deg or self.tilt_deg or self.mirror
                    or self.ghost_points or self.snr_jitter_db or self.point_drop or self.jitter)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Augment point windows without modifying the input tensor.

        Args:
            x: Windows (B, P, 7): x, y, z, Doppler, SNR, relative time and validity.

        Returns:
            A tensor of the same shape. Removed points become zero padding.
        """
        return self.apply_with_count(x)[0]

    def apply_with_count(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment points and keep the count before random point dropping.

        Args:
            x: Sensor-frame windows of shape (B, P, 7), before standardisation.

        Returns:
            Augmented windows and counts (B, 1) after geometry and ghost insertion,
            before random dropping. PointNet uses these counts as a separate feature.
        """
        B, P, _ = x.shape
        names = ("shift_m", "height_m", "yaw_deg", "tilt_deg", "mirror", "ghost_points", "snr_jitter_db")
        run = _pick_one_per_group(B, [n for n in names if getattr(self, n)], self.CONFLICTS, x.device)
        on = {n: run.get(n, torch.zeros(B, dtype=torch.bool, device=x.device)).float() for n in names}
        valid = x[..., 6] > 0.5
        xyz = x[..., :3]
        if self.mirror:
            flip = (torch.rand(B, device=x.device) < 0.5) & run["mirror"]
            sign = torch.where(flip, -1.0, 1.0).view(B, 1)
            xyz = torch.cat([xyz[..., :1] * sign.unsqueeze(-1), xyz[..., 1:]], dim=-1)
        if self.yaw_deg or self.tilt_deg:
            rotation = _rotations(B, self.tilt_deg * on["tilt_deg"], self.yaw_deg * on["yaw_deg"], x.device)
            xyz = torch.einsum("bij,bpj->bpi", rotation, xyz)
        if self.shift_m or self.height_m:
            along_floor, along_boresight = self.shift_m * on["shift_m"], self.height_m * on["height_m"]
            limit = torch.stack([along_floor, along_boresight, along_floor], dim=-1).view(B, 1, 3)
            xyz = xyz + (torch.rand(B, 1, 3, device=x.device) * 2 - 1) * limit

        r_old = x[..., :3].norm(dim=-1).clamp(min=1e-3)
        r_new = xyz.norm(dim=-1).clamp(min=1e-3)
        valid = valid & (r_new <= self.range_max_m)
        out = x.clone()
        out[..., :3] = xyz
        out[..., 4] = x[..., 4] + 40 * torch.log10(r_old / r_new)
        if self.snr_jitter_db:
            out[..., 4] = out[..., 4] + torch.randn(B, 1, device=x.device) * self.snr_jitter_db * on["snr_jitter_db"].view(B, 1)
        out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
        out[..., 6] = valid.to(out.dtype)
        if self.ghost_points:
            out = self._add_ghosts(out, on["ghost_points"])
        valid = out[..., 6] > 0.5
        n_points = valid.sum(dim=1, keepdim=True)
        if self.point_drop:
            valid = valid & (torch.rand_like(out[..., 6]) >= self.point_drop)
            out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
        if self.jitter:
            noise = (self.jitter * self._input_std.to(out)) * torch.randn_like(out[..., :5])
            out[..., :5] = out[..., :5] + noise * valid.unsqueeze(-1)
        return out, n_points

    def _add_ghosts(self, x: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        """Fill available padding slots with low-SNR false alarms at random ranges and times."""
        B, P, _ = x.shape
        dev = x.device
        free = x[..., 6] < 0.5
        n = torch.poisson(torch.full((B,), float(self.ghost_points), device=dev) * on)
        add = free & (free.cumsum(dim=1) <= n.unsqueeze(1))
        r = 0.5 + torch.rand(B, P, device=dev) * (self.range_max_m - 0.5)
        cone = torch.tensor(GHOST_CONE_DEG * torch.pi / 180, device=dev)
        polar = torch.acos(1 - torch.rand(B, P, device=dev) * (1 - cone.cos()))  # uniform on the cone's cap
        azimuth = torch.rand(B, P, device=dev) * 2 * torch.pi
        ghost = torch.stack([
            r * polar.sin() * azimuth.cos(), r * polar.cos(), r * polar.sin() * azimuth.sin(),
            0.3 * torch.randn(B, P, device=dev),  # Doppler velocity.
            11 + 8 * torch.rand(B, P, device=dev),  # Low synthetic SNR.
            torch.rand(B, P, device=dev),  # Relative time within the window.
            torch.ones(B, P, device=dev),
        ], dim=-1)
        return torch.where(add.unsqueeze(-1), ghost, x)
