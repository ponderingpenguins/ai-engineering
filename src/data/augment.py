"""Training-time augmentation of windows of heatmaps.

Runs on a batch of windows inside the training loop and never on validation or test data, so it adds variety
without touching the evaluation. Every transform is off by default.

The sizes below are meant for Config_4, the only sensor config used for model development: heatmaps of 256
range bins x 16 Doppler bins, 0.0237 m per range bin (a 6.07 m span) and 0.2 s per frame. The physical range
resolution is a coarser 0.042 m, the bins are an oversampled grid. Other configs have other bin sizes, so
the same numbers would mean something else.

The input is what `make_windows` produces: log1p of the raw heatmap, shape (B, T, 1, R, D), with zero velocity
at Doppler bin 0 (the axis is not centred, which is why nothing here rolls or shifts along Doppler).
"""

import torch
import torch.nn.functional as F

RANGE_BIN_M = 0.0237  # Config_4 range bin spacing; 13 bins is about 0.3 m


class HeatmapAugment:
    """Random augmentation of a batch of heatmap windows, applied in this order.

    1. `time_reverse`: plays a window backwards.
    2. `range_shift_bins`: moves the whole scene along range, as if the sensor were mounted higher or lower.
    3. `floor_jitter`: adds a smooth random baseline along range, as the noise floor differs between setups.
    4. `time_mask_frames`: replaces a short run of frames by the window's typical frame, like dropped frames.
    """

    def __init__(
        self,
        range_shift_bins: float = 0.0,
        floor_jitter: float = 0.0,
        time_reverse: bool = False,
        time_mask_frames: int = 0,
        floor_knots: int = 8,
    ) -> None:
        """Stores the settings.

        Args:
            range_shift_bins: Largest shift in range bins; each window gets a uniform random shift in
                [-n, n], fractional shifts included. One bin is 0.0237 m, so 13 bins is about 0.3 m.
                0 turns it off.
            floor_jitter: Std of the baseline added to the raw heatmap, in raw heatmap units (the floor of an
                empty room varies by roughly 100-180 units across range and recordings). 0 turns it off.
            time_reverse: Play each window backwards with probability 0.5. Only valid when the label does not
                depend on the order of the frames, which holds for the `any_lying` label rule but not for `last`.
            time_mask_frames: Each window loses one run of up to this many frames. 0 turns it off.
            floor_knots: Number of random values the baseline is interpolated between along range.

        Raises:
            ValueError: If a setting is negative or `floor_knots` is below 2.
        """
        if min(range_shift_bins, floor_jitter, time_mask_frames) < 0:
            raise ValueError("augmentation sizes must not be negative")
        if floor_knots < 2:
            raise ValueError("floor_knots must be at least 2")
        self.range_shift_bins = range_shift_bins
        self.floor_jitter = floor_jitter
        self.time_reverse = time_reverse
        self.time_mask_frames = time_mask_frames
        self.floor_knots = floor_knots

    @property
    def active(self) -> bool:
        """Whether any transform is switched on."""
        return bool(self.range_shift_bins or self.floor_jitter or self.time_reverse or self.time_mask_frames)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Augments a batch.

        Args:
            x: Windows of shape (B, T, 1, R, D), log1p of the raw heatmaps.

        Returns:
            A tensor of the same shape; `x` itself is not modified.
        """
        if self.time_reverse:
            x = self._reverse(x)
        if self.range_shift_bins:
            x = self._shift(x)
        if self.floor_jitter:
            x = self._jitter(x)
        if self.time_mask_frames:
            x = self._mask(x)
        return x

    def _reverse(self, x: torch.Tensor) -> torch.Tensor:
        flip = torch.rand(len(x), device=x.device) < 0.5
        return torch.where(flip.view(-1, 1, 1, 1, 1), x.flip(1), x)

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shifts every window along range by its own random amount, linearly interpolated.

        Bins that the shift uncovers are filled with the window's median per Doppler bin, which is the noise
        floor (a person covers only a few of the 256 range bins). Nothing wraps around.
        """
        B, T, C, R, D = x.shape
        delta = (torch.rand(B, device=x.device) * 2 - 1) * self.range_shift_bins
        src = (torch.arange(R, device=x.device)[None, :] - delta[:, None]).clamp(-1, R)  # output bin <- input position
        low = src.floor()
        weight = (src - low).view(B, 1, 1, R, 1)
        low = low.long()
        fill = x.permute(0, 4, 1, 2, 3).reshape(B, D, -1).median(-1).values.view(B, 1, 1, 1, D)
        padded = torch.cat([fill.expand(B, T, C, 1, D), x, fill.expand(B, T, C, 1, D)], dim=3)  # input j is at j + 1
        first = (low + 1).clamp(0, R + 1).view(B, 1, 1, R, 1).expand(B, T, C, R, D)
        second = (low + 2).clamp(0, R + 1).view(B, 1, 1, R, 1).expand(B, T, C, R, D)
        return (1 - weight) * padded.gather(3, first) + weight * padded.gather(3, second)

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a smooth random baseline along range to the raw heatmap, and takes log1p again."""
        B, _, _, R, _ = x.shape
        knots = torch.randn(B, 1, self.floor_knots, device=x.device) * self.floor_jitter
        baseline = F.interpolate(knots, size=R, mode="linear", align_corners=True).view(B, 1, 1, R, 1)
        return torch.log1p((torch.expm1(x) + baseline).clamp(min=0))

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        """Replaces one random run of frames per window by the window's median frame."""
        B, T = x.shape[:2]
        length = torch.randint(0, min(self.time_mask_frames, T - 1) + 1, (B, 1), device=x.device)
        start = torch.randint(0, T, (B, 1), device=x.device)
        t = torch.arange(T, device=x.device)[None, :]
        masked = ((t >= start) & (t < start + length)).view(B, T, 1, 1, 1)
        return torch.where(masked, x.median(dim=1, keepdim=True).values, x)


RANGE_MAX_M = 4.85  # Config_4 range limit; points moved beyond it are dropped, as the sensor would not report them
GHOST_CONE_DEG = 60.0  # ghost points appear within this angle of the boresight


def _rotations(b: int, tilt: float, yaw: float, device: torch.device) -> torch.Tensor:
    """One rotation matrix per window: a yaw about the boresight (y), then small tilts about z and x. Shape (B, 3, 3)."""

    def angle(limit_deg: float) -> torch.Tensor:
        return (torch.rand(b, device=device) * 2 - 1) * limit_deg * torch.pi / 180

    def stack(rows: list[list[torch.Tensor]]) -> torch.Tensor:
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
    """Random augmentation of a batch of radar point clouds, applied in this order.

    The points are in the sensor frame (see `data.windows._window_points`): x, y, z in metres, with y along the
    boresight. The sensor looks down at the floor, so y is the distance to the floor (median about 1.8 m in
    Config_4 recordings) and the floor plane is x-z. z is NOT height above the floor.

    1. `mirror`: flips x, swapping left and right.
    2. `yaw_deg` and `tilt_deg`: rotate about the boresight (the person lying in another direction on the floor),
       and tilt the sensor a few degrees (a mount that is not quite level).
    3. `shift_m` and `height_m`: translate along the floor (x and z) and along the boresight (y), i.e. a sensor
       that sits somewhere else. A point that ends up beyond the range limit is dropped, and the SNR of every
       point is rescaled for its new range (power falls as R^-4, so 40*log10(r_old/r_new) dB).
    4. `ghost_points`: adds false alarms in empty slots of the window.
    5. `snr_jitter_db`: one random SNR offset per window.

    Dropping points and jitter already happen inside `MiniPointNet` (`point_drop`, `jitter`), so they are not repeated.
    The sizes are for Config_4, the only sensor config used for model development.
    """

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
    ) -> None:
        """Stores the settings.

        Args:
            shift_m: Each window is moved by a uniform random amount in [-n, n] metres along x and along z.
            height_m: The same along y, the boresight.
            yaw_deg: Each window is rotated about the boresight by a uniform random angle in [-n, n] degrees.
                180 gives every direction. Changes which way a person lies, so it is only valid for labels that
                do not depend on that (`eval.collapse_lying=true`).
            tilt_deg: Each window is tilted about x and about z by a uniform random angle in [-n, n] degrees.
            mirror: Mirror half of the windows. Swaps left and right, so the same label restriction as `yaw_deg`.
            ghost_points: Mean number of false-alarm points added per window (Poisson), into empty slots only.
            snr_jitter_db: Std of one Gaussian SNR offset per window, in dB.
            range_max_m: Points farther than this from the sensor after the move are dropped.

        Raises:
            ValueError: If a size is negative.
        """
        if min(shift_m, height_m, yaw_deg, tilt_deg, ghost_points, snr_jitter_db) < 0:
            raise ValueError("augmentation sizes must not be negative")
        self.shift_m = shift_m
        self.height_m = height_m
        self.yaw_deg = yaw_deg
        self.tilt_deg = tilt_deg
        self.mirror = mirror
        self.ghost_points = ghost_points
        self.snr_jitter_db = snr_jitter_db
        self.range_max_m = range_max_m

    @property
    def active(self) -> bool:
        """Whether any transform is switched on."""
        return bool(self.shift_m or self.height_m or self.yaw_deg or self.tilt_deg or self.mirror
                    or self.ghost_points or self.snr_jitter_db)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Augments a batch.

        Args:
            x: Windows of shape (B, P, 7): x, y, z, doppler, snr, time, and a validity flag (0 for padding).

        Returns:
            A tensor of the same shape; `x` itself is not modified. Dropped points become padding (all zeros).
        """
        B, P, _ = x.shape
        valid = x[..., 6] > 0.5
        xyz = x[..., :3]
        if self.mirror:
            sign = torch.where(torch.rand(B, device=x.device) < 0.5, -1.0, 1.0).view(B, 1)
            xyz = torch.cat([xyz[..., :1] * sign.unsqueeze(-1), xyz[..., 1:]], dim=-1)
        if self.yaw_deg or self.tilt_deg:
            xyz = torch.einsum("bij,bpj->bpi", _rotations(B, self.tilt_deg, self.yaw_deg, x.device), xyz)
        if self.shift_m or self.height_m:
            limit = torch.tensor([self.shift_m, self.height_m, self.shift_m], device=x.device)
            xyz = xyz + (torch.rand(B, 1, 3, device=x.device) * 2 - 1) * limit

        r_old = x[..., :3].norm(dim=-1).clamp(min=1e-3)
        r_new = xyz.norm(dim=-1).clamp(min=1e-3)
        valid = valid & (r_new <= self.range_max_m)
        out = x.clone()
        out[..., :3] = xyz
        out[..., 4] = x[..., 4] + 40 * torch.log10(r_old / r_new)
        if self.snr_jitter_db:
            out[..., 4] = out[..., 4] + torch.randn(B, 1, device=x.device) * self.snr_jitter_db
        out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
        out[..., 6] = valid.to(out.dtype)
        if self.ghost_points:
            out = self._add_ghosts(out)
        return out

    def _add_ghosts(self, x: torch.Tensor) -> torch.Tensor:
        """Fills some empty slots with false alarms: random range and direction, low SNR, random time."""
        B, P, _ = x.shape
        dev = x.device
        free = x[..., 6] < 0.5
        n = torch.poisson(torch.full((B,), float(self.ghost_points), device=dev))
        add = free & (free.cumsum(dim=1) <= n.unsqueeze(1))
        r = 0.5 + torch.rand(B, P, device=dev) * (self.range_max_m - 0.5)
        cone = torch.tensor(GHOST_CONE_DEG * torch.pi / 180, device=dev)
        polar = torch.acos(1 - torch.rand(B, P, device=dev) * (1 - cone.cos()))  # uniform on the cone's cap
        azimuth = torch.rand(B, P, device=dev) * 2 * torch.pi
        ghost = torch.stack([
            r * polar.sin() * azimuth.cos(), r * polar.cos(), r * polar.sin() * azimuth.sin(),
            0.3 * torch.randn(B, P, device=dev),  # doppler
            11 + 8 * torch.rand(B, P, device=dev),  # just above the detection threshold
            torch.rand(B, P, device=dev),  # time within the window
            torch.ones(B, P, device=dev),
        ], dim=-1)
        return torch.where(add.unsqueeze(-1), ghost, x)
