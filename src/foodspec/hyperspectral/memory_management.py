"""
Memory-efficient processing for large hyperspectral image cubes.

Provides streaming, chunking, and tiling strategies for HyperspectralDataset
to handle large cubes without loading entirely into memory.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


class HyperspectralStreamReader:
    """Stream large Hyperspectral cubes chunk-by-chunk to avoid OOM.

    Useful for images > available RAM, e.g., 512x512x1000 = 262M pixels.
    """

    def __init__(
        self,
        data: np.ndarray,
        chunk_height: int = 64,
        chunk_width: int = 64,
    ):
        """Initialize reader.

        Parameters
        ----------
        data : np.ndarray
            Hyperspectral cube (height, width, bands).
        chunk_height : int, default=64
            Row chunk size.
        chunk_width : int, default=64
            Column chunk size.
        """
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {data.shape}")

        self.data = data
        self.height, self.width, self.bands = data.shape
        self.chunk_height = min(chunk_height, self.height)
        self.chunk_width = min(chunk_width, self.width)

    def chunks(self) -> Iterator[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Yield chunks and their spatial coordinates.

        Yields
        ------
        chunk : np.ndarray
            Subarray (chunk_h, chunk_w, bands).
        bounds : tuple
            (row_start, row_end, col_start, col_end).
        """
        for i in range(0, self.height, self.chunk_height):
            for j in range(0, self.width, self.chunk_width):
                row_end = min(i + self.chunk_height, self.height)
                col_end = min(j + self.chunk_width, self.width)

                chunk = self.data[i:row_end, j:col_end, :]
                yield chunk, (i, row_end, j, col_end)

    def chunk_count(self) -> int:
        """Total number of chunks."""
        n_row_chunks = int(np.ceil(self.height / self.chunk_height))
        n_col_chunks = int(np.ceil(self.width / self.chunk_width))
        return n_row_chunks * n_col_chunks


class HyperspectralTiler:
    """Tile hyperspectral cube for parallel processing.

    Supports overlapping tiles for convolution-based operations.
    """

    def __init__(
        self,
        data: np.ndarray,
        tile_height: int = 128,
        tile_width: int = 128,
        overlap: int = 0,
    ):
        """Initialize tiler.

        Parameters
        ----------
        data : np.ndarray
            Cube (height, width, bands).
        tile_height : int, default=128
            Tile height.
        tile_width : int, default=128
            Tile width.
        overlap : int, default=0
            Overlap pixels (for edge handling).
        """
        self.data = data
        self.height, self.width, self.bands = data.shape
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.overlap = overlap

    def tiles(self) -> Iterator[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Yield tiles with overlaps.

        Yields
        ------
        tile : np.ndarray
            Tile data.
        bounds : tuple
            Core tile bounds without overlap (row_start, row_end, col_start, col_end).
        """
        stride_h = self.tile_height - self.overlap
        stride_w = self.tile_width - self.overlap

        for i in range(0, self.height, stride_h):
            for j in range(0, self.width, stride_w):
                # Core bounds (without overlap)
                core_row_start = i
                core_row_end = min(i + stride_h, self.height)
                core_col_start = j
                core_col_end = min(j + stride_w, self.width)

                # Padded bounds (with overlap)
                pad_row_start = max(0, i - self.overlap)
                pad_row_end = min(self.height, i + self.tile_height + self.overlap)
                pad_col_start = max(0, j - self.overlap)
                pad_col_end = min(self.width, j + self.tile_width + self.overlap)

                tile = self.data[pad_row_start:pad_row_end, pad_col_start:pad_col_end, :]
                yield tile, (core_row_start, core_row_end, core_col_start, core_col_end)

    def tile_count(self) -> int:
        """Total number of tiles."""
        stride_h = self.tile_height - self.overlap
        stride_w = self.tile_width - self.overlap
        n_row_tiles = int(np.ceil((self.height - self.overlap) / stride_h))
        n_col_tiles = int(np.ceil((self.width - self.overlap) / stride_w))
        return n_row_tiles * n_col_tiles


def process_hyperspectral_chunks(
    cube: np.ndarray,
    func,
    chunk_height: int = 64,
    chunk_width: int = 64,
    **func_kwargs,
) -> np.ndarray:
    """Apply function to chunked hyperspectral cube and reassemble.

    Parameters
    ----------
    cube : np.ndarray
        Input cube (height, width, bands).
    func : callable
        Function(chunk, **kwargs) -> output.
        Should preserve spatial dimensions.
    chunk_height : int, default=64
        Chunk height.
    chunk_width : int, default=64
        Chunk width.
    **func_kwargs : dict
        Keyword arguments for func.

    Returns
    -------
    output : np.ndarray
        Processed cube (same shape as input or function-determined).
    """
    reader = HyperspectralStreamReader(cube, chunk_height, chunk_width)
    chunks = []
    bounds_list = []

    for chunk, bounds in reader.chunks():
        processed_chunk = func(chunk, **func_kwargs)
        chunks.append(processed_chunk)
        bounds_list.append(bounds)

    # Reassemble chunks
    height, width = cube.shape[:2]
    if chunks:
        output_bands = chunks[0].shape[2] if len(chunks[0].shape) == 3 else 1
        output = np.zeros((height, width, output_bands), dtype=chunks[0].dtype)

        for processed_chunk, (row_start, row_end, col_start, col_end) in zip(chunks, bounds_list):
            output[row_start:row_end, col_start:col_end, :] = processed_chunk

    return output


def estimate_memory_usage(
    height: int,
    width: int,
    bands: int,
    dtype: np.dtype = np.float32,
) -> Tuple[float, str]:
    """Estimate memory usage for hyperspectral cube.

    Parameters
    ----------
    height : int
        Image height.
    width : int
        Image width.
    bands : int
        Number of spectral bands.
    dtype : np.dtype, default=np.float32
        Data type.

    Returns
    -------
    memory : float
        Memory in MB.
    human_readable : str
        Formatted string (MB, GB, etc).
    """
    itemsize = np.dtype(dtype).itemsize
    total_bytes = height * width * bands * itemsize
    memory_mb = total_bytes / (1024 ** 2)

    if memory_mb < 1024:
        return memory_mb, f"{memory_mb:.2f} MB"
    else:
        memory_gb = memory_mb / 1024
        return memory_mb, f"{memory_gb:.2f} GB"


def recommend_chunk_size(
    cube_memory_mb: float,
    available_memory_mb: float = 1024,
    safety_factor: float = 0.5,
) -> Tuple[int, int]:
    """Recommend chunk size based on available memory.

    Parameters
    ----------
    cube_memory_mb : float
        Total cube size in MB.
    available_memory_mb : float, default=1024
        Available RAM in MB.
    safety_factor : float, default=0.5
        Keep chunk size to safety_factor * available_memory.

    Returns
    -------
    chunk_height : int
        Recommended chunk height (and width, assumed square).
    chunk_width : int
    """
    # Rough estimate: each dimension roughly sqrt(available * safety_factor / itemsize)
    chunk_size_mb = available_memory_mb * safety_factor
    # Assuming 4 bytes/pixel and 100 bands typical
    pixels_per_chunk = (chunk_size_mb * 1024 ** 2) / (4 * 100)
    chunk_dim = int(np.sqrt(pixels_per_chunk))
    # Round to power of 2 for efficiency
    chunk_dim = max(32, int(2 ** np.floor(np.log2(chunk_dim))))

    return chunk_dim, chunk_dim


__all__ = [
    "HyperspectralStreamReader",
    "HyperspectralTiler",
    "process_hyperspectral_chunks",
    "estimate_memory_usage",
    "recommend_chunk_size",
]
