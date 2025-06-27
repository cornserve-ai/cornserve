from concurrent.futures import ThreadPoolExecutor
import os
import random
import uuid

import cv2
import numpy as np
import soundfile as sf

rd = random.Random()
rd.seed(48105)
# rd.seed(time.time())
uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))

def create_dummy_video(
    height: int,
    width: int,
    duration: int,
    framerate: int,
    id: int = 0,
    bitrate: int = 3000,
    codec: str = 'mp4v',
    color_depth: int = 8,
) -> str:
    """
    Create a dummy video with specified parameters.
    
    Args:
        height: Video height in pixels
        width: Video width in pixels  
        duration: Video duration in seconds
        framerate: Frames per second
        id: Unique identifier for the video
        bitrate: Video bitrate in kbps (affects quality/file size)
        codec: Video codec ('mp4v', 'XVID', 'H264', etc.)
        color_depth: Bits per channel (8, 10, 12)
    """
    # normalize height and width
    if height < width:
        height, width = width, height
    # Create filename from parameters
    filename = f"{height}x{width}_{duration}s_{framerate}fps_{bitrate}kbps_{codec}_{color_depth}bit_{id}"
    filename += ".mp4"
    
    filepath = f"videos/{filename}"
    
    # Create videos directory if it doesn't exist
    os.makedirs("videos", exist_ok=True)

    if os.path.exists(filepath):
        return filename
    
    # Calculate total frames
    total_frames = duration * framerate
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter.fourcc(*codec)
    out = cv2.VideoWriter(filepath, fourcc, float(framerate), (width, height))
    
    # Check if VideoWriter opened successfully
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {filepath}")
    
    print(f"Creating dummy video: {filename}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Duration: {duration}s at {framerate}fps ({total_frames} frames)")
    print(f"  Bitrate: {bitrate}kbps, Codec: {codec}, Color depth: {color_depth}bit")
    
    # Pre-compute all time-independent values
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Pre-compute constants for sine calculations
    x_phase = x_coords * 0.01
    y_phase = y_coords * 0.01 + np.pi/3
    xy_phase = (x_coords + y_coords) * 0.01 + 2*np.pi/3
    
    # Pre-compute circle parameters
    radius = int(min(width, height) // 8)
    center_base_x = width // 2
    center_base_y = height // 2
    center_amplitude_x = width // 4
    center_amplitude_y = height // 4
    
    # Font setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Progress reporting interval
    progress_interval = max(1, total_frames // 20)  # Report every 5%
    
    for frame_num in range(total_frames):
        if frame_num % progress_interval == 0:
            print(f"  Processing frame {frame_num + 1}/{total_frames}", end='\r')
        
        # Pre-compute time factor and 2π multiplication
        time_factor_2pi = 2 * np.pi * frame_num / total_frames
        
        # Vectorized gradient calculation with pre-computed phases
        r = 128 + 127 * np.sin(time_factor_2pi + x_phase)
        g = 128 + 127 * np.sin(time_factor_2pi + y_phase)
        b = 128 + 127 * np.sin(time_factor_2pi + xy_phase)
        
        # Stack into BGR format and convert to uint8 in one operation
        frame = np.stack([b, g, r], axis=2).astype(np.uint8)
        
        # Pre-compute circle center
        sin_val = np.sin(time_factor_2pi)
        cos_val = np.cos(time_factor_2pi)
        center_x = int(center_base_x + center_amplitude_x * sin_val)
        center_y = int(center_base_y + center_amplitude_y * cos_val)
        
        # Add moving circle
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
        
        # Add frame counter text (less frequently for performance)
        if frame_num % 5 == 0:  # Only update text every 5 frames
            text = f"Frame {frame_num+1}/{total_frames}"
            cv2.putText(frame, text, (10, 30), font, 0.7, (0, 0, 0), 2)
        
        # Write frame
        out.write(frame)
    
    print()  # New line after progress
    
    # Release video writer
    out.release()
    print(f"Created: {filepath}")
    return filename

def create_test_videos(
    video_configs: list[tuple[int, int, int, int]],
    count: int = 10,
) -> list[str]:
    """Create a set of test videos with different parameters."""
    filenames = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        # Submit all tasks first
        futures = []
        for height, width, duration, framerate in video_configs:
            for i in range(count):
                future = executor.submit(
                    create_dummy_video, 
                    height, width, duration, framerate, id=i
                )
                futures.append(future)
        
        # Collect results after all are submitted
        for future in futures:
            filename = future.result()
            filenames.append(filename)

    return filenames

def create_dummy_audio(
    sample_rate: int,
    duration: int,
    channels: int = 2,
    bit_depth: int = 16,
    base_frequency: float = 85.5,
    id: int = 0,
) -> str:
    """
    Create a dummy WAV audio file with specified parameters.
    
    Args:
        sample_rate: Audio sample rate in Hz (e.g., 44100, 48000)
        duration: Audio duration in seconds
        channels: Number of audio channels (1=mono, 2=stereo, etc.)
        id: Unique identifier for the audio file
        bit_depth: Bits per sample (16, 24, 32)
        base_frequency: Base frequency in Hz for tone generation
    """
    # Create filename from parameters
    channel_str = "mono" if channels == 1 else f"{channels}ch"
    filename = f"{sample_rate}Hz_{duration}s_{channel_str}_{bit_depth}bit_{base_frequency}Hz_{id}"
    filename += ".wav"
    
    filepath = f"audios/{filename}"
    
    # Create audios directory if it doesn't exist
    os.makedirs("audios", exist_ok=True)

    if os.path.exists(filepath):
        return filename
    
    # Calculate total samples
    total_samples = duration * sample_rate
    
    print(f"Creating dummy audio: {filename}")
    print(f"  Sample rate: {sample_rate}Hz")
    print(f"  Duration: {duration}s ({total_samples:,} samples)")
    print(f"  Channels: {channels}, Bit depth: {bit_depth}bit, Format: WAV")
    print(f"  Base frequency: {base_frequency}Hz")
    
    # Generate time array
    t = np.linspace(0, duration, total_samples, dtype=np.float32)
    
    # Create audio data
    audio_data = np.zeros((total_samples, channels), dtype=np.float32)
    
    # Progress reporting interval
    progress_interval = max(1, channels // 5) if channels > 5 else 1
    
    for ch in range(channels):
        if ch % progress_interval == 0:
            print(f"  Generating channel {ch + 1}/{channels}", end='\r')
        
        # Create different frequency for each channel
        frequency = base_frequency * (1 + ch * 0.25)  # Each channel slightly higher
        
        # Generate complex waveform with multiple harmonics
        wave = np.zeros_like(t)
        
        # Add fundamental frequency
        wave += 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add harmonics with decreasing amplitude
        wave += 0.25 * np.sin(2 * np.pi * frequency * 2 * t)  # 2nd harmonic
        wave += 0.125 * np.sin(2 * np.pi * frequency * 3 * t)  # 3rd harmonic
        
        # Add slow amplitude modulation for interest
        modulation_freq = 0.5 + ch * 0.1  # Different modulation per channel
        amplitude_mod = 0.8 + 0.2 * np.sin(2 * np.pi * modulation_freq * t)
        wave *= amplitude_mod
        
        # Add frequency sweep for more dynamic content
        sweep_rate = 50 + ch * 10  # Different sweep rate per channel
        frequency_sweep = frequency + sweep_rate * np.sin(2 * np.pi * 0.1 * t)
        wave += 0.1 * np.sin(2 * np.pi * frequency_sweep * t)
        
        # Apply gentle fade in/out to avoid clicks
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            wave[:fade_samples] *= fade_in
            wave[-fade_samples:] *= fade_out
        
        # Normalize to prevent clipping
        wave = wave * 0.7  # Keep some headroom
        
        audio_data[:, ch] = wave
    
    print()  # New line after progress
    
    # Determine subtype based on bit depth
    subtype_map = {
        16: 'PCM_16',
        24: 'PCM_24', 
        32: 'PCM_32'
    }
    subtype = subtype_map.get(bit_depth, 'PCM_16')
    
    # Write WAV audio file
    try:
        sf.write(filepath, audio_data, sample_rate, subtype=subtype, format='WAV')
        print(f"Created: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to write audio file {filepath}: {e}")
    
    return filename


def create_test_audios(
    audio_configs: list[tuple[int, int, int]],  # (sample_rate, duration, channels)
    count: int = 10,
    base_frequency: float = 440.0,
) -> list[str]:
    """
    Create a set of test WAV audio files with different parameters.
    
    Args:
        audio_configs: List of tuples containing (sample_rate, duration, channels)
        count: Number of audio files to create for each configuration
        base_frequency: Base frequency for tone generation
    """
    filenames = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        # Submit all tasks first
        futures = []
        for sample_rate, duration, channels in audio_configs:
            for i in range(count):
                future = executor.submit(
                    create_dummy_audio,
                    sample_rate,
                    duration,
                    channels,
                    base_frequency=base_frequency,
                    id=i,
                )
                futures.append(future)
        
        # Collect results after all are submitted
        for future in futures:
            filename = future.result()
            filenames.append(filename)

    return filenames


def create_dummy_image(
    height: int,
    width: int,
    id: int = 0,
    format: str = 'png',
    color_depth: int = 8,
    pattern: str = 'gradient',
    quality: int = 95,
) -> str:
    """
    Create a dummy image with specified parameters.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        id: Unique identifier for the image
        format: Image format ('png', 'jpg', 'jpeg', 'bmp', 'tiff')
        color_depth: Bits per channel (8, 16)
        pattern: Visual pattern type ('gradient', 'checkerboard', 'noise', 'mandala')
        quality: JPEG quality (1-100, only applies to JPEG format)
    """
    # Normalize height and width
    if height < width:
        height, width = width, height
    
    # Create filename from parameters
    filename = f"{height}x{width}_{pattern}_{color_depth}bit_{format}_{id}"
    filename += f".{format.lower()}"
    
    filepath = f"images/{filename}"
    
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    if os.path.exists(filepath):
        return filename
    
    print(f"Creating dummy image: {filename}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Format: {format.upper()}, Color depth: {color_depth}bit, Pattern: {pattern}")
    if format.lower() in ['jpg', 'jpeg']:
        print(f"  JPEG Quality: {quality}")
    
    # Generate image based on pattern
    if pattern == 'gradient':
        image = create_gradient_pattern(height, width, id)
    elif pattern == 'checkerboard':
        image = create_checkerboard_pattern(height, width, id)
    elif pattern == 'noise':
        image = create_noise_pattern(height, width, id)
    elif pattern == 'mandala':
        image = create_mandala_pattern(height, width, id)
    else:
        # Default to gradient
        image = create_gradient_pattern(height, width, id)
    
    # Handle color depth
    if color_depth == 16:
        # Convert to 16-bit (note: most formats still save as 8-bit)
        image = (image.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
    else:
        image = image.astype(np.uint8)
    
    # Add text overlay with image info
    add_text_overlay(image, filename, width, height, id, color_depth)
    
    # Save image with appropriate parameters
    save_params = []
    if format.lower() in ['jpg', 'jpeg']:
        save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif format.lower() == 'png':
        save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # 0-9, 3 is good balance
    elif format.lower() == 'tiff':
        save_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]  # LZW compression
    
    # Convert back to 8-bit for saving (OpenCV limitation)
    if color_depth == 16:
        save_image = (image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    else:
        save_image = image
    
    success = cv2.imwrite(filepath, save_image, save_params)
    
    if not success:
        raise RuntimeError(f"Failed to save image: {filepath}")
    
    print(f"Created: {filepath}")
    return filename


def create_gradient_pattern(height: int, width: int, id: int) -> np.ndarray:
    """Create a colorful gradient pattern."""
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Use id to vary the pattern
    phase_offset = id * np.pi / 4
    
    # Create different gradient patterns
    x_phase = x_coords * 0.01 + phase_offset
    y_phase = y_coords * 0.01 + phase_offset + np.pi/3
    xy_phase = (x_coords + y_coords) * 0.005 + phase_offset + 2*np.pi/3
    
    # Generate RGB channels with different patterns
    r = 128 + 127 * np.sin(x_phase)
    g = 128 + 127 * np.sin(y_phase)
    b = 128 + 127 * np.sin(xy_phase)
    
    # Stack into BGR format for OpenCV
    image = np.stack([b, g, r], axis=2)
    
    # Add some radial variation
    center_x, center_y = width // 2, height // 2
    radius_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_radius = np.sqrt(center_x**2 + center_y**2)
    radial_factor = 0.8 + 0.4 * np.sin(radius_map / max_radius * 4 * np.pi + phase_offset)
    
    # Apply radial modulation
    image = image * radial_factor[:, :, np.newaxis]
    
    return np.clip(image, 0, 255)


def create_checkerboard_pattern(height: int, width: int, id: int) -> np.ndarray:
    """Create a colorful checkerboard pattern."""
    # Vary square size based on id
    square_size = 20 + (id % 10) * 5
    
    # Create checkerboard base
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    checker = ((x_coords // square_size) + (y_coords // square_size)) % 2
    
    # Create color variations based on id
    colors = [
        ([255, 100, 100], [100, 255, 100]),  # Red-Green
        ([100, 100, 255], [255, 255, 100]),  # Blue-Yellow
        ([255, 100, 255], [100, 255, 255]),  # Magenta-Cyan
        ([255, 150, 0], [150, 0, 255]),      # Orange-Purple
    ]
    
    color_pair = colors[id % len(colors)]
    
    # Create image
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[checker == 0] = color_pair[0]
    image[checker == 1] = color_pair[1]
    
    # Add gradient overlay
    gradient = np.linspace(0.7, 1.3, width)
    image = image * gradient[np.newaxis, :, np.newaxis]
    
    return np.clip(image, 0, 255)


def create_noise_pattern(height: int, width: int, id: int) -> np.ndarray:
    """Create a colored noise pattern."""
    # Set seed for reproducible noise based on id
    np.random.seed(id)
    
    # Generate different types of noise for each channel
    noise_r = np.random.normal(128, 40, (height, width))
    noise_g = np.random.normal(128, 40, (height, width))
    noise_b = np.random.normal(128, 40, (height, width))
    
    # Add some structure to the noise
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    structure = 50 * np.sin(x_coords * 0.02 + id) * np.sin(y_coords * 0.02 + id)
    
    # Combine noise with structure
    r = noise_r + structure
    g = noise_g + structure * 0.7
    b = noise_b + structure * 1.3
    
    # Stack into BGR format
    image = np.stack([b, g, r], axis=2)
    
    return np.clip(image, 0, 255)


def create_mandala_pattern(height: int, width: int, id: int) -> np.ndarray:
    """Create a mandala-like pattern."""
    center_x, center_y = width // 2, height // 2
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Convert to polar coordinates
    dx = x_coords - center_x
    dy = y_coords - center_y
    radius = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    
    # Create mandala pattern with id variation
    symmetry = 6 + (id % 8)  # 6-13 fold symmetry
    radial_freq = 0.1 + (id % 5) * 0.02
    angular_freq = symmetry
    
    # Generate pattern
    radial_pattern = np.sin(radius * radial_freq + id)
    angular_pattern = np.sin(theta * angular_freq + id * np.pi / 4)
    combined_pattern = radial_pattern * angular_pattern
    
    # Create color mapping
    r = 128 + 127 * combined_pattern
    g = 128 + 127 * np.sin(combined_pattern + 2*np.pi/3)
    b = 128 + 127 * np.sin(combined_pattern + 4*np.pi/3)
    
    # Add radial fade
    max_radius = min(center_x, center_y)
    fade = np.clip(1 - radius / max_radius, 0, 1)
    
    # Apply fade
    r = r * fade + 20 * (1 - fade)
    g = g * fade + 20 * (1 - fade)
    b = b * fade + 20 * (1 - fade)
    
    # Stack into BGR format
    image = np.stack([b, g, r], axis=2)
    
    return np.clip(image, 0, 255)


def add_text_overlay(image: np.ndarray, filename: str, width: int, height: int, id: int, color_depth: int):
    """Add text overlay with image information."""
    if image.dtype == np.uint16:
        # Convert to 8-bit for text overlay, then convert back
        img_8bit = (image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
        add_text_to_8bit(img_8bit, filename, width, height, id, color_depth)
        # Convert back to 16-bit
        image[:] = (img_8bit.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
    else:
        add_text_to_8bit(image, filename, width, height, id, color_depth)


def add_text_to_8bit(image: np.ndarray, filename: str, width: int, height: int, id: int, color_depth: int):
    """Add text overlay to 8-bit image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(width, height) / 1000)
    thickness = max(1, int(font_scale * 2))
    
    # Main info text
    text = f"{width}x{height} ID:{id} {color_depth}bit"
    cv2.putText(image, text, (10, 30), font, font_scale, (255, 255, 255), thickness + 1)
    cv2.putText(image, text, (10, 30), font, font_scale, (0, 0, 0), thickness)
    
    # Filename text at bottom
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    y_pos = height - 15
    cv2.putText(image, base_filename, (10, y_pos), font, font_scale * 0.7, (255, 255, 255), thickness + 1)
    cv2.putText(image, base_filename, (10, y_pos), font, font_scale * 0.7, (0, 0, 0), thickness)


def create_test_images(
    image_configs: list[tuple[int, int, str]],  # (height, width, pattern)
    count: int = 10,
    format: str = 'png',
) -> list[str]:
    """
    Create a set of test images with different parameters.
    
    Args:
        image_configs: List of tuples containing (height, width, pattern)
        count: Number of images to create for each configuration
        format: Image format for all generated images
        
    Returns:
        filenames: List of generated filenames
    """
    filenames = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        # Submit all tasks first
        futures = []
        for height, width, pattern in image_configs:
            for i in range(count):
                future = executor.submit(
                    create_dummy_image, 
                    height, width, id=i, format=format, pattern=pattern
                )
                futures.append(future)
        
        # Collect results after all are submitted
        for future in futures:
            filename = future.result()
            filenames.append(filename)

    return filenames

def get_benchmark_filenames(max_mm_count: int) -> tuple[list, list, list]:
    """ Get benchmark filenames for videos, audios, and images.

    Args:
        max_mm_count: Maximum number of multimedia files to create for each type
    """
    # width, height, fps, duration
    video_configs = [(1920, 1080, 5, 30)]
    video_filenames = create_test_videos(video_configs, count=max_mm_count)
    # sampling_rate, duration, channels
    audio_configs = [(44100, 60, 2)]
    audio_filenames = create_test_audios(audio_configs, count=max_mm_count)
    # width, height, type
    image_configs = [(1920, 1080, 'mandala')]
    image_filenames = create_test_images(image_configs, count=max_mm_count)
    return video_filenames, audio_filenames, image_filenames

def get_benchmark_configs() -> tuple[list, list, list]:
    """
    Get benchmark configurations for videos, audios, and images.

    Returns:
        video_configs: List of tuples containing (width, height, fps, duration)
        audio_configs: List of tuples containing (sampling_rate, duration, channels)
        image_configs: List of tuples containing (width, height, type)
    """
    # width, height, fps, duration
    video_configs = [(1920, 1080, 5, 30)]
    # sampling_rate, duration, channels
    audio_configs = [(44100, 60, 2)]
    # width, height, type
    image_configs = [(1920, 1080, 'mandala')]
    return video_configs, audio_configs, image_configs
    

if __name__ == "__main__":
    # Example configurations: (height, width, pattern)
    image_configs = [
        (1080, 1920, 'gradient'),
        (2160, 3840, 'mandala'),
    ]
    
    filenames = create_test_images(image_configs, count=3, format='png')
    print(f"Created {len(filenames)} test images")
